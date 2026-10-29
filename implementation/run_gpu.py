"""Finite single-GPU smoke, real resume, baseline training and separate evaluation.

Run only in a scheduled ACP worker, from the prepared source directory.
"""
from __future__ import annotations
import argparse
import datetime as dt
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "uni"
CGROUP = Path("/sys/fs/cgroup")
GIB = 1024**3


def read_cgroup(name):
    path = CGROUP / name
    return path.read_text().strip() if path.is_file() else None


def resources():
    record = {"utc":dt.datetime.now(dt.timezone.utc).isoformat(),
              "cgroup_memory_current":read_cgroup("memory.current"),
              "cgroup_memory_max":read_cgroup("memory.max"),
              "memory_events":read_cgroup("memory.events")}
    proc = subprocess.run(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu",
                           "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15)
    record["nvidia_smi"] = proc.stdout.strip()
    record["nvidia_smi_exit"] = proc.returncode
    return record


def run(name, arguments, deadline, stage_limit):
    folder = ROOT / "runs" / name
    folder.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-u", *arguments]
    launch = {"stage":name,"command":command,"started_at":dt.datetime.now(dt.timezone.utc).isoformat(),
              "maximum_seconds":min(stage_limit,max(0,int(deadline-time.monotonic())))}
    (folder / "launch.json").write_text(json.dumps(launch, indent=2)+"\n")
    print("SUPERMIMIC_STAGE_START "+json.dumps(launch),flush=True)
    start = time.monotonic()
    last_progress = start
    stop_reason = None
    with (folder / "stdout.txt").open("w", buffering=1) as log, (folder / "resources.jsonl").open("a", buffering=1) as trace:
        child = subprocess.Popen(command, cwd=SOURCE, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while child.poll() is None:
                if time.monotonic() > deadline or time.monotonic()-start > stage_limit:
                    stop_reason = "finite_wall_budget"
                row = resources()
                trace.write(json.dumps(row)+"\n")
                if time.monotonic()-last_progress>=60:
                    print("SUPERMIMIC_PROGRESS "+json.dumps({"stage":name,"seconds":time.monotonic()-start,
                          "resources":row,"log_tail":log_tail(folder/"stdout.txt")[-1500:]}),flush=True)
                    last_progress=time.monotonic()
                current, maximum = row["cgroup_memory_current"], row["cgroup_memory_max"]
                if current and maximum and maximum != "max" and int(maximum)-int(current)<6*GIB:
                    stop_reason = "host_memory_reserve_6GiB"
                if row["nvidia_smi_exit"] == 0:
                    fields = [x.strip() for x in row["nvidia_smi"].split(",")]
                    if len(fields)==7 and float(fields[5])<2048:
                        stop_reason = "GPU_memory_reserve_2GiB"
                if stop_reason:
                    os.killpg(child.pid,signal.SIGTERM)
                    try: child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid,signal.SIGKILL)
                        child.wait()
                    break
                time.sleep(5)
        finally:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try: child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL)
                    child.wait()
    receipt = {**launch,"elapsed_seconds":time.monotonic()-start,"returncode":child.returncode,
               "stop_reason":stop_reason,"ended_at":dt.datetime.now(dt.timezone.utc).isoformat()}
    (folder / "completion.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print("SUPERMIMIC_STAGE_END "+json.dumps(receipt),flush=True)
    print(log_tail(folder / "stdout.txt"),flush=True)
    if child.returncode or stop_reason:
        raise RuntimeError(f"Stage {name} stopped: {receipt}")


def log_tail(path):
    with path.open("rb") as stream:
        stream.seek(max(0,path.stat().st_size-6000))
        return stream.read().decode(errors="replace")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",choices=("acceptance","full"),default="acceptance")
    args=parser.parse_args()
    import torch
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    assert "H100" in torch.cuda.get_device_name(0)
    probe=torch.rand((128,128),device="cuda")
    assert bool(torch.isfinite(probe@probe).all())
    import warp as wp
    wp.init()
    assert wp.get_device("cuda:0").is_cuda
    metadata={"torch":torch.__version__,"torch_cuda":torch.version.cuda,"GPU":torch.cuda.get_device_name(0),
              "unilab":importlib.metadata.version("unilab"),"warp":importlib.metadata.version("warp-lang"),
              "mujoco":importlib.metadata.version("mujoco"),"cgroup_cpu_max":read_cgroup("cpu.max"),
              "cgroup_memory_max":read_cgroup("memory.max"),"resources":resources()}
    (ROOT/"evidence/gpu-environment.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print("SUPERMIMIC_GPU_PREFLIGHT_PASS "+json.dumps(metadata),flush=True)
    # Initial smoke retains default network widths, BF16 and 32-step BPTT.
    # Parallelism, batch counts and warmup are reduced solely for this acceptance.
    small=["algo.num_envs=128","algo.buffer_steps=128","algo.wm_batch=128","algo.pi_batch=128",
           "algo.wm_batches=2","algo.wm_only_iters=2","algo.warmup_steps=16","algo.critic_epochs=2",
           "algo.critic_minibatches=2","algo.compile=false","algo.ckpt_every=5","algo.log_every=1"]
    deadline=time.monotonic()+(2700 if args.mode=="acceptance" else 19800)
    smoke=ROOT/"runs/smoke-policy"
    if not (ROOT/"evidence/acceptance.json").exists():
        if smoke.exists(): raise RuntimeError("Existing smoke directory requires explicit inspection; refusing overwrite")
        run("smoke",[str(ROOT/"train_audited.py"),*small,"algo.iterations=5",f"training.log_dir={smoke}"],deadline,1200)
        run("resume",[str(ROOT/"train_audited.py"),*small,"algo.iterations=8","algo.resume=true",f"algo.load_run={smoke}","algo.checkpoint=last.pt"],deadline,900)
        first=json.loads((smoke/"update-proof-000005.json").read_text())
        resumed=json.loads((smoke/"update-proof-000008.json").read_text())
        assert first["passed"] and resumed["passed"] and resumed["before"]["iteration"]==5
        for key in ("wm","policy","critic"):
            assert first["after"][key]==resumed["before"][key]
        for key in ("wm_optimizer","policy_optimizer","critic_optimizer"):
            assert first["after"][key]==resumed["before"][key]
        (ROOT/"evidence/acceptance.json").write_text(json.dumps({"passed":True,"smoke":first,"resume":resumed},indent=2)+"\n")
        print("SUPERMIMIC_ACCEPTANCE_PASS",flush=True)
    if args.mode=="acceptance": return
    accepted=json.loads((ROOT/"evidence/acceptance.json").read_text())
    assert accepted["passed"]
    baseline=ROOT/"runs/baseline-seed0"
    if baseline.exists(): raise RuntimeError("Existing baseline requires explicit resume; refusing overwrite")
    run("training",[str(ROOT/"train_audited.py"),"algo.iterations=12000",f"training.log_dir={baseline}"],deadline,16200)
    run("evaluation",[str(ROOT/"evaluate_audited.py"),"--checkpoint",str(baseline/"ckpt_012000.pt"),
                       "--output",str(ROOT/"runs/baseline-evaluation.json")],deadline,2400)
    print("SUPERMIMIC_BASELINE_COMPLETE",flush=True)


if __name__=="__main__":
    main()
