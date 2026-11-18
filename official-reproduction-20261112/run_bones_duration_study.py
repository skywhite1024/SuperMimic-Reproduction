"""Run the four fixed-budget duration cases, with one training per allocated GPU."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from prepare_bones_duration import CASES, choose_subsets, copy_pack, read_identity
from push_recovery_contract import sha256, write_json
from run_lafan_learning_curve import execute, verify_training

SOURCE = Path(__file__).resolve().parent
STEPS = 16001
INTERVAL = 4000
POINTS = (4001, 8001, 16001)
TARGET = .90


def read(path):
    return json.loads(path.read_bytes())


def run_case(case, pack, common, output):
    pack_report = read(pack.with_name(pack.name + "-duration-pack.json"))
    common_report = read(common.with_name(common.name + "-duration-pack.json"))
    output.mkdir(exist_ok=False)
    train = output / "training"
    report = {"status": "training", "case": case, "pack": pack_report, "points": [],
              "nominal_threshold": TARGET, "real_envs": 4096, "same_seed": 42,
              "fixed_probe": common_report}
    write_json(output / "case-result.json", report)
    execute("run_lafan_pilot.py", ["--motions", pack, "--output", train,
            "--corpus", "bones-duration-proxy", "--steps", STEPS, "--num-envs", 4096,
            "--seed", 42, "--training-profile", "paper-a5-single-gpu", "--goal-backend", "eager",
            "--checkpoint-interval", INTERVAL, "--keep-checkpoint-series", "--record-checkpoint-times",
            "--record-learner-counts"], output / "train.log", 5400, stream_progress=True)
    timings = verify_training(train, STEPS, INTERVAL)
    report["status"] = "evaluating"
    for step in POINTS:
        ev = output / f"eval-{step:09d}"
        arguments = ["--score-state-forecasts"] if step == STEPS else []
        point = {"collections": step, "real_simulator_transitions": step * 4096}
        try:
            execute("evaluate_motion_corpus_v2.py", ["--run", train,
                "--checkpoint-collection-step", step, "--motions", common, "--output", ev,
                "--expected-clips", 64, "--num-envs", 64, "--order", "index", "--seed", 42,
                *arguments], output / f"{ev.name}.log", 600)
            execute("audit_corpus_result.py", ["--result", ev / "result.json",
                "--protocol", ev / "protocol.json", "--motions", common, "--expected-clips", 64,
                "--require-trajectories", "--output", ev / "accounting.json"],
                output / f"{ev.name}-accounting.log", 120)
            execute("audit_saved_initial_states_v2.py", ["--evaluation", ev, "--motions", common,
                "--expected-clips", 64, "--output", ev / "audit"], output / f"{ev.name}-audit.log", 120)
            accounting = read(ev / "accounting.json")
            protocol = read(ev / "protocol.json")
            checkpoint = protocol["checkpoint_selection"]
            if (protocol["motion_files"] != common_report["files"] or checkpoint["evaluated_collection_steps"] != step
                    or checkpoint["parent_result_sha256"] != sha256(train / "result.json")):
                raise RuntimeError("Checkpoint or evaluation motion pack differs")
            t = next(v for v in timings["checkpoints"] if v["collection_steps"] == step)
            point.update(status="complete", checkpoint_wall_seconds=t["retained_snapshot_complete"]["seconds_since_runner_run_entry"],
                         success=accounting["successes"], clips=accounting["clips"],
                         mean_normalized_return=accounting["mean_normalized_full_motion_return"])
            if step == STEPS:
                q = read(ev / "state-forecast.json")
                point.update(state_forecast_receipt_sha256=sha256(ev / "state-forecast.json"),
                             forecast_windows=q["windows"],
                             normalized_state_mse_step8=q["eighth_transition_normalized_state_mse"])
        except Exception as error:
            point.update(status="failed", error=f"{type(error).__name__}: {error}")
        report["points"].append(point)
        write_json(output / "case-result.json", report)
        print("BONES_DURATION_POINT " + json.dumps({"case": case, **point}), flush=True)
    report["status"] = ("complete" if all(p["status"] == "complete" for p in report["points"])
                        else "incomplete")
    report["first_sample_at_threshold"] = next(
        (p for p in report["points"] if p["status"] == "complete" and p["mean_normalized_return"] >= TARGET),
        None)
    write_json(output / "case-result.json", report)
    return report["status"] == "complete"


def orchestrate(args):
    import torch

    allocated = torch.cuda.device_count()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = visible.split(",") if visible else [str(i) for i in range(allocated)]
    if allocated != 2 or len(devices) != allocated or len(set(devices)) != 2:
        raise RuntimeError("This schedule requires two visible H100s within one worker")
    args.output.mkdir(parents=True, exist_ok=False)
    motions = args.output / "motions"
    cases = args.output / "cases"
    cases.mkdir()
    filter_result = read(args.filter_report)
    if filter_result["status"] != "full_proxy_filter_complete":
        raise ValueError("Full frozen feasibility mask is required")
    names, ends, lengths, keep = read_identity(args.raw, filter_result, args.mask)
    raw_identity = read(args.raw_manifest)["files"]
    for name, value in raw_identity.items():
        if sha256(args.raw / name) != value:
            raise RuntimeError(f"Physical raw pack differs from the prior BONES evaluation: {name}")
    subsets = choose_subsets(lengths, keep)
    reports = {}
    for case in (*CASES, "common64"):
        reports[case] = copy_pack(args.raw, motions / case, names, ends, lengths, case, subsets[case],
                                  sha256(args.filter_report))
    if not set(subsets["common64"]).issubset(set(subsets["1h"])):
        raise RuntimeError("The common training-set probe changed membership")
    report = {
        "status": "training", "collection_budget_each": STEPS, "evaluation_collections": POINTS,
        "available_reference_hours": reports["full"]["hours_at_50hz"],
        "paper_exact_duration_sweep": False, "paper_exact_membership": False,
        "filter_sha256": sha256(args.filter_report), "raw_manifest_sha256": sha256(args.raw_manifest),
        "requested_gpus": 2, "packs": reports, "case_return_codes": {},
    }
    write_json(args.output / "study.json", report)
    lanes = (("1h", "7h"), ("40h", "full"))
    for wave in lanes:
        processes = {}
        for gpu, case in enumerate(wave):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=devices[gpu])
            cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "case", "--key", case,
                   "--pack", str(motions / case), "--common", str(motions / "common64"),
                   "--output", str(cases / case)]
            print(f"BONES_DURATION_TRAINING_START case={case} logical_gpu={gpu}", flush=True)
            processes[case] = subprocess.Popen(cmd, env=env)
        for case, actor in processes.items():
            rc = actor.wait()
            print(f"BONES_DURATION_CASE_EXIT {case} {rc}", flush=True)
            report["case_return_codes"][case] = rc
            write_json(args.output / "study.json", report)
    report["status"] = ("complete" if all(rc == 0 for rc in report["case_return_codes"].values())
                        else "incomplete")
    write_json(args.output / "study.json", report)
    return report["status"] == "complete"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    dispatch = commands.add_parser("study")
    for arg in ("raw", "raw-manifest", "filter-report", "mask", "output"):
        dispatch.add_argument("--" + arg, type=Path, required=True)
    child = commands.add_parser("case")
    child.add_argument("--key", choices=CASES, required=True)
    for arg in ("pack", "common", "output"):
        child.add_argument("--" + arg, type=Path, required=True)
    args = parser.parse_args()
    if args.command == "study":
        ok = orchestrate(args)
    else:
        ok = run_case(args.key, args.pack, args.common, args.output)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
