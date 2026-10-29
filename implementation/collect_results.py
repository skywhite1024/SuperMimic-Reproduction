"""Freeze an allowlisted baseline result bundle after the scheduled job exits."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import tarfile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    assert str(root) == "/mnt/afs/USER_ID/SuperMimic/lafan1-20261024"
    acceptance = read(root / "evidence/acceptance.json")
    assert acceptance["passed"]
    training = root / "runs/baseline-seed0"
    proof = read(training / "update-proof-012000.json")
    evaluation = read(root / "runs/baseline-evaluation.json")
    assert proof["passed"] and proof["after"]["iteration"] == 12000
    assert proof["clips"] == 40 and len(proof["transitions_by_clip"]) == 40
    assert all(value > 0 for value in proof["transitions_by_clip"].values())
    assert len(evaluation["per_clip"]) == 40
    clip_rows=list(evaluation["per_clip"].values())
    assert all(math.isfinite(row["norm_return"]) and row["normalization_denominator"]>0 for row in clip_rows)
    assert evaluation["n_failed"]==sum(bool(row["failed"]) for row in clip_rows)
    assert abs(sum(row["norm_return"] for row in clip_rows)/40-evaluation["norm_return"])<1e-5
    assert abs(sum(row["reward_sum"] for row in clip_rows)/sum(row["normalization_denominator"] for row in clip_rows)-evaluation["norm_return_framew"])<1e-5
    checkpoint = training / "ckpt_012000.pt"
    assert sha256(checkpoint) == evaluation["checkpoint_sha256"]
    records = {}
    chosen = set()

    def add(path):
        path = Path(path)
        if path.is_file():
            resolved = path.resolve()
            assert resolved.is_relative_to(root) and not path.is_symlink()
            chosen.add(path.relative_to(root).as_posix())

    for name in ("environment.json", "headless-render.json", "gpu-environment.json", "acceptance.json"):
        add(root / "evidence" / name)
    for name in ("run_gpu.py", "run_gpu_v2.py", "run_gpu_v3.py", "train_audited_v2.py", "evaluate_audited_v2.py", "deploy_runtime_v2.py"):
        add(root / name)
    for name in ("args.json", "train_log.csv", "update-proof-012000.json", "ckpt_012000.pt", "last.pt"):
        add(training / name)
    for pattern in ("*.yaml", "*.yml", "*.json", "*.csv", "events.out.tfevents.*"):
        for path in training.rglob(pattern):
            add(path)
    add(root / "runs/baseline-evaluation.json")
    for name in ("smoke", "resume", "training", "evaluation", "video-0", "video-1", "video-2"):
        folder = root / "runs" / name
        completion = folder / "completion.json"
        if not completion.exists():
            assert name == "video-2" and not folder.exists()
            continue
        row = read(completion)
        assert row["returncode"] == 0 and row["stop_reason"] is None
        samples = [json.loads(line) for line in (folder / "resources.jsonl").read_text(encoding="utf-8").splitlines()]
        host = [int(s["cgroup_memory_current"]) for s in samples if s["cgroup_memory_current"]]
        used, utilization = [], []
        for sample in samples:
            fields = [x.strip() for x in sample["nvidia_smi"].split(",")]
            if sample["nvidia_smi_exit"] == 0 and len(fields) == 7:
                used.append(float(fields[4]))
                utilization.append(float(fields[6]))
        records[name] = {"elapsed_seconds":row["elapsed_seconds"], "samples":len(samples),
                         "sampled_host_peak_bytes":max(host,default=0),
                         "sampled_gpu_peak_mib":max(used,default=0),
                         "sampled_gpu_utilization_mean":sum(utilization)/max(1,len(utilization)),
                         "sampling_seconds":5,"peak_is_sampled_not_continuous":True}
        for filename in ("launch.json", "completion.json", "resources.jsonl", "stdout.txt"):
            add(folder / filename)
    for path in (root / "runs/smoke-policy").glob("update-proof-*.json"):
        add(path)
    videos = list((root / "runs/videos").glob("*.mp4"))
    assert 2 <= len(videos) <= 3 and all(p.stat().st_size > 0 for p in videos)
    for path in videos:
        add(path)
    summary = {"created_at":dt.datetime.now(dt.timezone.utc).isoformat(),"status":"baseline_complete",
               "community_baseline":True,"strict_paper_reproduction":False,
               "normalized_return_macro":evaluation["norm_return"],
               "normalized_return_frame_weighted":evaluation["norm_return_framew"],
               "failed_clips":evaluation["n_failed"],"clips":40,"iterations":12000,
               "reference_target":{"macro_at_least":0.82,"failed_clips":0,
                                   "met":evaluation["norm_return"]>=0.82 and evaluation["n_failed"]==0},
               "torch_allocator_peak_allocated_bytes":proof["cuda_peak_allocated_bytes"],
               "torch_allocator_peak_reserved_bytes":proof["cuda_peak_reserved_bytes"],
               "torch_allocator_excludes_non_torch_gpu_allocations":True,
               "checkpoint_sha256":evaluation["checkpoint_sha256"],"stages":records,
               "resource_release_verified":False,
               "boundary":"The platform terminal state and CCI/DNAT cleanup require separate verification."}
    (root / "result-summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    add(root / "result-summary.json")
    manifest = {name:{"bytes":(root/name).stat().st_size,"sha256":sha256(root/name)} for name in sorted(chosen)}
    (root / "result-manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    archive = root / "supermimic-lafan1-results.tar.gz"
    assert not archive.exists(), "Refusing to replace an earlier frozen bundle"
    with tarfile.open(archive,"w:gz",compresslevel=3) as tar:
        for name in sorted(chosen | {"result-manifest.json"}):
            tar.add(root/name,arcname=name,recursive=False)
    receipt={"archive":str(archive),"bytes":archive.stat().st_size,"sha256":sha256(archive),
             "files":len(manifest)+1,"payload_bytes":sum(x["bytes"] for x in manifest.values())}
    (root / "result-transfer-receipt.json").write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
    print("SUPERMIMIC_RESULT_SUMMARY "+json.dumps({key:summary[key] for key in (
        "normalized_return_macro","normalized_return_frame_weighted","failed_clips","clips","iterations","reference_target")}),flush=True)
    print("SUPERMIMIC_RESULTS_FROZEN "+json.dumps(receipt),flush=True)


if __name__ == "__main__":
    main()
