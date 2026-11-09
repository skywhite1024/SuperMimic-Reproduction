"""New-directory, evaluation-only J runner; reuse frozen I checkpoint bytes.

The caller supplies existing I evaluation-checkpoints.json and runtime directory.
No training, download, dataset modification or cloud submission is performed.
Run under an external 1800-second process timeout on one visible CUDA device.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-spec", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-seconds", type=float, default=1700)
    parser.add_argument("--per-model-seconds", type=float, default=500)
    args = parser.parse_args()
    if not (0 < args.max_seconds <= 1700 and 0 < args.per_model_seconds <= 500):
        parser.error("Finite limit: total <=1700 seconds, each model <=500 seconds")
    started = time.monotonic()
    deadline = started+args.max_seconds
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    spec = json.loads(args.source_spec.read_text(encoding="utf-8-sig"))
    models = {model["id"]: model for model in spec["models"]}
    if set(models) != {"parent", "f0", "f1"} or not all(model.get("available") for model in models.values()):
        raise ValueError("Require I's three available parent/F0/F1 models")
    manifest = json.loads(Path(spec["manifest"]).read_text(encoding="utf-8-sig"))
    old = [dict(row) for row in manifest["clips"] if row["group"] == "old"]
    if len(old) != 40:
        raise ValueError("I manifest must contain exactly 40 old clips")
    # Preserve I's fixed ordering. Do not choose, sort or truncate by outcomes.
    for row in old:
        row["path"] = str((Path(spec["manifest"]).parent/row["path"]).resolve())
        if sha(row["path"]) != row["sha256"]:
            raise ValueError("Frozen old CSV hash mismatch: "+row["path"])
    old_manifest = args.output/"old40-manifest.json"
    write(old_manifest, {**manifest, "reference_variant": "I_frozen_old40_only", "clips": old,
                         "independent_E_star": False, "training_performed": False})
    runtime = args.output/"runtime"
    runtime.mkdir()
    for name in ("evaluate.py", "sports_runtime.py"):
        shutil.copy2(args.runtime_dir/name, runtime/name)
    uni = Path(os.environ.get("SUPERMIMIC_UNI", str(args.runtime_dir.resolve().parents[1]/"implementation/uni"))).resolve()
    if not (uni/"src/unisupermimic").is_dir():
        raise ValueError("Specify SUPERMIMIC_UNI pointing at I's already verified uni runtime")
    child_env = {**os.environ, "SUPERMIMIC_UNI": str(uni)}
    evaluator = Path(__file__).with_name("evaluate_batch.py")
    shutil.copy2(evaluator, runtime/evaluator.name)
    checkpoint_dir = args.output/"checkpoints"
    checkpoint_dir.mkdir()
    report = {"status": "preparing", "training_performed": False, "new_policy_updates": 0,
              "source_spec": str(args.source_spec.resolve()), "source_spec_sha256": sha(args.source_spec),
              "device": args.device, "backend": "mujoco" if args.device == "cpu" else "mjwarp",
              "reused_uni_runtime": str(uni),
              "models": {}, "fixed_trials": 120, "fixed_validation_trials": 9,
              "total_soft_seconds": args.max_seconds, "per_model_soft_seconds": args.per_model_seconds,
              "external_timeout_required_seconds": 1800, "formal_E_star_acceptance": False,
              "runtime_sha256": {name: sha(runtime/name) for name in ("evaluate.py", "sports_runtime.py", "evaluate_batch.py")}}
    for name in ("parent", "f0", "f1"):
        model = models[name]
        original = Path(model["path"]).resolve()
        digest = sha(original)
        if digest != model["sha256"]:
            raise ValueError("I checkpoint hash changed: "+name)
        target = checkpoint_dir/(name+".pt")
        # Read-only-use symlink: preserves source, no multi-GB recopy or re-save.
        target.symlink_to(original)
        report["models"][name] = {"source_checkpoint": str(original), "checkpoint": str(target),
                                 "sha256_before": digest, "feedback": model["feedback"], "status": "not_run"}
    write(args.output/"j-evaluation.json", report)
    for name in ("parent", "f0", "f1"):
        row = report["models"][name]
        remaining = deadline-time.monotonic()-30
        if remaining <= 0:
            row["status"] = "not_run_engineering_deadline"
            continue
        budget = min(args.per_model_seconds, remaining)
        command = [sys.executable, "-u", str(runtime/evaluator.name), "--runtime-dir", str(runtime),
                   "--checkpoint", row["checkpoint"], "--manifest", str(old_manifest),
                   "--output", str(args.output/name), "--feedback", row["feedback"],
                   "--device", args.device, "--expect-count", "40", "--max-seconds", str(budget),
                   "--validation-steps", "100"]
        row.update(status="running", command=command)
        write(args.output/"j-evaluation.json", report)
        with (args.output/(name+".log")).open("w", encoding="utf-8") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=child_env)
            try:
                row["exit_code"] = child.wait(timeout=budget+10)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait(timeout=5)
                row.update(status="process_budget_interrupted", exit_code=child.returncode)
        summary_path = args.output/name/"summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            row.update(aggregate=summary.get("aggregate"),
                       short_prefix_validation=summary.get("batch_single_same_backend_validation"),
                       full_protocol_evaluation_complete=summary.get("full_protocol_evaluation_complete", False),
                       summary_sha256=sha(summary_path))
        if row["status"] == "running":
            row["status"] = "complete_native_batch_with_short_prefix_check" if row.get("full_protocol_evaluation_complete") else "incomplete_or_validation_failed"
        row["sha256_after"] = sha(row["source_checkpoint"])
        row["checkpoint_unchanged"] = row["sha256_before"] == row["sha256_after"]
        write(args.output/"j-evaluation.json", report)
        print("J_MODEL_RECEIPT "+json.dumps({"model": name, "status": row["status"],
                                           "aggregate": row.get("aggregate")}), flush=True)
    report.update(status="evaluation_only_finished", wall_seconds=time.monotonic()-started,
                  all120_native_outcomes_and_9_short_checks=all(row.get("full_protocol_evaluation_complete", False) for row in report["models"].values()))
    write(args.output/"j-evaluation.json", report)
    return 0 if report["all120_native_outcomes_and_9_short_checks"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
