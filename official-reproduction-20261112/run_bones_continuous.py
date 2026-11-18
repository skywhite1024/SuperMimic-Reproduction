"""Continuous native single-H100 training on the verified feasible BONES proxy."""

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

from push_recovery_contract import sha256, write_json
from run_lafan_learning_curve import execute, verify_training

STEPS = 320001
INTERVAL = 4000
EVAL_CHECKPOINTS = (80001, 320001)
TRAIN_SECONDS = 50400
EVAL_SECONDS = 7200
SCRIPTS = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motions", type=Path, required=True)
    parser.add_argument("--filter-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    motions = args.motions.resolve()
    pack_path = motions.with_name(motions.name + "-duration-pack.json")
    pack = json.loads(pack_path.read_bytes())
    if (pack["status"] != "bones_proxy_duration_pack_verified" or pack["case"] != "full"
            or pack["filter_report_sha256"] != args.filter_sha256
            or (pack["clips"], pack["frames"]) != (129543, 47374271)):
        raise RuntimeError("The verified full feasible proxy pack is required")
    if {p.name: sha256(p) for p in sorted(motions.iterdir()) if p.is_file()} != pack["files"]:
        raise RuntimeError("The feasible proxy pack changed")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    train = output / "training"
    report = {
        "status": "training",
        "seed": 42, "collection_target": STEPS, "single_continuous_training_process": True,
        "restarts_or_warm_start": False, "real_envs": 4096, "pack": pack,
        "pack_receipt_sha256": sha256(pack_path), "paper_exact_membership": False,
        "evaluation_plan": EVAL_CHECKPOINTS, "evaluations": [],
    }
    write_json(output / "result.json", report)
    try:
        execute("run_lafan_pilot.py", ["--motions", motions, "--output", train,
            "--corpus", "bones-duration-proxy", "--steps", STEPS, "--num-envs", 4096,
            "--seed", 42, "--training-profile", "paper-a5-single-gpu", "--goal-backend", "eager",
            "--checkpoint-interval", INTERVAL, "--keep-checkpoint-series", "--record-checkpoint-times",
            "--record-learner-counts"], output / "train.log", TRAIN_SECONDS, stream_progress=True)
        times = verify_training(train, STEPS, INTERVAL)
        clocks = {r["collection_steps"]: r["retained_snapshot_complete"]["seconds_since_runner_run_entry"]
                  for r in times["checkpoints"]}
        report.update(status="evaluating", confirmed_learner_continuity=True)
        write_json(output / "result.json", report)
        for index in EVAL_CHECKPOINTS:
            ev = output / f"eval-{index:09d}"
            point = {"collection_steps": index, "real_steps": index * 4096,
                     "saved_snapshot_wall_seconds": clocks[index]}
            try:
                execute("evaluate_motion_corpus_v2.py", ["--run", train, "--checkpoint-collection-step",
                    index, "--motions", motions, "--output", ev, "--expected-clips", pack["clips"],
                    "--num-envs", 1024, "--order", "length", "--seed", 42],
                    output / f"{ev.name}.log", EVAL_SECONDS)
                execute("audit_corpus_result.py", ["--result", ev / "result.json", "--protocol",
                    ev / "protocol.json", "--motions", motions, "--expected-clips", pack["clips"],
                    "--require-trajectories", "--output", ev / "accounting.json"],
                    output / f"{ev.name}-accounting.log", 1200)
                execute("audit_saved_initial_states_v2.py", ["--evaluation", ev, "--motions", motions,
                    "--expected-clips", pack["clips"], "--output", ev / "audit"],
                    output / f"{ev.name}-audit.log", 2400)
                protocol = json.loads((ev / "protocol.json").read_bytes())
                source = protocol["checkpoint_selection"]
                accounting = json.loads((ev / "accounting.json").read_bytes())
                if (protocol["motion_files"] != pack["files"]
                        or source["evaluated_collection_steps"] != index
                        or source["parent_result_sha256"] != sha256(train / "result.json")
                        or accounting["reference_frames"] != pack["frames"]):
                    raise RuntimeError("Source checkpoint or feasibility membership differs")
                point.update(status="complete", clips=accounting["clips"], successes=accounting["successes"],
                    failures=accounting["failures"], mean_normalized_return=accounting["mean_normalized_full_motion_return"],
                    result_sha256=sha256(ev / "result.json"),
                    trajectory_audit_sha256=sha256(ev / "audit/summary.json"))
                if not math.isfinite(point["mean_normalized_return"]):
                    raise RuntimeError("Nonfinite evaluation score")
            except Exception as error:
                point.update(status="failed", error=f"{type(error).__name__}: {error}")
            report["evaluations"].append(point)
            write_json(output / "result.json", report)
            print("BONES_CONTINUOUS_EVALUATION " + json.dumps(point, allow_nan=False), flush=True)
        report["status"] = ("complete" if all(p["status"] == "complete" for p in report["evaluations"])
                            else "incomplete")
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "result.json", report)
        originals = [pack_path, output / "result.json"]
        originals += [train / name for name in ("result.json", "resolved-config.json",
                                                 "checkpoint-times.json", "learner-counts.json")]
        # Long per-motion records remain in AFS. Bounded accounting and audits
        # identify each original by hash for a later independent readback.
        originals += [output / f"eval-{index:09d}" / file for index in EVAL_CHECKPOINTS
                      for file in ("protocol.json", "accounting.json", "audit/summary.json",
                                   "audit/initial-states.json")]
        subprocess.run([sys.executable, str(SCRIPTS / "emit_json_receipts.py"), "--max-bytes",
                        "4000000", *map(str, (p for p in originals if p.is_file()))],
                       check=True, timeout=180)
    if report["status"] != "complete":
        raise RuntimeError("Continuous training or benchmark is incomplete")


if __name__ == "__main__":
    main()
