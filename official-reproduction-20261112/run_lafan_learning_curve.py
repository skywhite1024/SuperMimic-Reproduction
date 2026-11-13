"""Finite LAFAN40 learning curve, after a real CUDA snapshot-selection integration."""

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

from select_completed_checkpoint import digest, select_completed_checkpoint


SCRIPTS = Path(__file__).resolve().parent
POINTS = [1, 8001, 16001, 24001, 32001, 40001, 48001, 56001, 64001]


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def execute(script, arguments, log, timeout_seconds):
    command = [sys.executable, str(SCRIPTS / script), *map(str, arguments)]
    print("CURVE_STAGE_START " + json.dumps({"script": script, "log": str(log),
          "timeout_seconds": timeout_seconds,
          "utc": datetime.now(timezone.utc).isoformat()}), flush=True)
    started = time.monotonic()
    # GNU timeout terminates the worker process group, including compile workers.
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(["timeout", "--signal=TERM", "--kill-after=30s",
                                    str(timeout_seconds) + "s", *command],
                                   stdout=stream, stderr=subprocess.STDOUT)
    print("CURVE_STAGE_EXIT " + json.dumps({"script": script,
          "returncode": completed.returncode, "elapsed_seconds": time.monotonic() - started}), flush=True)
    if completed.returncode:
        print(log.read_text(encoding="utf-8", errors="replace")[-12000:], flush=True)
        raise RuntimeError(f"{script} exited {completed.returncode}; inspect {log}")


def verify_training(run, steps, interval):
    training = json.loads((run / "result.json").read_bytes())
    timing = json.loads((run / "checkpoint-times.json").read_bytes())
    expected = list(range(1, steps + 1, interval))
    if training["status"] != "pilot_completed" or training["collection_steps"] != steps:
        raise ValueError("Unexpected completed training length")
    if [r["collection_steps"] for r in timing["checkpoints"]] != expected:
        raise ValueError("Incomplete checkpoint timing series")
    previous = {key: -1.0 for key in ("seconds_since_program_entry", "seconds_since_runner_run_entry")}
    for row in timing["checkpoints"]:
        # Also re-read the immutable manifest and every model byte hash.
        select_completed_checkpoint(run, row["collection_steps"])
        for event in ("callback_entry_after_native_save", "retained_snapshot_complete",
                      "offline_artifact_logging_returned"):
            for key in previous:
                value = row[event][key]
                if not math.isfinite(value) or value < previous[key] or value < 0:
                    raise ValueError("Nonmonotonic checkpoint clock")
                previous[key] = value
    _, last, _ = select_completed_checkpoint(run, steps)
    _, final, _ = select_completed_checkpoint(run)
    if {n: digest(p) for n, p in last.items()} != {n: digest(p) for n, p in final.items()}:
        raise ValueError("Final live weights differ from the final retained set")
    return timing


def run_case(folder, motions, steps, interval, points, train_timeout, eval_timeout,
             *, policy_horizon=None, without_critic=False):
    folder.mkdir()
    run = folder / "training"
    report = {"status": "training", "collection_steps": steps, "evaluation_steps": points,
              "points": [], "paper_results_reproduced": False,
              "seed": 42, "evaluation_protocol": "manual-reset-v2 full canonical LAFAN40",
              "threshold": 0.82, "threshold_convention": "First observed sampled checkpoint >=0.82; no interpolation or exact crossing-time claim",
              "initialization_and_snapshot_overhead_reported": True,
              "evaluation_time_excluded_from_training_time": True,
              "matched_ppo_sim_comparison_completed": False}
    write(folder / "result.json", report)
    receipts = []
    try:
        extra = (["--policy-horizon", policy_horizon] if policy_horizon is not None else [])
        if without_critic:
            extra += ["--without-critic"]
        execute("run_lafan_pilot.py", ["--motions", motions, "--output", run,
            "--steps", steps, "--checkpoint-interval", interval, "--keep-checkpoint-series",
            "--record-checkpoint-times", "--corpus", "lafan40", "--num-envs", 4096,
            "--seed", 42, "--training-profile", "paper-a5-single-gpu", "--goal-backend", "eager", *extra],
            folder / "training.log", train_timeout)
        training = json.loads((run / "result.json").read_bytes())
        if without_critic and (training["critic_enabled"] is not False or
                {Path(r["path"]).name for r in training["checkpoints"]} != {"world_model.pt", "actor.pt"}):
            raise ValueError("No-critic run must produce exactly two native model checkpoints")
        report.update(critic_enabled=not without_critic, imagination_horizon=policy_horizon or 32)
        timing = verify_training(run, steps, interval)
        report["training_snapshots_and_monotonic_timing_verified"] = True
        report["retained_snapshots"] = len(timing["checkpoints"])
        receipts += [run / n for n in ("result.json", "resolved-config.json", "checkpoint-times.json")]
        if steps == 21:
            hashes = [select_completed_checkpoint(run, s)[2]["files"][0]["sha256"] for s in (1, 11, 21)]
            if len(set(hashes)) != 3:
                raise ValueError("Short CUDA integration did not change the world-model weights")
            report["three_distinct_world_model_snapshots"] = True
        report["status"] = "evaluating"
        write(folder / "result.json", report)
        for step in points:
            evaluation = folder / f"eval-{step:09d}"
            row = {"collection_steps": step, "status": "incomplete"}
            try:
                execute("evaluate_motion_corpus_v2.py", ["--run", run,
                    "--checkpoint-collection-step", step, "--motions", motions,
                    "--output", evaluation, "--expected-clips", 40, "--num-envs", 40,
                    "--order", "index", "--seed", 42], folder / f"eval-{step:09d}.log", eval_timeout)
                execute("audit_corpus_result.py", ["--result", evaluation / "result.json",
                    "--protocol", evaluation / "protocol.json", "--motions", motions,
                    "--expected-clips", 40, "--require-trajectories",
                    "--output", evaluation / "accounting.json"], folder / f"accounting-{step:09d}.log", 180)
                execute("audit_saved_initial_states_v2.py", ["--evaluation", evaluation,
                    "--motions", motions, "--expected-clips", 40, "--output", evaluation / "audit"],
                    folder / f"audit-{step:09d}.log", 180)
                accounting = json.loads((evaluation / "accounting.json").read_bytes())
                protocol = json.loads((evaluation / "protocol.json").read_bytes())
                selection = protocol["checkpoint_selection"]
                if (selection["evaluated_collection_steps"] != step
                        or selection["parent_collection_steps"] != steps
                        or selection["parent_result_sha256"] != digest(run / "result.json")):
                    raise ValueError("Evaluation is not bound to the intended parent/snapshot")
                row.update(status="complete", mean_normalized_full_motion_return=
                    accounting["mean_normalized_full_motion_return"],
                    successes=accounting["successes"], failures=accounting["failures"],
                    reference_frames=accounting["reference_frames"],
                    checkpoint_time=next(r for r in timing["checkpoints"] if r["collection_steps"] == step),
                    result_sha256=digest(evaluation / "result.json"),
                    protocol_sha256=digest(evaluation / "protocol.json"),
                    accounting_sha256=digest(evaluation / "accounting.json"),
                    trajectory_audit_sha256=digest(evaluation / "audit/summary.json"),
                    initial_states_audit_sha256=digest(evaluation / "audit/initial-states.json"))
                receipts += [evaluation / n for n in ("result.json", "protocol.json", "accounting.json",
                                                       "audit/summary.json", "audit/initial-states.json")]
            except Exception as error:
                row.update(status="failed", error=f"{type(error).__name__}: {error}")
            report["points"].append(row)
            write(folder / "result.json", report)
            print("CURVE_POINT_RESULT " + json.dumps(row), flush=True)
        report["status"] = "complete" if all(r["status"] == "complete" for r in report["points"]) else "incomplete"
        report["first_observed_sample_at_or_above_threshold"] = next(
            (r for r in report["points"] if r["status"] == "complete" and
             r["mean_normalized_full_motion_return"] >= report["threshold"]), None)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
    write(folder / "result.json", report)
    receipts.append(folder / "result.json")
    print("LAFAN_CURVE_RECEIPTS_BEGIN " + folder.name, flush=True)
    subprocess.run([sys.executable, str(SCRIPTS / "emit_json_receipts.py"), *map(str, receipts)], check=True)
    print("LAFAN_CURVE_RECEIPTS_END " + folder.name, flush=True)
    if report["status"] != "complete":
        raise RuntimeError(f"Incomplete curve case: {folder}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-main-after-integration", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    motions = args.motions.resolve()
    run_case(output / "integration", motions, 21, 10, [11], 900, 600)
    print("LAFAN_CURVE_CUDA_INTEGRATION_PASSED", flush=True)
    if args.run_main_after_integration:
        run_case(output / "main", motions, 64001, 4000, POINTS, 14400, 900)
    print("LAFAN_CURVE_PIPELINE_COMPLETED", flush=True)


if __name__ == "__main__":
    main()
