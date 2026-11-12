"""Summarize complete native full40 receipts, including early-failure return.

Normalize by the full reference duration, never by the observed survival length.
Report both equal-clip and frame-weighted aggregates without asserting the paper's
unspecified aggregation convention. This does not re-evaluate physics or quality.
"""

import argparse
import hashlib
import json
import math
import os
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["MUJOCO_GL"] = "disable"
    import supermimic.tasks  # registers the pinned author task
    from mjlab.tasks.registry import load_env_cfg

    result = json.loads(args.result.read_text(encoding="utf-8"))
    rows = result["records"]
    if result["status"] != "full40_evaluation_completed" or result["clips"] != 40:
        raise ValueError("A complete full40 evaluation is required")
    if len(rows) != 40 or sorted(r["index"] for r in rows) != list(range(40)):
        raise ValueError("All 40 distinct clip indices are required")
    if result["successes"] != sum(r["success"] for r in rows):
        raise ValueError("Inconsistent success count")
    if result["failures"] != 40 - result["successes"]:
        raise ValueError("Inconsistent failure count")
    cfg = load_env_cfg("SuperMimic-GenCoord-G1", play=True)
    dt = cfg.sim.mujoco.timestep * cfg.decimation
    positive_weights = {name: term.weight for name, term in cfg.rewards.items() if term.weight > 0}
    # Each positive tracking exponential is <= 1; penalties are <= 0.
    maximum = dt * sum(positive_weights.values())
    per_clip = []
    for row in rows:
        n, steps, reward = row["reference_frames"], row["steps"], row["reward_sum"]
        if not 0 < steps <= n or not math.isfinite(reward):
            raise ValueError("Invalid duration or reward")
        if row["success"] and steps != n:
            raise ValueError("Successful clip did not reach the reference end")
        if not math.isclose(row["seconds"], steps * dt, abs_tol=1e-6):
            raise ValueError("Evaluation control step differs from loaded config")
        if reward > steps * maximum + 1e-4:
            raise ValueError("Recorded reward exceeds the assumed upper bound")
        per_clip.append({**row, "normalized_full_motion_return": reward / (n * maximum)})
    summary = {
        "input_sha256": hashlib.sha256(args.result.read_bytes()).hexdigest(),
        "task": "SuperMimic-GenCoord-G1", "control_dt": dt,
        "positive_reward_weights": positive_weights, "maximum_step_reward": maximum,
        "normalization": "undiscounted reward sum / (reference frames * maximum step reward)",
        "early_termination": "unused reference tail contributes zero reward",
        "clips": 40, "successes": result["successes"], "failures": result["failures"],
        "mean_normalized_full_motion_return": statistics.mean(
            r["normalized_full_motion_return"] for r in per_clip),
        "frame_weighted_normalized_full_motion_return": (
            sum(r["reward_sum"] for r in rows) / (sum(r["reference_frames"] for r in rows) * maximum)),
        "median_tracked_seconds": statistics.median(r["seconds"] for r in rows),
        "tracked_fraction_frames": sum(r["steps"] for r in rows) / sum(r["reference_frames"] for r in rows),
        "records": per_clip,
        "paper_aggregation_confirmed": False, "paper_results_reproduced": False,
        "strict_pose_or_contact_quality_verified": False,
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
