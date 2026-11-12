"""Verify complete motion membership and summarize native evaluation receipts.

The full result and trajectory files remain the source of every failure. This
audit checks accounting and native termination semantics, not tracking quality
or deterministic execution of the simulator.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

import numpy as np


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--motions", type=Path, required=True)
    parser.add_argument("--expected-clips", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-trajectories", action="store_true")
    args = parser.parse_args()
    os.environ["MUJOCO_GL"] = "disable"
    import supermimic.tasks  # registers pinned author task configuration
    from mjlab.tasks.registry import load_env_cfg

    result = json.loads(args.result.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    names = json.loads((args.motions / "motion_names.json").read_text(encoding="utf-8"))
    ends = np.load(args.motions / "motion_ends.npy")
    lengths = np.diff(np.concatenate(([0], ends)))
    count = args.expected_clips
    if (count <= 0 or len(names) != count or len(set(names)) != count
            or len(ends) != count or not np.issubdtype(ends.dtype, np.integer)
            or not np.all(lengths > 0)):
        raise ValueError("Expected complete named corpus with positive integer boundaries")
    for name in ("motion_names.json", "motion_ends.npy"):
        if digest(args.motions / name) != protocol["motion_files"][name]:
            raise ValueError("Evaluation protocol does not identify this frozen corpus")
    if (result["status"] not in ("full_corpus_evaluation_completed", "full40_evaluation_completed")
            or result["clips"] != count or protocol["clips"] != count):
        raise ValueError("A completed full-corpus result is required")
    rows = result["records"]
    if len(rows) != count or sorted(row["index"] for row in rows) != list(range(count)):
        raise ValueError("Missing or duplicate motion indices")
    cfg = load_env_cfg(protocol["task"], play=True)
    dt = cfg.sim.mujoco.timestep * cfg.decimation
    weights = {name: term.weight for name, term in cfg.rewards.items() if term.weight > 0}
    maximum = dt * sum(weights.values())
    if maximum <= 0:
        raise ValueError("Positive reward upper bound is required")
    failures = Counter()
    normalized, durations = [], []
    trajectory_names = set()
    trajectory_bytes = 0
    total_steps, reward_total, successes = 0, 0.0, 0
    for row in rows:
        index = row["index"]
        n, steps, reward = row["reference_frames"], row["steps"], row["reward_sum"]
        terms = row["termination_terms"]
        if row["name"] != names[index] or n != int(lengths[index]):
            raise ValueError("Result motion identity differs from the frozen corpus")
        if (not isinstance(steps, int) or not 0 < steps <= n or not math.isfinite(reward)
                or not math.isclose(row["seconds"], steps * dt, abs_tol=1e-6)
                or reward > steps * maximum + 1e-4):
            raise ValueError("Invalid duration, control step or native return")
        if not terms or any(name not in protocol["termination_terms"] for name in terms):
            raise ValueError("Missing or unknown termination reason")
        native_success = "motion_end" in terms and not any(
            not protocol["termination_terms"][name]["time_out"] for name in terms)
        if bool(row["success"]) != native_success or (native_success and steps != n):
            raise ValueError("Success accounting differs from native termination semantics")
        if Path(row["trajectory"]).name != row["trajectory"] or row["trajectory"] in trajectory_names:
            raise ValueError("Trajectory names must be unique filenames")
        trajectory_names.add(row["trajectory"])
        if args.require_trajectories:
            size = (args.result.parent / row["trajectory"]).stat().st_size
            if size <= 0:
                raise ValueError("Empty trajectory file")
            trajectory_bytes += size
        if not native_success:
            failures.update(["+".join(sorted(terms))])
        normalized.append(reward / (n * maximum))
        durations.append(steps * dt)
        total_steps += steps
        reward_total += reward
        successes += int(native_success)
    if result["successes"] != successes or result["failures"] != count - successes:
        raise ValueError("Final counts differ from all individual records")
    summary = {
        "status": "complete_corpus_accounting_verified",
        "result_sha256": digest(args.result), "protocol_sha256": digest(args.protocol),
        "task": protocol["task"], "clips": count, "reference_frames": int(ends[-1]),
        "successes": successes, "failures": count - successes,
        "failure_termination_counts": dict(failures),
        "control_dt": dt, "positive_reward_weights": weights,
        "maximum_step_reward": maximum,
        "normalization": "native undiscounted reward / full reference frames / maximum step reward; missing tail contributes zero",
        "mean_normalized_full_motion_return": statistics.mean(normalized),
        "frame_weighted_normalized_full_motion_return": reward_total / (int(ends[-1]) * maximum),
        "median_tracked_seconds": statistics.median(durations),
        "tracked_fraction_frames": total_steps / int(ends[-1]),
        "all_indices_names_lengths_verified": True,
        "all_trajectory_files_present_nonempty": args.require_trajectories,
        "trajectory_file_bytes": trajectory_bytes if args.require_trajectories else None,
        "trajectory_array_integrity_audited": False,
        "paper_aggregation_confirmed": False, "paper_results_reproduced": False,
        "deterministic_execution_verified": False,
        "strict_pose_or_contact_quality_verified": False,
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("CORPUS_ACCOUNTING " + json.dumps({key: summary[key] for key in
        ("status", "clips", "successes", "failures", "mean_normalized_full_motion_return")}), flush=True)


if __name__ == "__main__":
    main()
