"""Pair every motion from two completed, identically evaluated frozen policies.

Inputs are evaluation folders containing result.json, protocol.json and the
hash-bound accounting.json produced by audit_corpus_result.py. The output keeps
every motion in a CSV and reports descriptive aggregates, not significance or
paper equivalence. Training configurations may differ and are identified below.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_run(folder, count):
    result, protocol, accounting = (read(folder / name) for name in
        ("result.json", "protocol.json", "accounting.json"))
    if result["status"] not in ("full_corpus_evaluation_completed", "full40_evaluation_completed"):
        raise ValueError("Both evaluations must be complete")
    if accounting["status"] != "complete_corpus_accounting_verified":
        raise ValueError("Complete corpus accounting is required")
    if any(obj["clips"] != count for obj in (result, protocol, accounting)):
        raise ValueError("All inputs must describe the expected complete corpus")
    for name in ("result", "protocol"):
        if digest(folder / f"{name}.json") != accounting[f"{name}_sha256"]:
            raise ValueError("Accounting does not identify these exact input bytes")
    rows = result["records"]
    if len(rows) != count or sorted(row["index"] for row in rows) != list(range(count)):
        raise ValueError("Missing or duplicated motion indices")
    if sum(bool(row["success"]) for row in rows) != accounting["successes"]:
        raise ValueError("Success count differs from audited records")
    if sum(row["reference_frames"] for row in rows) != accounting["reference_frames"]:
        raise ValueError("Reference frame denominator differs from accounting")
    return result, protocol, accounting, {row["index"]: row for row in rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--expected-clips", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.expected_clips < 1:
        parser.error("a positive complete-corpus clip count is required")
    a, pa, aa, rows_a = load_run(args.reference, args.expected_clips)
    b, pb, ab, rows_b = load_run(args.candidate, args.expected_clips)
    # Compare the entire evaluation contract, including all packed data hashes.
    # Frozen training/model identity is the intended varying factor.
    if ({key: value for key, value in pa.items() if key != "training"}
            != {key: value for key, value in pb.items() if key != "training"}):
        raise ValueError("Evaluation protocols differ; do not silently mix them")
    if a.get("goal_assembly_backend") != b.get("goal_assembly_backend"):
        raise ValueError("Goal assembly execution paths differ")
    maximum = aa["maximum_step_reward"]
    if not math.isfinite(maximum) or maximum <= 0 or maximum != ab["maximum_step_reward"]:
        raise ValueError("A common positive native reward bound is required")
    count = args.expected_clips
    returns_a, returns_b, frames = np.empty(count), np.empty(count), np.empty(count)
    steps_a, steps_b = np.empty(count), np.empty(count)
    transitions = Counter({key: 0 for key in
        ("both_success", "recovered_success", "regressed_failure", "both_failure")})
    failures_a, failures_b = Counter(), Counter()
    csv_rows = []
    for index in range(count):
        x, y = rows_a[index], rows_b[index]
        if x["name"] != y["name"] or x["reference_frames"] != y["reference_frames"]:
            raise ValueError("Motion identity or duration differs")
        n = x["reference_frames"]
        if n <= 0 or not all(math.isfinite(row["reward_sum"]) for row in (x, y)):
            raise ValueError("Invalid native reward or reference duration")
        frames[index] = n
        returns_a[index], returns_b[index] = (row["reward_sum"] / (n * maximum) for row in (x, y))
        steps_a[index], steps_b[index] = x["steps"], y["steps"]
        transition = (("both_success" if y["success"] else "regressed_failure") if x["success"]
                      else ("recovered_success" if y["success"] else "both_failure"))
        transitions[transition] += 1
        if not x["success"]:
            failures_a["+".join(sorted(x["termination_terms"]))] += 1
        if not y["success"]:
            failures_b["+".join(sorted(y["termination_terms"]))] += 1
        csv_rows.append((index, x["name"], n, int(x["success"]), int(y["success"]), transition,
            x["steps"], y["steps"], returns_a[index], returns_b[index],
            returns_b[index] - returns_a[index],
            "+".join(x["termination_terms"]), "+".join(y["termination_terms"])))
    for values, audit in ((returns_a, aa), (returns_b, ab)):
        if not math.isclose(float(values.mean()), audit["mean_normalized_full_motion_return"], abs_tol=1e-12, rel_tol=0):
            raise ValueError("Recomputed full-corpus mean differs from accounting")
        weighted = float(np.dot(values, frames) / frames.sum())
        if not math.isclose(weighted, audit["frame_weighted_normalized_full_motion_return"], abs_tol=1e-12, rel_tol=0):
            raise ValueError("Recomputed frame-weighted return differs from accounting")
    delta = returns_b - returns_a
    if not np.isfinite(delta).all():
        raise ValueError("Nonfinite paired return difference")
    args.output.mkdir(parents=True, exist_ok=False)
    csv_path = args.output / "paired-motions.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("index", "name", "reference_frames", "reference_success", "candidate_success",
            "transition", "reference_steps", "candidate_steps", "reference_normalized_return",
            "candidate_normalized_return", "normalized_return_delta", "reference_terminations", "candidate_terminations"))
        writer.writerows(csv_rows)

    def describe(folder, protocol, accounting, failures):
        training = protocol["training"]
        return {"evaluation": str(folder), "result_sha256": accounting["result_sha256"],
            "protocol_sha256": accounting["protocol_sha256"],
            "accounting_sha256": digest(folder / "accounting.json"),
            "training_collection_steps": training["collection_steps"],
            "training_seed": training["seed"], "training_profile": training.get("training_profile", "public_defaults"),
            "checkpoints": training["checkpoints"],
            "successes": accounting["successes"], "failures": accounting["failures"],
            "mean_normalized_full_motion_return": accounting["mean_normalized_full_motion_return"],
            "frame_weighted_normalized_full_motion_return": accounting["frame_weighted_normalized_full_motion_return"],
            "tracked_fraction_frames": accounting["tracked_fraction_frames"],
            "failure_termination_counts": dict(failures),
            "all_trajectory_files_present_nonempty": accounting["all_trajectory_files_present_nonempty"],
            "trajectory_array_integrity_audited": accounting["trajectory_array_integrity_audited"]}

    summary = {"status": "complete_corpus_pairing_verified", "clips": count,
        "reference_frames": int(frames.sum()), "identical_evaluation_protocol": True,
        "identical_packed_motion_hashes": True, "all_indices_paired_once": True,
        "reference": describe(args.reference, pa, aa, failures_a),
        "candidate": describe(args.candidate, pb, ab, failures_b),
        "success_transitions": dict(transitions),
        "success_rate_delta_percentage_points": 100 * (ab["successes"] - aa["successes"]) / count,
        "equal_clip_mean_return_delta": float(delta.mean()),
        "frame_weighted_return_delta": float(np.dot(delta, frames) / frames.sum()),
        "tracked_fraction_delta": float((steps_b.sum() - steps_a.sum()) / frames.sum()),
        "per_clip_return_delta_quantiles": {str(q): float(np.quantile(delta, q)) for q in (0, 0.05, 0.5, 0.95, 1)},
        "per_clip_return_change_counts": {"positive": int((delta > 0).sum()),
            "negative": int((delta < 0).sum()), "exact_zero": int((delta == 0).sum())},
        "paired_csv": {"file": csv_path.name, "rows": count, "bytes": csv_path.stat().st_size,
            "sha256": digest(csv_path)},
        "limitations": ["Descriptive comparison of frozen runs; no statistical significance claim",
            "Matching evaluation contracts do not establish matching training configurations",
            "Simulator determinism, strict pose/contact quality, held-out generalization and paper equivalence remain unverified"]}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print("COMPLETE_CORPUS_COMPARISON " + json.dumps({key: summary[key] for key in
        ("status", "clips", "success_transitions", "equal_clip_mean_return_delta")}), flush=True)


if __name__ == "__main__":
    main()
