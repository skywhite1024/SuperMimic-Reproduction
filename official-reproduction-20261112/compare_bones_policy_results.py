"""Compare policies on the exact ordered BONES membership used for evaluation."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def load_policy(path):
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    accounting = json.loads(path.with_name("accounting.json").read_bytes())
    maximum_step_reward = accounting["maximum_step_reward"]
    assert accounting["result_sha256"] == digest and maximum_step_reward == .1
    result = json.loads(data)
    rows = result["records"]
    assert result["status"] == "full_corpus_evaluation_completed"
    assert [row["index"] for row in rows] == list(range(len(rows)))
    identities = [(row["name"], row["reference_frames"]) for row in rows]
    assert len(set(identities)) == len(identities)
    lengths = np.array([frames for _, frames in identities], dtype=np.int64)
    assert len(lengths) and np.all(lengths > 0)
    finished = np.array([row["success"] for row in rows], dtype=bool)
    scores = np.array([row["reward_sum"] for row in rows], dtype=np.float64) / (maximum_step_reward * lengths)
    assert np.all(np.isfinite(scores))
    assert math.isclose(float(scores.mean()), accounting["mean_normalized_full_motion_return"], rel_tol=1e-13)
    assert np.count_nonzero(finished) == result["successes"]
    assert np.count_nonzero(~finished) == result["failures"]
    failure_terms = Counter("+".join(sorted(row["termination_terms"]))
                            for row in rows if not row["success"])
    metrics = {"successes": result["successes"], "failures": result["failures"],
               "success_fraction": float(finished.mean()),
               "mean_normalized_full_motion_return": float(scores.mean()),
               "frame_weighted_normalized_return": float(np.average(scores, weights=lengths)),
               "failure_terms": dict(sorted(failure_terms.items())),
               "original_sha256": digest}
    return identities, scores, finished, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    identities, baseline, baseline_finished, baseline_metrics = load_policy(args.baseline)
    compared, candidate, candidate_finished, candidate_metrics = load_policy(args.candidate)
    assert identities == compared
    assert len(identities) == 129543 and sum(frames for _, frames in identities) == 47374271
    complete_both = baseline_finished & candidate_finished
    added = ~baseline_finished & candidate_finished
    lost = baseline_finished & ~candidate_finished
    failed_both = ~baseline_finished & ~candidate_finished
    difference = candidate - baseline
    shared_return_difference = float(difference[complete_both].mean()) if complete_both.any() else None
    report = {"baseline_label": args.baseline_label, "candidate_label": args.candidate_label,
              "clips": len(identities), "reference_frames": 47374271,
              "exact_ordered_identities_match": True,
              "baseline": baseline_metrics, "candidate": candidate_metrics,
              "paired_completion": {"both": int(complete_both.sum()), "candidate_only": int(added.sum()),
                                    "baseline_only": int(lost.sum()), "neither": int(failed_both.sum())},
              "paired_return": {"mean_candidate_minus_baseline": float(difference.mean()),
                                "median_candidate_minus_baseline": float(np.median(difference)),
                                "mean_candidate_minus_baseline_on_shared_completed_clips": shared_return_difference}}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
