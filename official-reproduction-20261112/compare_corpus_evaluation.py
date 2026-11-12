"""Compare a batched evaluation against a preserved full40 receipt.

The tolerances below are diagnostic criteria, not paper quality thresholds.
Every motion is compared, including failed motions. A mismatch exits nonzero
after writing the complete difference report.
"""

import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-return-difference", type=float, default=1e-6)
    parser.add_argument("--max-step-difference", type=int, default=0)
    args = parser.parse_args()
    if args.max_return_difference < 0 or args.max_step_difference < 0:
        parser.error("nonnegative tolerances are required")
    reference, candidate = read(args.reference), read(args.candidate)
    if reference["status"] != "full40_evaluation_completed":
        raise ValueError("Reference must be the completed frozen full40 evaluation")
    if candidate["status"] != "full_corpus_evaluation_completed":
        raise ValueError("Candidate must be a completed corpus evaluation")
    a = {row["index"]: row for row in reference["records"]}
    b = {row["index"]: row for row in candidate["records"]}
    if (len(reference["records"]) != 40 or len(candidate["records"]) != 40
            or set(a) != set(range(40)) or set(b) != set(a)):
        raise ValueError("Both evaluations must contain each of the same 40 indices once")
    differences = []
    for index in range(40):
        x, y = a[index], b[index]
        if x["name"] != y["name"] or x["reference_frames"] != y["reference_frames"]:
            raise ValueError("Motion identity mismatch")
        normalized_return_difference = abs(x["reward_sum"] - y["reward_sum"]) / (x["reference_frames"] * 0.1)
        row = {
            "index": index, "name": x["name"],
            "same_success": x["success"] == y["success"],
            "same_termination_terms": set(x["termination_terms"]) == set(y["termination_terms"]),
            "step_difference": abs(x["steps"] - y["steps"]),
            "normalized_return_difference": normalized_return_difference,
        }
        row["passed"] = (row["same_success"] and row["same_termination_terms"]
                         and row["step_difference"] <= args.max_step_difference
                         and normalized_return_difference <= args.max_return_difference)
        differences.append(row)
    passed = all(row["passed"] for row in differences)
    result = {
        "status": "passed" if passed else "difference_requires_review",
        "scope": "40-clip outcome and return comparison; not trajectory equality or physics qualification",
        "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        "candidate_sha256": hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
        "candidate_num_envs": candidate["num_envs"],
        "candidate_batch_order": candidate["batch_order"],
        "tolerances": {"normalized_return": args.max_return_difference, "steps": args.max_step_difference},
        "same_success_count": sum(row["same_success"] for row in differences),
        "passed_clips": sum(row["passed"] for row in differences),
        "max_normalized_return_difference": max(row["normalized_return_difference"] for row in differences),
        "max_step_difference": max(row["step_difference"] for row in differences),
        "records": differences,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("CORPUS_COMPARISON " + json.dumps(result), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
