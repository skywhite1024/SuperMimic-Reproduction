"""Locate the first trajectory differences between two full40 evaluations."""

import argparse
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def first_exceedance(a, b, tolerance):
    differences = np.abs(a - b)
    axes = tuple(range(1, differences.ndim))
    exceeded = np.any(differences > tolerance, axis=axes) if axes else differences > tolerance
    positions = np.flatnonzero(exceeded)
    return int(positions[0]) if len(positions) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocols = [read(folder / "protocol.json") for folder in (args.reference, args.candidate)]
    if protocols[0]["motion_files"] != protocols[1]["motion_files"]:
        raise ValueError("Frozen motion content differs")
    hashes = [{Path(row["path"]).name: row["sha256"] for row in p["training"]["checkpoints"]}
              for p in protocols]
    if hashes[0] != hashes[1]:
        raise ValueError("Training checkpoint identity differs")
    results = [read(folder / "result.json") for folder in (args.reference, args.candidate)]
    records = [{row["index"]: row for row in r["records"]} for r in results]
    if any(set(rows) != set(range(40)) for rows in records):
        raise ValueError("Both complete 40-motion evaluations required")
    report = []
    for index in range(40):
        x, y = (rows[index] for rows in records)
        if x["name"] != y["name"] or x["reference_frames"] != y["reference_frames"]:
            raise ValueError("Motion identity differs")
        row = {"index": index, "name": x["name"],
               "reference_success": x["success"], "candidate_success": y["success"],
               "reference_steps": x["steps"], "candidate_steps": y["steps"], "fields": {}}
        with np.load(args.reference / x["trajectory"]) as a, np.load(args.candidate / y["trajectory"]) as b:
            for field in ("full_state", "actions", "rewards"):
                u, v = a[field], b[field]
                if not np.isfinite(u).all() or not np.isfinite(v).all():
                    raise ValueError("Nonfinite saved trajectory")
                if len(u) != x["steps"] or len(v) != y["steps"] or u.shape[1:] != v.shape[1:]:
                    raise ValueError("Trajectory shape does not match the result receipt")
                length = min(len(u), len(v))
                u, v = u[:length], v[:length]
                row["fields"][field] = {
                    "common_frames": length,
                    "first_frame_exact": bool(np.array_equal(u[0], v[0])),
                    "first_frame_max_abs_difference": float(np.max(np.abs(u[0] - v[0]))),
                    "max_abs_difference": float(np.max(np.abs(u - v))),
                    "first_exceedance": {str(e): first_exceedance(u, v, e) for e in (0.0, 1e-7, 1e-5, 1e-3)},
                }
        report.append(row)
        state, action = row["fields"]["full_state"], row["fields"]["actions"]
        print("DIAG_ROW " + json.dumps({"i": index,
            "s0": state["first_frame_max_abs_difference"],
            "a0": action["first_frame_max_abs_difference"],
            "s1e5": state["first_exceedance"]["1e-05"],
            "a1e5": action["first_exceedance"]["1e-05"]}), flush=True)
        print("TRAJECTORY_DIFFERENCE " + json.dumps(row), flush=True)
    result = {
        "status": "comparison_completed", "clips": 40,
        "same_frozen_motion_hashes": True, "same_checkpoint_hashes": True,
        "first_state_exact_clips": sum(row["fields"]["full_state"]["first_frame_exact"] for row in report),
        "first_action_exact_clips": sum(row["fields"]["actions"]["first_frame_exact"] for row in report),
        "same_success_clips": sum(row["reference_success"] == row["candidate_success"] for row in report),
        "reference_protocol_seed": protocols[0]["seed"], "candidate_protocol_seed": protocols[1]["seed"],
        "records": report,
        "causal_limit": "Locates divergence; does not by itself attribute it to evaluator code or numerical simulation",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("TRAJECTORY_DIAGNOSIS " + json.dumps({k: v for k, v in result.items() if k != "records"}), flush=True)


if __name__ == "__main__":
    main()
