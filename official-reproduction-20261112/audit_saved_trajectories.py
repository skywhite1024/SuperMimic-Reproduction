"""Read every saved trajectory, including failures, and verify its recorded arrays.

This CPU audit checks stored file integrity and reward accounting. It does not
re-simulate a policy, evaluate motion quality or establish simulator determinism.
"""

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import time

import numpy as np


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def audit(folder, output, expected_clips, *, initial_state_check=None):
    folder, output = Path(folder), Path(output)
    result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    accounting = json.loads((folder / "accounting.json").read_text(encoding="utf-8"))
    if (expected_clips < 1 or accounting["status"] != "complete_corpus_accounting_verified"
            or result["clips"] != expected_clips or accounting["clips"] != expected_clips
            or accounting["result_sha256"] != digest(folder / "result.json")):
        raise ValueError("Complete, hash-bound corpus accounting is required")
    rows = sorted(result["records"], key=lambda row: row["index"])
    if len(rows) != expected_clips or [row["index"] for row in rows] != list(range(expected_clips)):
        raise ValueError("Missing or duplicated motion indices")
    filenames = [row["trajectory"] for row in rows]
    if len(set(filenames)) != expected_clips or any(Path(name).name != name for name in filenames):
        raise ValueError("Trajectory filenames must be unique and local")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    summary = {"status": "trajectory_integrity_incomplete", "expected_clips": expected_clips,
        "result_sha256": accounting["result_sha256"], "accounting_sha256": digest(folder / "accounting.json"),
        "verified_clips": 0, "verified_successes": 0, "verified_failures": 0,
        "verified_steps": 0, "verified_bytes": 0, "maximum_reward_sum_difference": 0.0,
        "reward_sum_absolute_tolerance": 1e-9, "array_widths": None,
        "all_trajectory_arrays_verified": False,
        "scope": "Read all NPZ arrays with ZIP CRC checks, float32 dtype, finite values, common state/action widths, step counts and float64 reward-sum agreement; no replay or physics quality claim"}
    partial = output / "files.partial.csv"
    try:
        with partial.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("index", "trajectory", "success", "steps", "bytes", "sha256"))
            for row in rows:
                raw = (folder / row["trajectory"]).read_bytes()
                with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
                    if set(archive.files) != {"full_state", "actions", "rewards"} or len(archive.files) != 3:
                        raise ValueError("Unexpected trajectory array members")
                    state, actions, rewards = (archive[name] for name in ("full_state", "actions", "rewards"))
                steps = row["steps"]
                if (not isinstance(steps, int) or steps < 1 or state.ndim != 2 or actions.ndim != 2
                        or state.shape[0] != steps or actions.shape[0] != steps or rewards.shape != (steps,)
                        or state.shape[1] < 1 or actions.shape[1] < 1):
                    raise ValueError(f"Trajectory shape/step mismatch at index {row['index']}")
                for values in (state, actions, rewards):
                    if values.dtype != np.float32 or not np.isfinite(values).all():
                        raise ValueError(f"Invalid dtype or nonfinite array at index {row['index']}")
                widths = {"full_state": state.shape[1], "actions": actions.shape[1]}
                if summary["array_widths"] is None:
                    summary["array_widths"] = widths
                elif widths != summary["array_widths"]:
                    raise ValueError("State/action widths vary across stored motions")
                if initial_state_check is not None:
                    initial_state_check(row, state[0])
                difference = abs(float(np.sum(rewards, dtype=np.float64)) - row["reward_sum"])
                if not math.isfinite(difference) or difference > summary["reward_sum_absolute_tolerance"]:
                    raise ValueError(f"Stored reward sum differs from result at index {row['index']}")
                summary["maximum_reward_sum_difference"] = max(summary["maximum_reward_sum_difference"], difference)
                summary["verified_clips"] += 1
                summary["verified_successes" if row["success"] else "verified_failures"] += 1
                summary["verified_steps"] += steps
                summary["verified_bytes"] += len(raw)
                writer.writerow((row["index"], row["trajectory"], int(row["success"]), steps,
                    len(raw), hashlib.sha256(raw).hexdigest()))
                if summary["verified_clips"] % 5000 == 0:
                    stream.flush()
                    print("TRAJECTORY_AUDIT_PROGRESS " + json.dumps({key: summary[key] for key in
                        ("verified_clips", "verified_bytes", "verified_failures")}), flush=True)
        if (summary["verified_successes"] != accounting["successes"]
                or summary["verified_failures"] != accounting["failures"]
                or summary["verified_steps"] != sum(row["steps"] for row in rows)):
            raise ValueError("Full trajectory totals differ from result/accounting")
        ledger = output / "files.csv"
        partial.rename(ledger)
        summary.update(status="complete_trajectory_integrity_verified", all_trajectory_arrays_verified=True,
            ledger={"file": ledger.name, "sha256": digest(ledger), "bytes": ledger.stat().st_size})
    except Exception as error:
        summary.update(status="trajectory_integrity_failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        summary["elapsed_seconds"] = time.monotonic() - started
        (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n",
            encoding="utf-8", newline="\n")
    print("TRAJECTORY_AUDIT_COMPLETED " + json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--expected-clips", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.evaluation, args.output, args.expected_clips)


if __name__ == "__main__":
    main()
