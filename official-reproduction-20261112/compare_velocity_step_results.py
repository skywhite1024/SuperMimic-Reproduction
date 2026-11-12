"""Compare every fixed step-test identity from two complete receipt directories.

Uses only original result/protocol JSON; the evaluator's saved-array checks are
reported separately from this independent score/accounting recomputation. A
finite error ranks ahead of all-fall; two all-fall outcomes remain undefined.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def same_number(a, b):
    return a is None and b is None or (a is not None and b is not None
        and math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12))


def inspect(folder, mode, purpose):
    result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    protocol = json.loads((folder / "protocol.json").read_text(encoding="utf-8"))
    if (result["status"] != "velocity_step_test_completed" or result["mode"] != mode
            or result["purpose"] != purpose or protocol["mode"] != mode or protocol["purpose"] != purpose
            or result["protocol_sha256"] != digest(folder / "protocol.json")):
        raise ValueError("Result/protocol identity differs: " + mode)
    if (result["commands"] != 672 or result["total_trials"] != 2688
            or result["trials_per_command"] != 4 or protocol["grid_identities"] != 672
            or protocol["unique_physical_commands"] != 630 or protocol["trials"] != 4
            or protocol["fps"] != 50 or protocol["seconds"] != 5 or protocol["steady_state_seconds"] != 2):
        raise ValueError("The complete672x4 five-second protocol is required")
    if purpose == "experiment" and protocol["training"]["collection_steps"] != 100000:
        raise ValueError("Experimental comparison requires both complete100k actors")
    if not result["all_declared_initial_poses_verified"]:
        raise ValueError("Standing initialization was not verified")
    records = result["records"]
    if len(records) != 672 or [r["index"] for r in records] != list(range(672)):
        raise ValueError("Complete unique ordered command identities are required")
    fields = ("plane", "row", "column", "command")
    if [{k: r[k] for k in fields} for r in records] != protocol["commands"]:
        raise ValueError("Result commands differ from the declared grid")
    if len({tuple(r["command"]) for r in records}) != 630:
        raise ValueError("Changed unique physical command count")
    defined, display, nonfall, allfall = [], [], 0, 0
    for r in records:
        errors = r["trial_errors"]
        valid = [value for value in errors if value is not None]
        if len(errors) != 4 or any(not math.isfinite(v) or v < 0 for v in valid):
            raise ValueError("Invalid per-trial errors")
        if len(valid) != r["non_falling_trials"] or r["all_trials_fell"] != (not valid):
            raise ValueError("Per-command physical failure accounting differs")
        median = statistics.median(valid) if valid else None
        capped = min(median, .9) if valid else .9
        if not same_number(median, r["median_relative_error"]) or not same_number(capped, r["display_relative_error"]):
            raise ValueError("Per-command median or display cap differs")
        if median is not None:
            defined.append(median)
        display.append(capped)
        nonfall += len(valid)
        allfall += not valid
    raw_median = statistics.median(defined) if defined else None
    display_median = statistics.median(display)
    if (nonfall + result["fallen_trials"] != 2688 or allfall != result["all_fall_commands"]
            or not same_number(raw_median, result["median_defined_command_error"])
            or not same_number(display_median, result["median_display_command_error"])):
        raise ValueError("Complete score/failure totals do not recompute")
    trajectory = result["trajectory"]
    for key in ("saved_arrays_readback_verified", "saved_falls_recomputed",
                "saved_heading_velocity_recomputed_float64", "command_scores_recomputed_from_saved_traces"):
        if not trajectory[key]:
            raise ValueError("Saved-array evaluator validation is incomplete")
    return result, protocol, {"mode": mode, "result_sha256": digest(folder / "result.json"),
        "protocol_sha256": digest(folder / "protocol.json"), "fallen_trials": result["fallen_trials"],
        "all_fall_commands": allfall, "median_defined_command_error": raw_median,
        "median_display_command_error": display_median, "trajectory": trajectory}


def compare(rl, distill, purpose):
    left, lp, ls = inspect(rl, "rl", purpose)
    right, rp, rs = inspect(distill, "distill", purpose)
    for key in ("commands", "fps", "seconds", "steady_state_seconds", "trial_seeds", "initial_pose",
                "standing_root_state", "standing_joint_positions", "joint_names", "trial_perturbations",
                "domain_randomization", "observation_noise", "friction", "fall", "motion_manager", "packed_files"):
        if lp[key] != rp[key]:
            raise ValueError("Unequal evaluation protocols: " + key)
    if lp["training"]["collection_steps"] != rp["training"]["collection_steps"]:
        raise ValueError("Unequal completed training lengths")
    counts = {"rl_lower": 0, "distill_lower": 0, "finite_tie": 0, "both_all_fall": 0}
    rows = []
    for a, b in zip(left["records"], right["records"], strict=True):
        av, bv = a["median_relative_error"], b["median_relative_error"]
        winner = ("both_all_fall" if av is None and bv is None else "distill_lower" if av is None
                  else "rl_lower" if bv is None else "rl_lower" if av < bv
                  else "distill_lower" if bv < av else "finite_tie")
        counts[winner] += 1
        rows.append({k: a[k] for k in ("index", "plane", "row", "column", "command")} | {
            "rl_raw_error": av, "distill_raw_error": bv, "comparison": winner,
            "rl_minus_distill_raw": av - bv if av is not None and bv is not None else None,
            "rl_non_falling_trials": a["non_falling_trials"], "distill_non_falling_trials": b["non_falling_trials"]})
    if sum(counts.values()) != 672:
        raise RuntimeError("Incomplete paired comparison")
    return {"status": "complete_velocity_step_comparison", "purpose": purpose,
        "commands": 672, "unique_physical_commands": 630, "trials_per_model": 2688,
        "training_steps_per_model": lp["training"]["collection_steps"], "counts": counts,
        "ordering": "Uncapped raw command medians; finite ranks ahead of all-fall; both all-fall undefined",
        "independent_recomputation": "Python statistics.median from every original trial-error record;1e-12 numeric tolerance",
        "trajectory_file_bytes_rechecked_here": False,
        "trajectory_array_validation_source": "Evaluator reread files and independently recomputed falls, heading projection and command scores",
        "models": {"rl": ls, "distill": rs}, "records": rows,
        "repeatability_verified": False, "paper_results_reproduced": False,
        "limitations": ["One training seed; nominally identical standing trial initial conditions",
                        "Author library membership and exact initial standing pose are unconfirmed",
                        "Integration actors are short pipeline checks and cannot supply paper scores"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rl", type=Path, required=True)
    parser.add_argument("--distill", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--purpose", choices=("experiment", "integration"), default="experiment")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("A fresh comparison output is required")
    result = compare(args.rl, args.distill, args.purpose)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("VELOCITY_STEP_COMPARISON " + json.dumps({k: v for k, v in result.items() if k != "records"}))


if __name__ == "__main__":
    main()
