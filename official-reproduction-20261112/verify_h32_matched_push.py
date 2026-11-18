"""Recover the matched-H32 push originals and recompute the paired outcomes."""

import argparse
import json
from pathlib import Path
import statistics

from push_recovery_contract import DT, PUSH_STEP, STEPS, pulse_grid, sha256, write_json
from verify_horizon_grid_receipts import decode_group
from verify_resumed_experiment_receipts import ROOT, plain_log


POSE = "lafan_walk1_first_frame"
POLICIES = ("critic_h32", "no_critic_h32", "critic_h32_at_64k", "no_critic_h32_at_64k")


def read(path):
    return json.loads(path.read_bytes())


def check_outcomes(result, policy, labels, vectors):
    records = result["records"]
    assert result["policy"] == policy and result["pose"] == POSE
    assert result["status"] == "pose_sweep_completed" and result["cases"] == len(records) == len(labels)
    assert result["trace"]["readback_recomputed"] and result["max_root_qvel_jump_error"] <= 2e-5
    for row, label, vector in zip(records, labels, vectors, strict=True):
        assert row["case"] == label and row["delta_velocity_m_s"] == vector
        assert 1 <= row["observed_steps"] <= STEPS
        assert row["impulse_applied"] == (label != "zero" and row["observed_steps"] > PUSH_STEP)
        if row["observed_steps"] < STEPS:
            assert row["fell"]
    control = records[0]
    assert result["zero_push_eligible"] == (
        not control["fell"] and control["prepush_stable_final_1s"] and control["final_1s_stable"])
    pushed = [row for row in records if row["impulse_applied"]]
    assert len(pushed) == result["impulses"]
    falls = sum(row["fell"] for row in pushed)
    assert falls == result["falls_after_push"] and len(pushed) - falls == result["pushed_survived"]
    absolute = [row["recovery_seconds"] for row in pushed if row["recovery_seconds"] is not None]
    assert len(absolute) == result["pushed_settled"]
    relative = result["relative_to_prepush_posture"]
    assert relative["baseline_samples"] == 25
    elapsed = relative["recovery_seconds_in_grid_order"]
    shared = [seconds for row, seconds in zip(records, elapsed, strict=True)
              if row["impulse_applied"] and seconds is not None]
    assert len(shared) == relative["pushed_settled"]
    assert all(0 < seconds <= (STEPS - PUSH_STEP) * DT for seconds in shared)
    return {"trial_count": len(records), "applied_impulses": len(pushed),
            "falls_after_impulse": falls, "prepush_failures": result["prepush_failures"],
            "zero_impulse_eligible_absolute": result["zero_push_eligible"],
            "absolute_recoveries": len(absolute),
            "zero_impulse_eligible_relative_to_prepush": relative["zero_push_eligible"],
            "prepush_relative_recoveries": len(shared),
            "prepush_relative_median_seconds_among_recoveries": statistics.median(shared) if shared else None,
            "trace_sha256": result["trace"]["sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("plan", "scientific-plan", "log", "output"):
        parser.add_argument("--" + option, type=Path, required=True)
    args = parser.parse_args()
    plan = read(args.plan)
    source = plain_log(args.log)
    assert "PUSH_LAFAN_BOOTSTRAP_SHA256 " + plan["bootstrap_sha256"] in source
    assert "PUSH_LAFAN_BOOTSTRAP_EXIT_CODE 0" in source and "NVIDIA H100 80GB HBM3" in source
    run = plan["output"].removeprefix(ROOT + "/")
    stem = Path(run).name
    science_file = f"launch/{plan['source_commit']}-{stem}/scientific-plan.json"
    reference_file = f"{run}/reference/reference.json"
    driver_file = f"{run}/driver-result.json"
    expected = {science_file, reference_file, driver_file,
                *(f"{run}/{policy}/{name}" for policy in POLICIES for name in ("protocol.json", f"{POSE}.json"))}
    files, ledger = decode_group(source, ROOT, args.output, expected)
    assert set(files) == expected and len(files) == plan["expected_original_JSONs"]
    write_json(args.output / "receipt-ledger.json", {"receipts": ledger})
    assert sha256(args.output / science_file) == sha256(args.scientific_plan)
    science = read(args.output / science_file)
    driver = read(args.output / driver_file)
    reference = read(args.output / reference_file)
    assert driver["status"] == "completed" and driver["child_return_codes"] == dict.fromkeys(POLICIES, 0)
    assert driver["receipt_hashes"] == {file: sha256(args.output / file) for file in expected - {driver_file}}
    assert reference["static_pack_sha256"] == science["expected_static_pack_hashes"]
    assert reference["source_pack_sha256"] == science["canonical_source_hashes"]

    labels, vectors = pulse_grid()
    results = {}
    for policy in POLICIES:
        folder = args.output / run / policy
        protocol = read(folder / "protocol.json")
        result = read(folder / f"{POSE}.json")
        expected_policy = science["policies"][policy]
        assert protocol["policy"] == expected_policy and expected_policy["horizon"] == 32
        assert protocol["reference_poses"] == [POSE] and protocol["seed"] == 42
        assert result["protocol_sha256"] == sha256(folder / "protocol.json")
        assert protocol["dt_s"] == DT and protocol["impulse_at_step"] == PUSH_STEP
        assert protocol["elapsed_steps"] == STEPS and not protocol["auto_reset"] and protocol["weights_frozen"]
        results[policy] = dict(check_outcomes(result, policy, labels, vectors.tolist()),
                               collection_steps=expected_policy["collection"],
                               full40_normalized_return=expected_policy["full40_normalized_return"])
    assert sum(row["trial_count"] for row in results.values()) == plan["expected_new_trials"]
    for comparison in science["comparisons"].values():
        left, right = [results[name] for name in comparison["policies"]]
        assert abs(left["full40_normalized_return"] - right["full40_normalized_return"]) == comparison["return_gap"]
        assert min(left["full40_normalized_return"], right["full40_normalized_return"]) >= .82
        assert comparison["return_gap"] <= .005
    assert results[POLICIES[2]]["collection_steps"] == results[POLICIES[3]]["collection_steps"] == 64001
    report = {"status": "all_same_H32_push_originals_and_trial_scores_verified",
              "original_JSONs": len(files), "log_sha256": sha256(args.log), "horizon": 32,
              "pose": POSE, "total_trials": plan["expected_new_trials"],
              "comparisons": science["comparisons"], "policies": results}
    write_json(args.output / "validation.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
