"""Recover original results and verify native motion scores for resumed jobs."""

import argparse
import gzip
import json
import math
from pathlib import Path
import re

import numpy as np

from prepare_bones_duration import CASES, read_identity, choose_subsets
from push_recovery_contract import sha256, write_json
from verify_horizon_grid_receipts import decode_group

ROOT = "/mnt/afs/260010076/SuperMimic/official-20261112"
EVALUATION_FILES = ("result.json", "protocol.json", "accounting.json",
                    "audit/summary.json", "audit/initial-states.json")
TRAINING_FILES = ("result.json", "resolved-config.json", "checkpoint-times.json", "learner-counts.json")


def read(path):
    return json.loads(path.read_bytes())


def score_evaluation(folder, identities, physical_hashes):
    result, protocol, accounting, arrays, states = [
        read(folder / name) for name in EVALUATION_FILES
    ]
    n = len(identities)
    rows = result["records"]
    assert result["status"] == "full_corpus_evaluation_completed"
    assert result["clips"] == protocol["clips"] == accounting["clips"] == len(rows) == n
    assert result["manual_reset_version"] == protocol["manual_reset_version"] == 2
    assert protocol["num_envs"] == n and protocol["batch_order"] == "index"
    assert not protocol["domain_randomization"] and not protocol["observation_noise"]
    assert protocol["motion_files"] == physical_hashes
    assert accounting["maximum_step_reward"] == .1
    assert [r["index"] for r in rows] == list(range(n))
    count, duration, denom = 0, 0, 0
    values = []
    for row, identity in zip(rows, identities, strict=True):
        assert (row["name"], row["reference_frames"]) == identity
        success = "motion_end" in row["termination_terms"] and not any(
            not protocol["termination_terms"][term]["time_out"] for term in row["termination_terms"])
        steps, frames = row["steps"], row["reference_frames"]
        reward = row["reward_sum"]
        assert success == row["success"] and (steps == frames if success else 0 < steps <= frames)
        assert math.isfinite(reward) and reward <= steps * .1 + 1e-4
        values.append(reward / (.1 * frames))
        count += int(success)
        duration += steps
        denom += frames
    mean = math.fsum(values) / n
    assert math.isclose(mean, accounting["mean_normalized_full_motion_return"], rel_tol=1e-13)
    assert count == result["successes"] == accounting["successes"] == arrays["verified_successes"]
    assert result["failures"] == n - count
    assert denom == accounting["reference_frames"]
    assert accounting["result_sha256"] == arrays["result_sha256"] == sha256(folder / "result.json")
    assert accounting["protocol_sha256"] == states["protocol_sha256"] == sha256(folder / "protocol.json")
    assert arrays["accounting_sha256"] == sha256(folder / "accounting.json")
    assert states["trajectory_integrity_summary_sha256"] == sha256(folder / "audit/summary.json")
    assert arrays["all_trajectory_arrays_verified"] and arrays["verified_steps"] == duration
    assert states["all_saved_initial_states_verified"] and states["saved_initial_states_verified"] == n
    return {"clips": n, "successes": count, "mean_normalized_full_motion_return": mean,
            "reference_frames": denom}


def native_point(folder, train, step, identities, physical_hashes):
    selected = read(folder / "protocol.json")["checkpoint_selection"]
    assert selected["evaluated_collection_steps"] == step
    assert selected["parent_result_sha256"] == sha256(train / "result.json")
    assert selected["resolved_config_sha256"] == sha256(train / "resolved-config.json")
    trained = read(train / "result.json")
    assert trained["status"] == "pilot_completed"
    assert selected["parent_collection_steps"] == trained["collection_steps"]
    snapshot = next(s for s in trained["retained_checkpoint_series"] if s["collection_steps"] == step)
    assert {Path(r["path"]).name: r["sha256"] for r in selected["files"]} == {
        r["file"]: r["sha256"] for r in snapshot["files"]
    }
    return {"collection_steps": step, **score_evaluation(folder, identities, physical_hashes),
            "real_transitions": step * 4096}


def verify_forecast(folder):
    forecast = read(folder / "state-forecast.json")
    result = read(folder / "result.json")
    assert forecast["status"] == "fresh_native_rollout_state_forecasts_verified"
    assert forecast["horizon_transitions"] == 8
    assert len(forecast["window_order"]) == len(result["records"])
    windows = 0
    for record, window in zip(result["records"], forecast["window_order"], strict=True):
        assert window["index"] == record["index"] and window["steps"] == record["steps"]
        assert window["offset"] == windows and window["windows"] == max(0, record["steps"] - 8)
        windows += window["windows"]
    assert windows == forecast["windows"] > 0
    dims = np.asarray(forecast["state_group_dimensions"], dtype=np.float64)
    raw = np.asarray(forecast["group_mse_by_horizon"])
    nom = np.asarray(forecast["nominal_errors"])
    assert dims.tolist() == [3, 9, 29, 29] and raw.shape == (8, 4) and (nom > 0).all()
    assert np.isfinite(raw).all() and (raw >= 0).all()
    independently = raw @ (dims / nom**2 / dims.sum())
    assert np.allclose(independently, forecast["normalized_state_mse_by_horizon"], atol=0, rtol=1e-12)
    assert math.isclose(independently[-1], forecast["eighth_transition_normalized_state_mse"], rel_tol=1e-12)
    assert math.isclose(independently.mean(), forecast["mean_normalized_state_mse_over_horizons"], rel_tol=1e-12)
    return {"forecast_windows": windows, "eighth_transition_normalized_state_mse": float(independently[-1]),
            "mean_normalized_state_mse_over_horizons": float(independently.mean())}


def verify_sonic(args, plan, source):
    assert "SONIC_BONES_CUDA_BOOTSTRAP_SHA256 " + plan["bootstrap_sha256"] in source
    assert "NVIDIA H100 80GB HBM3" in source
    allowed = {"driver.json", *(f"evaluation/{name}" for name in EVALUATION_FILES)}
    args.output.mkdir(parents=True, exist_ok=True)
    names, ledger = (decode_group(source, plan["cloud_output"], args.output, allowed)
                     if "\nRECEIPT_META " in source else ([], []))
    write_json(args.output / "receipt-ledger.json", {"receipts": ledger})
    validation = {"status": "incomplete_originals_preserved", "original_JSONs": len(names),
                  "log_sha256": sha256(args.log), "source_commit": plan["source_commit"]}
    if set(names) == allowed:
        driver = read(args.output / "driver.json")
        assert driver["status"] == "completed" and driver["original_comparison_and_cuda_kernels_verified"]
        assert driver["wheel_sha256"] == plan["wheel_sha256"] and driver["ONNX_runtime_version"] == plan["ORT_version"]
        assert "SONIC_BONES_CUDA_BOOTSTRAP_EXIT_CODE 0" in source
        study = read(args.duration_study)
        assert study["status"] == "complete" and study["filter_sha256"] == sha256(args.filter_report)
        hashes = study["packs"]["common64"]["files"]
        assert hashes == driver["motion_files"]
        names_raw, _, sizes, keep = read_identity(args.raw_bones, read(args.filter_report), args.feasible_mask)
        identities = [(names_raw[i], int(sizes[i])) for i in choose_subsets(sizes, keep)["common64"]]
        ev = args.output / "evaluation"
        result = read(ev / "result.json")
        protocol = read(ev / "protocol.json")
        assert driver["result_sha256"] == sha256(ev / "result.json")
        assert protocol["SONIC_ONNX_provider"] == result["ONNX_provider"] == "CUDAExecutionProvider"
        assert protocol["SONIC_control_contract_sha256"] == plan["source_file_hashes"]["sonic-g1-control-contract.json"]
        assert protocol["all_compiled_CPU_and_live_CUDA_control_fields_verified"]
        assert result["original_batch1_vs_batched_observed_state_comparisons"] >= 2
        profiles = result["ONNX_runtime_compute_profiles"]
        assert set(profiles) == {"encoder", "decoder"}
        for profile in profiles.values():
            assert len(profile["profile_sha256"]) == 64 and sum(profile["cuda_compute_ops"].values()) > 0
        assert len(identities) == 64 and sum(length for _, length in identities) == 20397
        validation.update(status="sonic_cuda_and_all_native_motion_scores_verified",
                          **score_evaluation(ev, identities, hashes), CUDA_compute_profiles=profiles,
                          original_CPU_output_comparisons=result["original_batch1_vs_batched_observed_state_comparisons"],
                          maximum_original_action_difference=result["maximum_original_action_difference"],
                          maximum_original_token_difference=result["maximum_original_token_difference"],
                          paper_exact_membership=False)
    write_json(args.output / "validation.json", validation)
    print(json.dumps(validation, indent=2))
    if validation["status"] != "sonic_cuda_and_all_native_motion_scores_verified":
        raise RuntimeError("SONIC has no complete original evaluation")


def verify(args):
    plan = read(args.plan)
    blob = args.log.read_bytes()
    raw = gzip.decompress(blob).decode("utf-8-sig") if args.log.suffix == ".gz" else blob.decode("utf-8-sig")
    source = "\n".join(re.sub(r"^\[[0-9TZ: .+-]+\] ", "", line) for line in raw.splitlines())
    if args.kind == "sonic":
        verify_sonic(args, plan, source)
        return
    duration = args.kind == "duration"
    marker = "BONES_DURATION" if duration else "LOCOMOTION18_CONTROL"
    assert marker + "_BOOTSTRAP_SHA256 " + plan["bootstrap_sha256"] in source
    assert "NVIDIA H100 80GB HBM3" in source
    points = plan["evaluation_collections"]
    run = (plan["cloud_output"] if duration else plan["output"]).removeprefix(ROOT + "/")
    if duration:
        launch = "launch/" + plan["source_commit"] + "-" + Path(run).name
        drivers = [launch + "/filter-result.json", launch + "/raw-manifest.json", run + "/study.json"]
        cases = {case: f"{run}/cases/{case}" for case in CASES}
    else:
        drivers = [run + "/motion-subset.json"]
        cases = {"same18": run + "/main"}
    allowed = set(drivers)
    for path in cases.values():
        allowed.update(path + "/training/" + name for name in TRAINING_FILES)
        allowed.add(path + ("/case-result.json" if duration else "/result.json"))
        allowed.update(f"{path}/eval-{step:09d}/{name}" for step in points for name in EVALUATION_FILES)
        if duration:
            allowed.add(f"{path}/eval-{points[-1]:09d}/state-forecast.json")
    assert len(allowed) == (87 if duration else 51)
    names, ledger = decode_group(source, ROOT, args.output, allowed)
    write_json(args.output / "receipt-ledger.json", {"receipts": ledger})
    if set(names) != allowed:
        write_json(args.output / "validation.json", {"status": "incomplete_originals_preserved",
                   "original_JSONs": len(names), "missing": sorted(allowed - set(names))})
        raise RuntimeError("An incomplete run or log export remains")

    if duration:
        assert "BONES_DURATION_BOOTSTRAP_EXIT_CODE 0" in source
        dispatch = read(args.output / drivers[-1])
        assert dispatch["status"] == "complete" and dispatch["case_return_codes"] == dict.fromkeys(CASES, 0)
        frozen_filter = read(args.output / drivers[0])
        names, _, sizes, keep = read_identity(args.raw_bones, frozen_filter, args.feasible_mask)
        partitions = choose_subsets(sizes, keep)
        identities = [(names[i], int(sizes[i])) for i in partitions["common64"]]
        hashes = dispatch["packs"]["common64"]["files"]
        assert sum(length for _, length in identities) == 20397
    else:
        assert "LOCOMOTION18_CONTROL_JOB_EXIT=0" in source
        subset = read(args.output / drivers[0])
        identities = [(r["name"], r["reference_frames"]) for r in subset["selected"]]
        assert len(identities) == 18 and sum(x[1] for x in identities) == 220352
        hashes = subset["packed_motion_files"]
    summaries = {}
    for case, prefix in cases.items():
        root = args.output / prefix
        training = root / "training"
        curve = read(root / ("case-result.json" if duration else "result.json"))
        assert curve["status"] == "complete"
        model = read(training / "result.json")
        assert model["collection_steps"] == points[-1]
        assert read(training / "resolved-config.json")["seed"] == 42
        assert read(training / "learner-counts.json")["status"] == "completed_native_calls_counted"
        verified = [native_point(root / f"eval-{step:09d}", training, step, identities, hashes)
                    for step in points]
        assert len(curve["points"]) == len(verified)
        for recorded, score in zip(curve["points"], verified, strict=True):
            expected = recorded["mean_normalized_return" if duration else "mean_normalized_full_motion_return"]
            assert math.isclose(expected, score["mean_normalized_full_motion_return"], rel_tol=1e-13)
        summaries[case] = {"points": verified}
        if duration:
            summaries[case]["forecast"] = verify_forecast(root / f"eval-{points[-1]:09d}")
            summaries[case]["data_hours"] = curve["pack"]["hours_at_50hz"]
    result = {"status": "all_originals_and_native_scores_verified", "kind": args.kind,
              "original_JSONs": len(ledger), "log_sha256": sha256(args.log),
              "source_commit": plan["source_commit"], "cases": summaries,
              "paper_exact_membership": False}
    write_json(args.output / "validation.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("same18", "duration", "sonic"), required=True)
    for name in ("plan", "log", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--raw-bones", type=Path)
    parser.add_argument("--feasible-mask", type=Path)
    parser.add_argument("--filter-report", type=Path)
    parser.add_argument("--duration-study", type=Path)
    args = parser.parse_args()
    if args.kind in {"duration", "sonic"} and not (args.raw_bones and args.feasible_mask):
        parser.error("BONES audit needs the local raw pack and feasibility mask")
    if args.kind == "sonic" and not (args.filter_report and args.duration_study):
        parser.error("SONIC audit needs the frozen filter report and the duration-study receipt")
    verify(args)
