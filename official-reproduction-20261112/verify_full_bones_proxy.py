"""Verify original BONES proxy benchmark receipts and every per-motion score."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil

from prepare_bones_duration import read_identity
from prepare_sonic_onnx_batch import VARIANT_HASHES
from push_recovery_contract import sha256, write_json
from verify_horizon_grid_receipts import decode_group
from verify_resumed_experiment_receipts import EVALUATION_FILES, TRAINING_FILES, ROOT, native_point, plain_log, score_evaluation

PACK = "runs/bones-duration-1118c/motions/full-duration-pack.json"


def read(path):
    return json.loads(path.read_bytes())


def verify_variant_gate(source, plan, output):
    gate = plan["variant_gate_output"].removeprefix(ROOT + "/")
    gate_source, separator, full_source = source.partition("SONIC_BONES_VARIANT_GATE_COMPLETE")
    assert separator and "SONIC_BONES_VARIANT_GATE_COMPLETE" not in full_source
    required = {f"{gate}/driver.json", f"{gate}/models-batched/validation.json",
                *(f"{gate}/evaluation/{name}" for name in EVALUATION_FILES)}
    names, receipts = decode_group(gate_source, ROOT, output, required)
    assert set(names) == required
    gate_dir = output / gate
    driver_file = gate_dir / "driver.json"
    result_file = gate_dir / "evaluation/result.json"
    driver, result = read(driver_file), read(result_file)
    variant = plan["SONIC_variant"]
    derived = read(gate_dir / "models-batched/validation.json")
    assert driver["status"] == "completed" and driver["result_sha256"] == sha256(result_file)
    assert result["SONIC_variant"] == driver["SONIC_variant"] == derived["variant"] == variant
    assert derived["original_files"] == VARIANT_HASHES[variant] and derived["every_retained_parameter_byte_unchanged"]
    assert driver["original_comparison_and_cuda_kernels_verified"]
    assert result["ONNX_provider"] == "CUDAExecutionProvider"
    assert result["original_batch1_vs_batched_observed_state_comparisons"] > 0
    assert all(row["cuda_compute_ops"] for row in result["ONNX_runtime_compute_profiles"].values())
    accounting = read(gate_dir / "evaluation/accounting.json")
    states = read(gate_dir / "evaluation/audit/initial-states.json")
    assert accounting["clips"] == 64 and accounting["reference_frames"] == 20397
    assert accounting["result_sha256"] == sha256(result_file) and states["all_saved_initial_states_verified"]
    return full_source, receipts, sha256(driver_file), sha256(result_file)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("continuous", "sonic", "two-rank"), required=True)
    for arg in ("plan", "log", "raw-bones", "filter-report", "feasible-mask", "output"):
        parser.add_argument("--" + arg, type=Path, required=True)
    parser.add_argument("--per-motion", nargs="+", type=Path, required=True,
                        help="Hash-bound original files in checkpoint order")
    parser.add_argument("--two-rank-training-receipts", type=Path)
    args = parser.parse_args()
    plan = read(args.plan)
    source = plain_log(args.log)
    continuous = args.kind == "continuous"
    sonic = args.kind == "sonic"
    marker = {"continuous": "BONES_CONTINUOUS", "sonic": "SONIC_BONES_FULL",
              "two-rank": "BONES_NATIVE_2GPU_EVAL"}[args.kind]
    assert marker + "_BOOTSTRAP_SHA256 " + plan["bootstrap_sha256"] in source
    assert marker + "_BOOTSTRAP_EXIT_CODE 0" in source
    assert "NVIDIA H100 80GB HBM3" in source
    run = plan["clouds_output" if sonic else "output"].removeprefix(ROOT + "/")
    driver = f"{run}/result.json"
    if continuous:
        steps = plan["full_evaluation_collections"]
        evaluations = [f"{run}/eval-{step:09d}" for step in steps]
        training = {f"{run}/training/{name}" for name in TRAINING_FILES}
    else:
        steps, evaluations, training = [None], [f"{run}/evaluation"], set()
    gate_receipts = []
    if sonic and "variant_gate_output" in plan:
        source, gate_receipts, gate_driver_digest, gate_result_digest = verify_variant_gate(source, plan, args.output)
    elif sonic:
        gate_driver_digest, gate_result_digest = plan["cuda_gate_driver_sha256"], plan["cuda_gate_result_sha256"]
    assert len(args.per_motion) == len(evaluations)
    allowed = {PACK, driver, *training,
               *(f"{base}/{name}" for base in evaluations for name in EVALUATION_FILES[1:])}
    names, ledger = decode_group(source, ROOT, args.output, allowed)
    write_json(args.output / "receipt-ledger.json", {"receipts": gate_receipts + ledger})
    assert set(names) == allowed
    pack = read(args.output / PACK)
    parent = read(args.output / driver)
    assert parent["status"] == "complete" and pack["filter_report_sha256"] == sha256(args.filter_report)
    motion_names, _, lengths, indices = read_identity(args.raw_bones, read(args.filter_report), args.feasible_mask)
    identities = [(motion_names[i], int(lengths[i])) for i in indices]
    assert hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest() == pack["membership_sha256"]
    assert len(identities) == 129543 and sum(length for _, length in identities) == 47374271
    if continuous:
        assert parent["confirmed_learner_continuity"] and parent["single_continuous_training_process"]
        assert not parent["restarts_or_warm_start"] and plan["collection_target"] == 320001
        assert len(steps) == 2 and steps == [80001, 320001]
        assert parent["pack"] == pack
        assert [row["collection_steps"] for row in parent["evaluations"]] == steps
    elif sonic:
        assert pack["membership_sha256"] == plan["membership_sha256"]
        assert parent["gate_driver_sha256"] == gate_driver_digest
        assert parent["gate_result_sha256"] == gate_result_digest
        assert parent.get("SONIC_variant", "default") == plan.get("SONIC_variant", "default")
    else:
        originals = args.two_rank_training_receipts
        if originals is None:
            parser.error("two-rank evaluation requires its verified training receipts")
        rank0_path = originals / "training/rank-0/result.json"
        group_path = originals / "training/result.json"
        validation = read(originals / "validation.json")
        assert validation["status"] == "actual_two_H100_NCCL_native_BONES_training_originals_verified"
        assert parent["training_driver_sha256"] == sha256(originals / "result.json") == plan["training_driver_sha256"]
        assert parent["training_rank0_result_sha256"] == sha256(rank0_path)
        assert parent["training_group_sha256"] == sha256(group_path)
        assert parent["global_real_transitions"] == validation["global_real_simulator_steps"] == 40001 * 8192
        assert parent["training_collections_per_rank"] == 40001 and parent["training_ranks"] == 2
        assert pack["membership_sha256"] == plan["membership_sha256"]

    scored = []
    for index, (base, step, original) in enumerate(zip(evaluations, steps, args.per_motion, strict=True)):
        folder = args.output / base
        accounting = read(folder / "accounting.json")
        measured = sha256(original)
        assert measured == accounting["result_sha256"]
        parent_hash = (parent["evaluations"][index]["result_sha256"] if continuous
                       else parent["per_motion_result_sha256"])
        assert measured == parent_hash
        target = folder / "result.json"
        if target.is_file():
            assert sha256(target) == measured
        else:
            shutil.copyfile(original, target)
        if continuous:
            score = native_point(folder, args.output / run / "training", step, identities, pack["files"],
                                 num_envs=1024, batch_order="length")
            driver_row = parent["evaluations"][index]
            score["saved_snapshot_wall_seconds"] = driver_row["saved_snapshot_wall_seconds"]
        else:
            score = score_evaluation(folder, identities, pack["files"], num_envs=1024, batch_order="length")
            driver_row = parent
            if sonic:
                result = read(target)
                assert result["ONNX_provider"] == "CUDAExecutionProvider"
                assert result.get("SONIC_variant", "default") == parent.get("SONIC_variant", "default")
                assert result["ONNX_runtime_compute_profiles"] == parent["profiles"]
                assert all(profile["cuda_compute_ops"] for profile in parent["profiles"].values())
            else:
                protocol = read(folder / "protocol.json")
                selected = protocol["checkpoint_selection"]
                assert selected["kind"] == "two_rank_final_models"
                assert selected["parent_result_sha256"] == parent["training_rank0_result_sha256"]
                assert selected["verified_driver_sha256"] == parent["training_driver_sha256"]
                assert selected["group_result_sha256"] == parent["training_group_sha256"]
                assert selected["evaluated_collection_steps"] == 40001
                assert selected["global_real_simulator_steps"] == parent["global_real_transitions"]
                assert protocol["training_algorithm"] == "first-order-two-rank"
        assert math.isclose(score["mean_normalized_full_motion_return"],
                            driver_row["mean_normalized_return"], rel_tol=1e-13)
        assert score["successes"] == driver_row["successes"]
        scored.append(dict(score, per_motion_original_sha256=measured))
    validation = {"status": "all_proxy_per_motion_records_and_original_audits_verified", "kind": args.kind,
                  "original_bounded_JSONs": len(ledger), "original_large_per_motion_JSONs": len(scored),
                  "main_log_sha256": sha256(args.log), "source_commit": plan["source_commit"],
                  "membership_sha256": pack["membership_sha256"], "exact_author_membership": False,
                  "scores": scored}
    if gate_receipts:
        validation["original_variant_gate_JSONs"] = len(gate_receipts)
        validation["SONIC_variant"] = plan["SONIC_variant"]
    write_json(args.output / "validation.json", validation)
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
