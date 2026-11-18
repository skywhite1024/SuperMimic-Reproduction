"""Verify the PPO-Sim training clock, immutable snapshots and native evaluation."""

import argparse
import json
import math
from pathlib import Path
import re
import shutil

from push_recovery_contract import sha256, write_json
from verify_horizon_grid_receipts import decode_group
from verify_resumed_experiment_receipts import EVALUATION_FILES, plain_log, score_evaluation


def read(path):
    return json.loads(path.read_bytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("plan", "log", "reference-result", "output"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--original-training", type=Path, help="Original full training JSON for the 18-hour job")
    args = parser.parse_args()
    plan = read(args.plan)
    targets = plan["expected_timed_checkpoints_seconds"]
    smoke = targets == [20, 60]
    source = plain_log(args.log)
    assert "PPO_SIM_TIMED_BOOTSTRAP_SHA256 " + plan["bootstrap_sha256"] in source
    assert "PPO_SIM_TIMED_JOB_EXIT=0" in source and "PPO_SIM_TIMED_PIPELINE_COMPLETED " in source
    assert "NVIDIA H100 80GB HBM3" in source
    assert all(f"PPO_SIM_TIMED_SOURCE_SHA256 {file} {digest}" in source
               for file, digest in plan["source_files"].items())
    observed_evals = re.findall(r'"log": "[^"]+/eval-(\d+)/evaluation.log"', source)
    expected = {"result.json", "training/resolved-config.json", "training/checkpoint-manifest.json",
                *(f"eval-{index}/evaluation/{name}" for index in observed_evals for name in EVALUATION_FILES)}
    if smoke:
        expected.add("training/result.json")
    names, ledger = decode_group(source, plan["output"], args.output, expected)
    write_json(args.output / "receipt-ledger.json", {"receipts": ledger})
    assert set(names) == expected
    root = args.output
    parent = read(root / "result.json")
    cfg_file, series_file, training_file = [root / "training" / name for name in (
        "resolved-config.json", "checkpoint-manifest.json", "result.json")]
    assert parent["status"] == "complete" and parent["timing_smoke"] == smoke
    assert parent["checkpoint_training_seconds"] == targets and parent["timed_snapshots_verified"]
    if not smoke:
        if args.original_training is None:
            raise ValueError("The hash-bound full training original is required for the complete 18-hour verification")
        assert sha256(args.original_training) == parent["training_result_sha256"]
        if not training_file.is_file():
            shutil.copyfile(args.original_training, training_file)
    assert sha256(training_file) == parent["training_result_sha256"]
    training, series, cfg = map(read, (training_file, series_file, cfg_file))
    updates = training["updates"]
    assert training["source_config_sha256"] == sha256(cfg_file) and training["checkpoint_manifest_sha256"] == sha256(series_file)
    assert len(updates) == cfg["outer_updates"] == training["ppo_update_calls"] == parent["actual_update_calls"]
    assert training["status"] == "real_G1_simulator_PPO_finite_training_completed"
    assert training["checkpoint_sets"] == series
    assert cfg["checkpoint_seconds"] == targets and cfg["seed"] == 42 and cfg["rollout_steps"] == 24
    assert cfg["no_world_model_constructed"] and cfg["num_envs"] == 8192
    assert parent["imagined_transitions"] == training["imagined_transitions"] == training["world_model_updates"] == 0
    assert training["simulator_steps"] == parent["real_simulator_steps"] == len(updates) * 8192 * 24
    assert training["elapsed_training_seconds"] >= targets[-1]
    assert [second for snap in series for second in snap["crossed_training_clock_seconds"]] == targets
    for snap in series:
        count = snap["ppo_update"]
        last = updates[count - 2]["update_complete_seconds_since_training_entry"] if count > 1 else 0
        current = updates[count - 1]["update_complete_seconds_since_training_entry"]
        assert all(last < target <= current for target in snap["crossed_training_clock_seconds"])
        assert all(file["model_state_readback_verified"] for file in snap["files"])
    identities = [(row["name"], row["reference_frames"]) for row in read(args.reference_result)["records"]]
    assert len(identities) == 40 and sum(length for _, length in identities) == 441140
    scores = []
    for point in parent["evaluations"]:
        index = point["ppo_update"]
        folder = root / f"eval-{index:06d}/evaluation"
        score = score_evaluation(folder, identities, plan["canonical_motion_hashes"])
        selection = read(folder / "protocol.json")["checkpoint_selection"]
        assert selection == point["checkpoint_selection"]
        assert selection["parent_result_sha256"] == sha256(training_file)
        assert selection["resolved_config_sha256"] == sha256(cfg_file)
        assert selection["manifest_sha256"] == sha256(series_file)
        assert selection["no_world_model_loaded"] and selection["evaluated_ppo_update"] == index
        assert math.isclose(score["mean_normalized_full_motion_return"], point["mean_normalized_full_motion_return"], rel_tol=1e-13)
        assert score["successes"] == point["successes"]
        scores.append(dict(score, ppo_update=index, crossed_training_clock_seconds=point["crossed_training_clock_seconds"]))
    assert len(scores) == (1 if smoke else len(series))
    proof = {"status": "timed_ppo_sim_training_and_native_eval_verified", "timing_smoke": smoke,
             "original_JSONs": len(names), "log_sha256": sha256(args.log),
             "source_commit": plan["source_commit"], "checkpoints": targets, "updates": len(updates),
             "real_simulator_steps": training["simulator_steps"], "clock_seconds": training["elapsed_training_seconds"],
             "scores": scores, "paper_exact_sampler_confirmed": False}
    write_json(root / "validation.json", proof)
    print(json.dumps(proof, indent=2))


if __name__ == "__main__":
    main()
