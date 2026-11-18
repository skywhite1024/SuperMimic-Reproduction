"""Verify the H32 critic-free curve and select comparable full-LAFAN40 checkpoints."""

import argparse
import json
import math
from pathlib import Path
import re

from push_recovery_contract import sha256, write_json
from verify_horizon_grid_receipts import decode_group
from verify_resumed_experiment_receipts import EVALUATION_FILES, TRAINING_FILES, native_point, plain_log


def read(path):
    return json.loads(path.read_bytes())


def check_control(folder, steps, expected_points, motions, identities):
    train = folder / "training"
    run = read(train / "result.json")
    config = read(train / "resolved-config.json")
    assert run["status"] == "pilot_completed" and run["collection_steps"] == steps
    assert not run["critic_enabled"] and config["runner"]["policy"]["horizon"] == 32
    assert config["seed"] == 42 and config["num_envs"] == 4096
    assert all({item["file"] for item in snapshot["files"]} == {"actor.pt", "world_model.pt"}
               for snapshot in run["retained_checkpoint_series"])
    summary = read(folder / "result.json")
    assert summary["status"] == "complete"
    points = []
    for step, reported in zip(expected_points, summary["points"], strict=True):
        evaluation = folder / f"eval-{step:09d}"
        score = native_point(evaluation, train, step, identities, motions)
        assert math.isclose(score["mean_normalized_full_motion_return"],
                            reported["mean_normalized_full_motion_return"], rel_tol=1e-13)
        assert score["successes"] == reported["successes"]
        selection = read(evaluation / "protocol.json")["checkpoint_selection"]
        points.append(dict(score, checkpoint_selection=selection))
    return points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("plan", "log", "critic", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    plan = read(args.plan)
    source = plain_log(args.log)
    assert "LAFAN_CURVE_BOOTSTRAP_SHA256 " + plan["command_sha256"] in source
    assert "LAFAN_CURVE_JOB_EXIT=0" in source and "LAFAN_CURVE_CUDA_INTEGRATION_PASSED" in source
    assert all(f"LAFAN_CURVE_SOURCE_SHA256 {key} {digest}" in source for key, digest in plan["source_files"].items())
    ledger = {}
    for phase in ("integration", "main"):
        spec = plan[phase]
        block, = re.findall(rf"(?ms)^LAFAN_CURVE_RECEIPTS_BEGIN {phase}\n(.*?)^LAFAN_CURVE_RECEIPTS_END {phase}$", source)
        allowed = {f"{phase}/result.json", *(f"{phase}/training/{name}" for name in TRAINING_FILES
                   if phase == "main" or name != "learner-counts.json"),
                   *(f"{phase}/eval-{step:09d}/{name}" for step in spec["evaluation_steps"] for name in EVALUATION_FILES)}
        found, ledger[phase] = decode_group(block, plan["output"], args.output, allowed)
        assert set(found) == allowed and len(found) == spec["expected_original_json_receipts"]
    write_json(args.output / "receipt-ledger.json", {"originals_by_phase": ledger})

    base = args.critic
    references = read(base / "eval-000064001/result.json")["records"]
    identities = [(row["name"], row["reference_frames"]) for row in references]
    assert len(identities) == 40 and sum(length for _, length in identities) == 441140
    motions = plan["canonical_motion_hashes"]
    integration = check_control(args.output / "integration", 21, [11], motions, identities)
    no_critic = check_control(args.output / "main", 64001, plan["main"]["evaluation_steps"], motions, identities)
    critic_train = base / "training"
    assert read(critic_train / "result.json")["critic_enabled"]
    assert read(critic_train / "resolved-config.json")["runner"]["policy"]["horizon"] == 32
    critic = []
    for step in plan["main"]["evaluation_steps"]:
        source_dir = base / f"eval-{step:09d}"
        score = native_point(source_dir, critic_train, step, identities, motions)
        protocol = read(source_dir / "protocol.json")
        compared = read(args.output / "main" / f"eval-{step:09d}/protocol.json")
        assert protocol["termination_terms"] == compared["termination_terms"]
        critic.append(dict(score, checkpoint_selection=protocol["checkpoint_selection"]))
    key = "mean_normalized_full_motion_return"
    eligible = [(abs(a[key] - b[key]), -min(a[key], b[key]), abs(a["collection_steps"] - b["collection_steps"]), a, b)
                for a in critic for b in no_critic if min(a[key], b[key]) >= .82]
    pair = None
    if eligible:
        distance, _, _, a, b = min(eligible, key=lambda record: record[:3])
        pair = {"mean_normalized_return_difference": distance, "critic": a, "no_critic": b,
                "near_match_within_0_005": distance <= .005}
    result = {"status": "full40_H32_critic_free_originals_verified", "original_JSONs": sum(map(len, ledger.values())),
              "log_sha256": sha256(args.log), "source_commit": plan["source_commit"],
              "comparison": "Matched benchmark membership and horizon; seed 42; checkpoint selection by return",
              "critic_free_CUDA_integration": integration, "critic_curve": critic, "no_critic_curve": no_critic,
              "above_0_82_pairs": len(eligible), "nearest_eligible_pair": pair}
    write_json(args.output / "validation.json", result)
    print(json.dumps({"status": result["status"], "original_JSONs": result["original_JSONs"],
                      "nearest_eligible_pair": pair}, indent=2))


if __name__ == "__main__":
    main()
