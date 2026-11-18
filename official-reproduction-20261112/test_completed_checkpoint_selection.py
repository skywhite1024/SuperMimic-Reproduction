"""Exercise retained native model selection against overwrite and corrupt evidence."""

import json

import pytest
import torch

from checkpoint_series import preserve_checkpoint
from select_completed_checkpoint import digest, select_completed_checkpoint


def completed_run(root):
    live = root / "live"
    live.mkdir()
    paths = [live / n for n in ("world_model.pt", "actor.pt", "critic.pt")]
    for path in paths:
        torch.save({"weight": torch.tensor([11.0])}, path)
    receipt = preserve_checkpoint(paths, root / "checkpoint-series", 11)
    for path in paths:
        torch.save({"weight": torch.tensor([21.0])}, path)
    result = {"status": "pilot_completed", "collection_steps": 21,
              "retained_checkpoint_series": [receipt],
              "checkpoints": [{"path": str(p), "bytes": p.stat().st_size,
                               "sha256": digest(p)} for p in paths]}
    (root / "result.json").write_text(json.dumps(result))
    return result


def test_selected_snapshot_stays_distinct_from_final_model(tmp_path):
    completed_run(tmp_path)
    training, paths, evidence = select_completed_checkpoint(tmp_path, 11)
    assert training["collection_steps"] == 21
    assert evidence["evaluated_collection_steps"] == 11
    assert torch.load(paths["actor.pt"], weights_only=True)["weight"].item() == 11
    _, final_paths, final = select_completed_checkpoint(tmp_path)
    assert final["evaluated_collection_steps"] == 21
    assert torch.load(final_paths["actor.pt"], weights_only=True)["weight"].item() == 21
    assert evidence["parent_result_sha256"] == final["parent_result_sha256"]


def test_reject_incomplete_parent_missing_step_and_corrupt_snapshot(tmp_path):
    result = completed_run(tmp_path)
    with pytest.raises(ValueError, match="Exactly one retained"):
        select_completed_checkpoint(tmp_path, 10)
    result["status"] = "running"
    (tmp_path / "result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="completed parent"):
        select_completed_checkpoint(tmp_path, 11)
    result["status"] = "pilot_completed"
    (tmp_path / "result.json").write_text(json.dumps(result))
    target = tmp_path / "checkpoint-series/step-000000011/actor.pt"
    original = target.read_bytes()
    target.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    with pytest.raises(ValueError, match="hash mismatch"):
        select_completed_checkpoint(tmp_path, 11)


def test_two_rank_selection_binds_final_weights_to_the_completed_driver(tmp_path):
    group_dir = tmp_path / "training"
    rank_dir = group_dir / "rank-0"
    rank_dir.mkdir(parents=True)
    config = rank_dir / "resolved-config.json"
    config.write_text('{"goal_assembly_backend": "eager"}')
    records = []
    for name in ("world_model.pt", "actor.pt", "critic.pt"):
        path = rank_dir / name
        torch.save({"weight": torch.tensor([6.0])}, path)
        records.append({"file": name, "path": str(path), "bytes": path.stat().st_size,
                        "sha256": digest(path), "native_model_readback_verified": True})
    participant = {"rank": 0, "status": "two_rank_native_participant_completed", "local_collection_steps": 40001,
                   "local_simulator_steps": 4096 * 40001, "final_model_state_hashes": {"actor": "shared"},
                   "checkpoints": records, "resolved_config_sha256": digest(config)}
    group = {"status": "two_rank_native_training_completed", "world_size": 2, "backend": "nccl",
             "global_simulator_steps": 8192 * 40001,
             "participants": [participant, dict(participant, rank=1)]}
    group_path = group_dir / "result.json"
    (rank_dir / "result.json").write_text(json.dumps(participant))
    group_path.write_text(json.dumps(group))
    driver_path = tmp_path / "result.json"
    driver = {"status": "two_GPU_native_gate_completed", "parent_result_sha256": digest(group_path)}
    driver_path.write_text(json.dumps(driver))
    with pytest.raises(ValueError, match="completed parent"):
        select_completed_checkpoint(rank_dir)
    _, selected, evidence = select_completed_checkpoint(rank_dir, two_rank=True)
    assert evidence["kind"] == "two_rank_final_models" and evidence["global_real_simulator_steps"] == 8192 * 40001
    assert torch.load(selected["actor.pt"], weights_only=True)["weight"].item() == 6
    group["participants"][1]["final_model_state_hashes"] = {"actor": "diverged"}
    group_path.write_text(json.dumps(group))
    with pytest.raises(ValueError, match="completed and bound"):
        select_completed_checkpoint(rank_dir, two_rank=True)
    driver["parent_result_sha256"] = digest(group_path)
    driver_path.write_text(json.dumps(driver))
    with pytest.raises(ValueError, match="weights.*disagree"):
        select_completed_checkpoint(rank_dir, two_rank=True)
