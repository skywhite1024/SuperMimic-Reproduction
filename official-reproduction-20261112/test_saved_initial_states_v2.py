"""Reject cached or unclipped first rows while retaining complete NPZ audits."""

import json

import numpy as np
import pytest

from audit_saved_initial_states_v2 import InitialStateCheck
from audit_saved_trajectories import audit, digest


def fixture(tmp_path, corruption=None):
    motions, evaluation = tmp_path / "motions", tmp_path / "evaluation"
    motions.mkdir()
    evaluation.mkdir()
    write = lambda path, value: path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    write(motions / "motion_names.json", ["success", "failure"])
    write(motions / "joint_names.json", [f"joint{i}" for i in range(29)])
    np.save(motions / "motion_ends.npy", np.array([2, 4], dtype=np.int64))
    fields = {"root_pos_w": 3, "root_quat_w": 4, "joint_pos": 29,
        "root_lin_vel_w": 3, "root_ang_vel_w": 3, "joint_vel": 29}
    arrays = {}
    for name, width in fields.items():
        values = np.zeros((4, width), dtype=np.float32)
        values[2:] = .4
        if name == "root_quat_w":
            values[:] = [1, 0, 0, 0]
        if name == "joint_pos":
            values[2:] = 1.2
        arrays[name] = values
        np.save(motions / (name + ".npy"), values)
    protocol = {"manual_reset_version": 2,
        "observation_cache_refreshed_after_pose_write": True,
        "actuator_histories_backfilled_after_pose_write": True,
        "native_joint_names": [f"joint{i}" for i in range(29)],
        "native_soft_joint_pos_limits": [[-1, 1]] * 29,
        "motion_files": {p.name: digest(p) for p in motions.iterdir()}}
    write(evaluation / "protocol.json", protocol)
    rows = []
    for index in range(2):
        initial = np.concatenate([value[index * 2] for value in arrays.values()])
        initial[7:36] = np.clip(initial[7:36], -1, 1)
        if index == 1 and corruption == "cached":
            initial = np.concatenate([value[0] for value in arrays.values()])
        if index == 1 and corruption == "unclipped":
            initial[7:36] = 1.2
        state = np.repeat(initial[None], 2, axis=0)
        rewards = np.array([.05, .06], dtype=np.float32)
        name = f"trajectory-{index:06d}.npz"
        np.savez_compressed(evaluation / name, full_state=state,
            actions=np.zeros((2, 29), dtype=np.float32), rewards=rewards)
        rows.append({"index": index, "name": ["success", "failure"][index],
            "reference_frames": 2, "steps": 2, "success": index == 0,
            "trajectory": name, "reward_sum": float(rewards.sum(dtype=np.float64))})
    result = {"clips": 2, "records": rows, "manual_reset_version": 2,
        "initial_reference_state_verified_for_all_clips": True}
    write(evaluation / "result.json", result)
    accounting = {"status": "complete_corpus_accounting_verified", "clips": 2,
        "successes": 1, "failures": 1, "result_sha256": digest(evaluation / "result.json"),
        "protocol_sha256": digest(evaluation / "protocol.json")}
    write(evaluation / "accounting.json", accounting)
    return motions, evaluation, protocol, result, accounting


def test_saved_first_states_include_failed_motion_and_native_clipping(tmp_path):
    motions, evaluation, protocol, result, accounting = fixture(tmp_path)
    check = InitialStateCheck(motions, protocol, result, accounting, 2)
    summary = audit(evaluation, tmp_path / "audit", 2, initial_state_check=check)
    assert summary["all_trajectory_arrays_verified"]
    assert summary["verified_successes"] == summary["verified_failures"] == 1
    assert check.count == 2 and check.maximum_error == 0


@pytest.mark.parametrize("corruption", ["cached", "unclipped"])
def test_invalid_saved_first_row_rejected_even_with_fresh_observation_flag(tmp_path, corruption):
    motions, evaluation, protocol, result, accounting = fixture(tmp_path, corruption)
    check = InitialStateCheck(motions, protocol, result, accounting, 2)
    with pytest.raises(ValueError, match="Saved first state differs.*index 1"):
        audit(evaluation, tmp_path / "audit", 2, initial_state_check=check)
    saved = json.loads((tmp_path / "audit/summary.json").read_text())
    assert saved["status"] == "trajectory_integrity_failed" and check.count == 1


def test_changed_pack_rejected_before_trajectory_read(tmp_path):
    motions, _, protocol, result, accounting = fixture(tmp_path)
    values = np.load(motions / "root_pos_w.npy")
    values[2, 0] += .1
    np.save(motions / "root_pos_w.npy", values)
    with pytest.raises(ValueError, match="Packed source hash differs"):
        InitialStateCheck(motions, protocol, result, accounting, 2)
