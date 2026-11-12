"""Regression against the actual native observation manager's cached reset read.

Deterministic tensor scene/control stubs isolate cache and backfill behavior;
the GPU standing step test separately verifies the real G1 initial pose.
"""

from types import SimpleNamespace

import torch
from mjlab.managers.observation_manager import ObservationManager, ObservationGroupCfg, ObservationTermCfg

from manual_reset_observations import refresh_manual_reset_observations


def read_state(env):
    return env.state


def test_pose_write_needs_native_cache_refresh_and_control_history_backfill():
    state = torch.zeros(2, 71)
    state[:, 2], state[:, 3] = .76, 1
    calls = []
    data = SimpleNamespace(root_link_pos_w=state[:, :3], root_link_quat_w=state[:, 3:7],
        joint_pos=state[:, 7:36], root_link_lin_vel_w=state[:, 36:39],
        root_link_ang_vel_w=state[:, 39:42], joint_vel=state[:, 42:])
    class Scene:
        env_origins = torch.zeros(2, 3)

        def __getitem__(self, name):
            assert name == "robot"
            return SimpleNamespace(data=data)

        def write_data_to_sim(self, *, env_ids):
            assert env_ids.tolist() == [0, 1]
            calls.append("actuator_history_backfill")

    base = SimpleNamespace(state=state, num_envs=2, device="cpu", step_dt=.02,
        scene=Scene(), sim=SimpleNamespace(forward=lambda: calls.append("forward"),
            sense=lambda: calls.append("sense")),
        command_manager=SimpleNamespace(get_term=lambda _: SimpleNamespace(
            update_relative_body_poses=lambda: calls.append("relative_poses"))))
    cfg = {"full_state": ObservationGroupCfg(terms={"state": ObservationTermCfg(func=read_state)},
        enable_corruption=False, concatenate_terms=True)}
    base.observation_manager = ObservationManager(cfg, base)
    before = base.observation_manager.compute(update_history=True)
    state[:, 0], state[:, 7] = 2, .1
    assert base.observation_manager.compute() is before
    assert before["full_state"][:, 0].tolist() == [0, 0]
    fresh, evidence = refresh_manual_reset_observations(base)
    assert torch.equal(fresh["full_state"], state)
    assert calls == ["actuator_history_backfill", "forward", "relative_poses", "sense"]
    assert evidence == {"current_physical_state_verified": True,
        "observation_cache_reset_and_recomputed": True, "actuator_histories_backfilled": True,
        "cached_vs_fresh_state_max_abs_difference": 2.0}
