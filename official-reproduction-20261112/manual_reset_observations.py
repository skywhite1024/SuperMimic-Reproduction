"""Refresh controls and observations after an explicit full-batch pose write.

The native observation manager caches get_observations() results. Writing qpos
and calling forward()/sense() does not invalidate that cache. Mirror the native
reset backfill after the final pose write, then explicitly rebuild observations.
"""

import torch


def refresh_manual_reset_observations(base):
    cached = base.observation_manager.compute()["full_state"].clone()
    ids = torch.arange(base.num_envs, device=base.device)
    base.scene.write_data_to_sim(env_ids=ids)
    base.sim.forward()
    base.command_manager.get_term("motion").update_relative_body_poses()
    base.sim.sense()
    base.observation_manager.reset()
    obs = base.observation_manager.compute(update_history=True)
    robot = base.scene["robot"].data
    actual = torch.cat((robot.root_link_pos_w - base.scene.env_origins,
        robot.root_link_quat_w, robot.joint_pos, robot.root_link_lin_vel_w,
        robot.root_link_ang_vel_w, robot.joint_vel), -1)
    if not torch.allclose(obs["full_state"], actual, atol=1e-5, rtol=0):
        raise RuntimeError("Fresh observations disagree with the current simulated state")
    evidence = {"current_physical_state_verified": True,
        "observation_cache_reset_and_recomputed": True,
        "actuator_histories_backfilled": True,
        "cached_vs_fresh_state_max_abs_difference": float((cached - obs["full_state"]).abs().max())}
    return obs, evidence
