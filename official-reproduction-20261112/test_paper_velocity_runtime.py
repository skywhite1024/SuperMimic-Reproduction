"""Focused routing/replay/final-save tests before real-simulator GPU integration."""

from types import SimpleNamespace as NS

import pytest
import torch
from tensordict import TensorDict
from mjlab.managers.reward_manager import RewardManager, RewardTermCfg
from mjlab.managers.termination_manager import TerminationManager, TerminationTermCfg
from mjlab.utils.lab_api.math import matrix_from_quat
from supermimic.replay_buffer import ReplayBuffer, FieldSpec
from supermimic.tasks.task_spec import TaskSpec

from paper_velocity_runtime import (
    InterfaceReplayView, MixedVelocityWorldModelEnv, configure_mixed_environment,
    full_state_velocity_reward, root_gravity_z,
)
from run_velocity_training import final_model_snapshot
from checkpoint_series import preserve_checkpoint


def motion_reward(env):
    return torch.ones(env.num_envs)


def motion_failure(env):
    return torch.ones(env.num_envs, dtype=torch.bool)


def no_failure(env):
    return torch.zeros(env.num_envs, dtype=torch.bool)


def config():
    return NS(scale_rewards_by_dt=True,
        rewards={"motion": RewardTermCfg(func=motion_reward, weight=3.)},
        terminations={"anchor_pos": TerminationTermCfg(func=motion_failure, params={"threshold": .25}),
                      "anchor_ori": TerminationTermCfg(func=no_failure, params={"threshold": .8}),
                      "motion_end": TerminationTermCfg(func=no_failure, time_out=True)})


def fake_sim(cfg):
    zeros = torch.zeros(4, 3)
    quat = torch.tensor([[1., 0., 0., 0.]]).repeat(4, 1)
    data = NS(root_link_pos_w=torch.tensor([[0., 0., .8]]).repeat(4, 1), root_link_quat_w=quat,
        root_link_lin_vel_w=zeros.clone(), root_link_ang_vel_w=zeros.clone(),
        joint_pos=torch.zeros(4, 2), joint_vel=torch.zeros(4, 2),
        soft_joint_pos_limits=torch.tensor([[[-1., 1.], [-1., 1.]]]).repeat(4, 1, 1),
        body_link_quat_w=quat[:, None], body_link_ang_vel_w=zeros[:, None])
    class Scene(dict):
        env_origins = torch.zeros(4, 3)
    env = NS(num_envs=4, device="cpu", cfg=cfg, step_dt=.02, max_episode_length_s=10,
        scene=Scene(robot=NS(body_names=["torso_link"], data=data)),
        action_manager=NS(action=torch.zeros(4, 2), prev_action=torch.zeros(4, 2)),
        _paper_velocity_command=torch.zeros(4, 3))
    return env


@pytest.mark.parametrize("mode", ["rl", "distill"])
def test_real_manager_routes_rewards_and_hard_failures_by_interface_once_per_dt(mode):
    cfg = configure_mixed_environment(config(), mode)
    env = fake_sim(cfg)
    # Threshold-only kwargs are accepted by the actual native tracking functions;
    # remove them here because these simple deterministic fixtures do not need them.
    for term in cfg.terminations.values():
        term.params.pop("threshold", None)
    reward_manager, termination_manager = RewardManager(cfg.rewards, env), TerminationManager(cfg.terminations, env)
    reward = reward_manager.compute(.02)
    expected = [.06, .06, .08, .08] if mode == "rl" else [.06] * 4
    torch.testing.assert_close(reward, torch.tensor(expected))
    env.scene["robot"].data.root_link_pos_w[3, 2] = .34
    done = termination_manager.compute()
    assert done.tolist() == ([True, True, False, True] if mode == "rl" else [True] * 4)


def buffer_schema():
    return {"state": FieldSpec(17), "prev_action": FieldSpec(2), "actor_obs": FieldSpec(17),
        "sim_ctx": FieldSpec(1), "motion_time": FieldSpec(dtype=torch.long),
        "velocity_command": FieldSpec(3), "velocity_held": FieldSpec(3),
        "velocity_reference": FieldSpec(dtype=torch.bool), "velocity_age": FieldSpec(dtype=torch.long),
        "velocity_interface": FieldSpec(dtype=torch.bool),
        "terminated": FieldSpec(dtype=torch.bool), "truncated": FieldSpec(dtype=torch.bool)}


def batch(age=0, *, grouped=False):
    state = torch.zeros(4, 17)
    state[:, 2], state[:, 3] = .8, 1.
    if grouped:
        state[:, 0] = torch.tensor([0., 1., 102., 103.])
    return TensorDict({"state": state, "prev_action": torch.zeros(4, 2), "actor_obs": state.clone(),
        "sim_ctx": torch.zeros(4, 1), "motion_time": torch.ones(4, dtype=torch.long),
        "velocity_command": torch.zeros(4, 3), "velocity_held": torch.zeros(4, 3),
        "velocity_reference": torch.ones(4, dtype=torch.bool),
        "velocity_age": torch.full((4,), age, dtype=torch.long),
        "velocity_interface": torch.tensor([False, False, True, True]),
        "terminated": torch.zeros(4, dtype=torch.bool), "truncated": torch.zeros(4, dtype=torch.bool)}, batch_size=[4])


def test_replay_views_keep_interfaces_separate_and_follow_ring_cursor_and_causal_history():
    parent = ReplayBuffer(buffer_schema(), buffer_size=4, num_envs=4)
    motion, velocity = InterfaceReplayView(parent, 0, 2), InterfaceReplayView(parent, 2, 4)
    for i in range(7):
        value = batch(grouped=True)
        value["state"][:, 1] = i
        value["actor_obs"] = value["state"].clone()
        parent.insert(value)
    torch.manual_seed(42)
    for view, group in ((motion, False), (velocity, True)):
        window, padded = view.sample_causal(3, 64)
        assert bool((window["state"][..., 0] >= 100).all()) == group
        assert (window["state"][..., 0] >= 100).any().item() == group
        assert window["state"][..., 1].min() >= 3 and window["state"][..., 1].max() <= 6
        assert padded.shape == (64, 3) and view._step == parent._step and view.num_transitions == 4


class ToyRepresentation:
    dt = .02
    _soft_joint_pos_limits = torch.tensor([[-1., 1.], [-1., 1.]])
    def represent(self, full):
        return full
    def extract_aux_state(self, full):
        return full[:, :7]
    def reconstruct_full_state(self, aux, state):
        return state
    def joint_state(self, state):
        return torch.cat((state[:, 7:9], state[:, 15:]), -1)
    def compute_actor_sensor(self, state, *_, **__):
        return state
    def add_proprio_noise(self, sensor):
        return sensor  # Deterministic routing fixture, not a sensor-noise model.
    def compute_wm_goal(self, motion, times, aux):
        goal = torch.zeros(len(times), 5)
        return goal, goal
    def world_frame_bodies(self, aux, state):
        pos = aux[:, None, :3].repeat(1, 2, 1)
        matrices = matrix_from_quat(aux[:, 3:7])[:, None].repeat(1, 2, 1, 1)
        return pos, matrices, state[:, None, 9:12].repeat(1, 2, 1), state[:, None, 12:15].repeat(1, 2, 1)


class ToyDynamics(torch.nn.Module):
    def forward(self, state, action, aux, context):
        change = torch.zeros_like(state)
        change[:, 9], change[:, 14] = action[:, 0] * .1, action[:, 1] * .1
        return aux, state + change


def wm(mode, history_length=1):
    representation = ToyRepresentation()
    reference = batch()["state"]
    motion = NS(motion_ends=torch.tensor([4]), root_quat_w=reference[:, 3:7],
                root_lin_vel_w=reference[:, 9:12], root_ang_vel_w=reference[:, 12:15])
    def reference_bodies(times):
        pos, _, linear, angular = representation.world_frame_bodies(reference[times, :7], reference[times])
        # Native MotionLoader references carry quaternions; representation FK
        # predictions carry rotation matrices. Keep those interfaces distinct.
        quat = reference[times, None, 3:7].repeat(1, 2, 1)
        return pos, quat, linear, angular
    motion.body_state = reference_bodies
    command = NS(motion=motion, motion_anchor_body_index=0, cfg=NS(body_names=["pelvis", "torso_link"]))
    env = NS(cfg=config(), num_actions=2, step_dt=.02, command_manager=NS(get_term=lambda _: command))
    env.unwrapped = env
    replay = ReplayBuffer(buffer_schema(), buffer_size=16, num_envs=4)
    specs = [TaskSpec("motion", 5, lambda *a: None), TaskSpec("velocity", 3, lambda *a: None)]
    result = MixedVelocityWorldModelEnv(dynamics=ToyDynamics(), representation=representation,
        env=env, batch_size=4, replay_buffer=replay, task_specs=specs, history_length=history_length,
        termination_softness=.05, velocity_mode=mode)
    return result


@pytest.mark.parametrize("mode", ["rl", "distill"])
def test_world_model_routes_reward_fall_and_critic_identity_and_keeps_action_gradients(mode):
    model = wm(mode)
    start = batch()
    start["state"][2:, 2] = .4  # Deviates from reference but is above the hard fall threshold.
    obs = model.reset(start)
    assert obs["critic"][:, -1].tolist() == [0, 0, 1, 1]
    action = torch.full((4, 2), .2, requires_grad=True)
    obs, reward, termination, _ = model.step(action)
    if mode == "rl":
        full = model.representation.reconstruct_full_state(model.aux_state, model.rep_state)
        bodies = model.representation.world_frame_bodies(model.aux_state, model.rep_state)
        expected = full_state_velocity_reward(full, bodies[1][:, 1, :, 2], bodies[3][:, 1],
            start["velocity_command"], action, start["prev_action"], model.representation._soft_joint_pos_limits, dt=.02)
        torch.testing.assert_close(reward[2:], expected[2:])
        assert (termination[2:] < .5).all()
    else:
        assert (termination[2:] > .5).all()
    reward.sum().backward()
    assert action.grad is not None and torch.isfinite(action.grad).all() and action.grad.abs().sum() > 0


def test_world_model_own_interface_starts_and_command_age_continue_at_resampling_boundary():
    model = wm("rl")
    for _ in range(3):
        model.replay_buffer.insert(batch(grouped=True))
    sampled = model.sample_initial_states()
    assert (sampled["state"][:2, 0] < 100).all() and (sampled["state"][2:, 0] >= 100).all()
    start = batch(age=248)
    start["velocity_reference"][:] = False
    start["velocity_held"][:] = .4
    start["velocity_command"][:] = .4
    model.reset(start)
    next_obs, *_ = model.step(torch.zeros(4, 2))
    torch.testing.assert_close(next_obs["goal_velocity"], torch.full((4, 3), .4))
    assert (model.command_schedule.age == 250).all()
    model.step(torch.zeros(4, 2))
    assert (model.command_schedule.age == 1).all()


def test_ten_frame_causal_warm_start_keeps_history_in_its_own_simulator_interface():
    model = wm("rl", history_length=10)
    for _ in range(3):
        model.replay_buffer.insert(batch(grouped=True))
    sampled = model.sample_initial_states()
    assert sampled["sensor_window"].shape == (4, 10, 17)
    obs = model.reset(sampled)
    assert obs["history"].shape == (4, 10, 17)
    assert (obs["history"][:2, :, 0] < 100).all()
    assert (obs["history"][2:, :, 0] >= 100).all()


def test_final_tail_save_preserves_exact100000_weights_without_overwriting96001(tmp_path):
    actor, critic = torch.nn.Linear(2, 1), torch.nn.Linear(2, 1)
    world_state = {"weight": torch.tensor([1.])}
    runner = NS(_wm=NS(bundle=lambda: world_state), policy_learner=NS(actor=actor, critic=critic))
    paths = [tmp_path / name for name in ("world_model.pt", "actor.pt", "critic.pt")]
    series = tmp_path / "series"
    first = final_model_snapshot(runner, paths, series, 96001)
    old = {entry["file"]: (series / "step-000096001" / entry["file"]).read_bytes() for entry in first["files"]}
    world_state["weight"].fill_(2)
    with torch.no_grad():
        actor.weight.add_(1)
    final = final_model_snapshot(runner, paths, series, 100000)
    assert final["collection_steps"] == 100000 and final["source"] == "post_loop_final_save"
    for name, original in old.items():
        assert (series / "step-000096001" / name).read_bytes() == original
    restored = torch.load(series / "step-000100000" / "actor.pt", weights_only=True)
    torch.testing.assert_close(restored["weight"], actor.weight)
