"""Targeted numerical and autograd checks for the pending velocity runner."""

from collections import Counter
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from paper_velocity_contract import (
    COMMAND_HIGH, COMMAND_LOW, ReferenceVelocityCommands, VelocityCommandSchedule,
    aggregate_step_trials, heading_velocity, step_command_grid, velocity_fall, velocity_reward,
)
from paper_velocity_actor import InterfaceCritic, PaperVelocityActor, interface_critic_input
from supermimic.tasks.task_spec import TaskSpec


def test_heading_frame_ignores_pitch_and_keeps_world_yaw_rate():
    cy, sy = math.cos(math.pi / 4), math.sin(math.pi / 4)
    cp, sp = math.cos(math.pi / 8), math.sin(math.pi / 8)
    quat = torch.tensor([[cy * cp, -sy * sp, cy * sp, sy * cp]], dtype=torch.float64)
    command = heading_velocity(quat, torch.tensor([[0., 1., 0.]], dtype=torch.float64),
                               torch.tensor([[0., 0., 1.]], dtype=torch.float64))
    torch.testing.assert_close(command, torch.tensor([[1., 0., 1.]], dtype=torch.float64))


def test_reference_projection_precedes_average_and_clip_edges_do_not_mix():
    motion = SimpleNamespace(
        motion_ends=torch.tensor([2, 4]),
        root_quat_w=torch.tensor([[1., 0., 0., 0.], [2**-0.5, 0., 0., 2**-0.5],
                                 [1., 0., 0., 0.], [1., 0., 0., 0.]], dtype=torch.float64),
        root_lin_vel_w=torch.tensor([[1., 0., 0.], [0., 1., 0.], [6., 0., 0.], [6., 0., 0.]], dtype=torch.float64),
        root_ang_vel_w=torch.zeros(4, 3, dtype=torch.float64),
    )
    table = ReferenceVelocityCommands(motion, window_frames=3)
    result = table.lookup(torch.tensor([0, 1, 2, 3]))
    torch.testing.assert_close(result[:, 0], torch.tensor([1., 1., 6., 6.], dtype=torch.float64))
    torch.testing.assert_close(result[:, 1:], torch.zeros(4, 2, dtype=torch.float64), atol=1e-15, rtol=0)
    with pytest.raises(ValueError, match="outside"):
        table.lookup(torch.tensor([-1]))


def test_held_command_lasts_250_controls_and_partial_reset_only_resamples_its_row():
    torch.manual_seed(123)
    schedule = VelocityCommandSchedule(8, reference_probability=0, standing_probability=0)
    reference = torch.full((8, 3), 99.)
    first, mask = schedule.command(reference)
    assert not mask.any()
    first = first.clone()
    assert torch.all(first >= torch.tensor(COMMAND_LOW)) and torch.all(first <= torch.tensor(COMMAND_HIGH))
    for _ in range(249):
        current, _ = schedule.command(reference)
        torch.testing.assert_close(current, first, rtol=0, atol=0)
    assert torch.all(schedule.age == 250)
    resampled, _ = schedule.command(reference)
    assert not torch.equal(resampled, first)
    before = resampled.clone()
    reset = torch.tensor([True] + [False] * 7)
    current, _ = schedule.command(reference, reset)
    torch.testing.assert_close(current[1:], before[1:], rtol=0, atol=0)
    assert not torch.equal(current[0], before[0])
    assert schedule.age[0] == 1 and torch.all(schedule.age[1:] == 2)


def test_reference_commands_are_unclipped_and_standing_only_changes_uniform_commands():
    reference = torch.tensor([[4., -3., 6.], [5., -4., 7.]])
    for mode in ("rl", "distill"):
        schedule = VelocityCommandSchedule(2, mode=mode, reference_probability=1, standing_probability=1)
        current, mask = schedule.command(reference)
        torch.testing.assert_close(current, reference)
        updated, _ = schedule.command(reference + 1)
        torch.testing.assert_close(updated, reference + 1)
        assert mask.all()
    schedule = VelocityCommandSchedule(2, reference_probability=0, standing_probability=1)
    current, mask = schedule.command(reference)
    assert not mask.any() and torch.count_nonzero(current) == 0


def test_table_a6_tracking_denominators_regularizers_and_fall_thresholds():
    dtype = torch.float64
    zero = torch.zeros(1, 3, dtype=dtype)
    actions = torch.zeros(1, 2, dtype=dtype)
    joints = torch.zeros(1, 2, dtype=dtype)
    limits = torch.tensor([[-1., 1.], [-1., 1.]], dtype=dtype)
    def score(velocity=zero, action=actions, q=joints, sine=torch.zeros(1, dtype=dtype),
              rates=torch.zeros(1, 2, dtype=dtype)):
        return velocity_reward(velocity, zero, action, actions, q, limits, sine, rates)
    assert score().item() == pytest.approx(0.08)
    assert score(torch.tensor([[.5, 0., 0.]], dtype=dtype)).item() == pytest.approx(.02 * (2 * math.exp(-1) + 2))
    assert score(torch.tensor([[0., 0., .5]], dtype=dtype)).item() == pytest.approx(.02 * (2 + 2 * math.exp(-.5)))
    assert score(action=torch.tensor([[.5, 0.]], dtype=dtype), q=torch.tensor([[-1.1, 1.2]], dtype=dtype),
                 sine=torch.tensor([.5], dtype=dtype), rates=torch.tensor([[1., 2.]], dtype=dtype)).item() == pytest.approx(.0415)
    fall = velocity_fall(torch.tensor([.35, .349, .9, .9], dtype=dtype),
                         torch.tensor([-1., -1., -.3, -.299], dtype=dtype))
    assert fall.tolist() == [False, True, False, True]


def test_step_grid_preserves_plane_identities_and_coincident_axis_commands():
    grid = step_command_grid()
    assert len(grid) == 672 and Counter(row["plane"] for row in grid) == {"vx-vy": 224, "vx-wz": 224, "vy-wz": 224}
    assert len({(r["plane"], r["row"], r["column"]) for r in grid}) == 672
    assert len({tuple(row["command"]) for row in grid}) == 630
    assert all(np.linalg.norm(row["command"]) > 0 for row in grid)


def test_step_error_uses_final_window_mean_vector_and_uncapped_survivor_median():
    command = np.array([1., 0., 0.])
    measured = np.broadcast_to(command, (4, 250, 3)).copy()
    measured[:, :150] = 1000  # Outside the final two seconds.
    measured[0, 150:, 0] += np.tile([1., -1.], 50)
    measured[1:, 150:, 0] += np.array([.2, .8, 2.])[:, None]
    result = aggregate_step_trials(command, measured, np.zeros(4, dtype=bool))
    assert result["trial_errors"] == pytest.approx([0, .2, .8, 2])
    assert result["median_relative_error"] == pytest.approx(.5)  # Even-sample median averages the middle two.
    result = aggregate_step_trials(command, measured, np.array([True, True, True, False]))
    assert result["median_relative_error"] == pytest.approx(2) and result["display_relative_error"] == .9
    all_fall = aggregate_step_trials(command, np.full_like(measured, np.nan), np.ones(4, dtype=bool))
    assert all_fall["all_trials_fell"] and all_fall["median_relative_error"] is None and all_fall["display_relative_error"] == .9
    measured[0, -1, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        aggregate_step_trials(command, measured, np.zeros(4, dtype=bool))


def make_actor(mix, *, training=False):
    specs = [TaskSpec("motion", 5, lambda *args: None), TaskSpec("velocity", 3, lambda *args: None)]
    torch.manual_seed(42)
    actor = PaperVelocityActor(3, 2, specs, mix, history_length=2, latent_dim=4,
                              motion_hidden_dims=(8, 8), velocity_hidden_dims=(8, 8))
    actor.train(training)
    history = torch.randn(4, 2, 3, requires_grad=True)
    obs = TensorDict({"history": history, "goal_motion": torch.randn(4, 5),
                      "goal_velocity": torch.randn(4, 3),
                      "velocity_reference_mask": torch.tensor([True, False, False, True])}, batch_size=[4])
    return actor, obs, history


def grad_total(module):
    return sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)


def test_downstream_return_freezes_all_decoder_weights_but_keeps_input_and_encoder_gradients():
    actor, obs, history = make_actor({"velocity": 1.})
    assert actor.encoders["velocity"][0].in_features == 3 + 2 * 3
    actor(obs).square().sum().backward()
    assert grad_total(actor.decoder) == 0 and grad_total(actor.encoders["motion"]) == 0
    assert grad_total(actor.encoders["velocity"]) > 0 and history.grad.abs().sum() > 0
    actor, obs, _ = make_actor({"motion": 1.})
    actor(obs).square().sum().backward()
    assert grad_total(actor.decoder) > 0 and grad_total(actor.encoders["motion"]) > 0
    assert grad_total(actor.encoders["velocity"]) == 0


def test_latent_alignment_is_reference_only_and_encoder_only():
    actor, obs, history = make_actor({"velocity": 1.}, training=True)
    actor(obs)
    loss = actor.auxiliary_loss()
    assert loss[0] > 0 and loss[3] > 0 and loss[1] == 0 and loss[2] == 0
    loss.sum().backward()
    assert grad_total(actor.encoders["velocity"]) > 0
    assert grad_total(actor.decoder) == 0 and grad_total(actor.encoders["motion"]) == 0
    assert history.grad is None  # The separate BC loss cannot train its rollout inputs.


def test_critics_follow_explicit_identity_through_time_flattening_and_shuffling():
    torch.manual_seed(123)
    critic = InterfaceCritic(4, 5, 2, hidden_dims=(8, 8))
    state, goal, command, context = torch.randn(8, 4), torch.randn(8, 5), torch.randn(8, 3), torch.randn(8, 2)
    mask = torch.tensor([False, True] * 4)
    inputs = interface_critic_input(state, goal, command, context, mask)
    values = critic(inputs)
    permutation = torch.tensor([4, 1, 7, 3, 6, 0, 2, 5])
    torch.testing.assert_close(critic(inputs[permutation]), values[permutation])
    torch.testing.assert_close(critic(inputs.reshape(2, 4, -1)).reshape(8, 1), values)
    velocity_only = interface_critic_input(state, goal, command, context, torch.ones(8, dtype=torch.bool))
    changed = velocity_only.clone(); changed[:, 7:9] += 123
    torch.testing.assert_close(critic(changed), critic(velocity_only))
    critic(velocity_only).sum().backward()
    assert grad_total(critic.velocity) > 0 and grad_total(critic.motion) == 0
