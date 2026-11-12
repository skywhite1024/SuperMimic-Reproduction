"""Mixed motion/velocity simulator and differentiable rollout plumbing.

The torso sideways-lean sine is the heading-frame Y component of its unit
Z axis (angle to the heading XZ plane). Smooth fall probabilities reuse the
native softness with normalized height/tilt margins. Both are explicit local
choices; the paper provides a hard fall rule but not this smoothing formula.
"""

import inspect

import torch
from tensordict import TensorDict
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.utils.lab_api.math import matrix_from_quat

from supermimic.envs.multi_task_world_model_env import MultiTaskWorldModelEnv
from supermimic.replay_buffer import ReplayBuffer
from supermimic.representations import tracking_reward
from supermimic.tasks.task_spec import build_actor_obs_td

from paper_velocity_actor import interface_critic_input
from paper_velocity_contract import (
    ReferenceVelocityCommands, VelocityCommandSchedule, heading_velocity,
    velocity_fall, velocity_reward,
)


def velocity_mask(env):
    if env.num_envs % 2:
        raise ValueError("An even environment count is required for uniform interfaces")
    return torch.arange(env.num_envs, device=env.device) >= env.num_envs // 2


def root_gravity_z(full_state):
    return 2 * full_state[..., 4:6].square().sum(-1) - 1


def full_state_velocity_reward(full_state, torso_up_world, torso_angular_world,
                               command, action, last_action, joint_limits, *, dt):
    """Identical reward kernel for simulator states and predicted FK states."""
    joints = (full_state.shape[-1] - 13) // 2
    if joints < 1 or full_state.shape[-1] != 13 + 2 * joints:
        raise ValueError("Generalized-coordinate full state is required")
    quat = full_state[..., 3:7]
    v_start = 7 + joints
    command_velocity = heading_velocity(quat, full_state[..., v_start:v_start + 3],
                                         full_state[..., v_start + 3:v_start + 6])
    lean_sine = heading_velocity(quat, torso_up_world, torso_angular_world)[..., 1]
    torso_rates = heading_velocity(quat, torso_angular_world, torso_angular_world)[..., :2]
    return velocity_reward(command_velocity, command, action, last_action,
                           full_state[..., 7:7 + joints], joint_limits, lean_sine,
                           torso_rates, dt=dt)


def simulator_full_state(env):
    data = env.scene["robot"].data
    return torch.cat((data.root_link_pos_w - env.scene.env_origins,
                      data.root_link_quat_w, data.joint_pos,
                      data.root_link_lin_vel_w, data.root_link_ang_vel_w,
                      data.joint_vel), -1)


def motion_only_term(env, original_func, **params):
    value = original_func(env, **params)
    return torch.where(velocity_mask(env), torch.zeros_like(value), value)


def simulator_velocity_reward_rate(env):
    robot = env.scene["robot"]
    torso = robot.body_names.index("torso_link")
    up = matrix_from_quat(robot.data.body_link_quat_w[:, torso])[..., :, 2]
    angular = robot.data.body_link_ang_vel_w[:, torso]
    value = full_state_velocity_reward(simulator_full_state(env), up, angular,
        env._paper_velocity_command, env.action_manager.action,
        env.action_manager.prev_action, robot.data.soft_joint_pos_limits, dt=1.0)
    return torch.where(velocity_mask(env), value, torch.zeros_like(value))


def simulator_velocity_fall(env):
    full = simulator_full_state(env)
    return velocity_mask(env) & velocity_fall(full[:, 2], root_gravity_z(full))


def configure_mixed_environment(cfg, mode):
    if mode not in ("rl", "distill"):
        raise ValueError("Velocity mode must be rl or distill")
    if mode == "distill":
        return cfg
    if not cfg.scale_rewards_by_dt:
        raise ValueError("Table A.6 needs the native dt-scaled reward manager")
    failure_names = {name for name, term in cfg.terminations.items() if not term.time_out}
    if failure_names != {"anchor_pos", "anchor_ori"}:
        raise ValueError("Review changed native failure terms before routing velocity slots")
    for term in cfg.rewards.values():
        if inspect.isclass(term.func) or "original_func" in term.params:
            raise ValueError("Unsupported native reward wrapper")
        term.params = dict(term.params, original_func=term.func)
        term.func = motion_only_term
    for name in failure_names:
        term = cfg.terminations[name]
        if inspect.isclass(term.func) or "original_func" in term.params:
            raise ValueError("Unsupported native termination wrapper")
        term.params = dict(term.params, original_func=term.func)
        term.func = motion_only_term
    cfg.rewards["paper_velocity"] = RewardTermCfg(func=simulator_velocity_reward_rate, weight=1.0)
    cfg.terminations["paper_velocity_fall"] = TerminationTermCfg(func=simulator_velocity_fall)
    # Native timeouts (including reference clip end) remain truncations; only
    # velocity failure termination is replaced by the Table A.6 fall condition.
    return cfg


class InterfaceReplayView:
    """Live read-only environment slice; cursor and fill count follow the parent."""

    def __init__(self, parent, lo, hi):
        if not 0 <= lo < hi <= parent.num_envs:
            raise ValueError("A nonempty simulator-interface slice is required")
        self.parent, self.lo, self.hi = parent, lo, hi
        self.num_envs, self.buffer_size, self.device = hi - lo, parent.buffer_size, parent.device

    @property
    def _storage(self):
        return self.parent._storage[self.lo:self.hi]

    @property
    def _step(self):
        return self.parent._step

    @property
    def num_transitions(self):
        return self.parent.num_transitions

    def sample(self, sequence_length, batch_size):
        return ReplayBuffer.sample(self, sequence_length, batch_size)

    def sample_causal(self, sequence_length, batch_size):
        return ReplayBuffer.sample_causal(self, sequence_length, batch_size)


class MixedVelocityWorldModelEnv(MultiTaskWorldModelEnv):
    def __init__(self, *args, velocity_mode, **kwargs):
        if velocity_mode not in ("rl", "distill"):
            raise ValueError("Velocity mode must be rl or distill")
        self.velocity_mode = velocity_mode
        super().__init__(*args, **kwargs)
        if self.num_envs % 2 or self.replay_buffer.num_envs % 2:
            raise ValueError("Both simulator and imagined batches must split evenly")
        self.reference_commands = ReferenceVelocityCommands(self.motion)
        self.command_schedule = VelocityCommandSchedule(self.num_envs, mode=velocity_mode, device=self.device)
        self.velocity_rows = torch.arange(self.num_envs, device=self.device) >= self.num_envs // 2
        split = self.replay_buffer.num_envs // 2
        self.replay_views = (InterfaceReplayView(self.replay_buffer, 0, split),
                             InterfaceReplayView(self.replay_buffer, split, 2 * split))
        env = kwargs["env"].unwrapped
        self.torso_index = env.command_manager.get_term("motion").cfg.body_names.index("torso_link")
        self.motion_reward_ceiling = env.step_dt * sum(
            term.weight for name, term in env.cfg.rewards.items()
            if name != "paper_velocity" and term.weight > 0)
        # A common valid upper bound for the inherited critic clipping, including
        # both interfaces. Their networks and goals remain independent.
        self.max_step_reward = max(self.motion_reward_ceiling, 4 * env.step_dt)

    def sample_initial_states(self):
        batches = []
        for view in self.replay_views:
            count = self.num_envs // 2
            if self.history_length == 1:
                batch = view.sample(1, count).squeeze(1)
            else:
                window, padded = view.sample_causal(self.history_length, count)
                batch = window[:, -1]
                batch["sensor_window"], batch["pad_mask"] = window["actor_obs"], padded
            batches.append(batch)
        combined = torch.cat(batches, dim=0)
        # This sampling happens outside the captured update, as in the native
        # learner. Check logged provenance rather than trusting row position.
        if not torch.equal(combined["velocity_interface"], self.velocity_rows):
            raise RuntimeError("Imagined start state came from the wrong simulator interface")
        return combined

    def _on_reset(self, batch):
        self._velocity_command = batch["velocity_command"].detach().clone()
        self.command_schedule.reference = batch["velocity_reference"].detach().clone()
        self.command_schedule.held = batch["velocity_held"].detach().clone()
        # Stored age is the current zero-based command age; that command has
        # already been served to the current observation at reset.
        self.command_schedule.age = batch["velocity_age"].detach().clone() + 1
        super()._on_reset(batch)

    def _build_task_goals(self):
        self._task_goal_obs = {"motion": self._actor_goal_obs, "velocity": self._velocity_command}

    def _compute_commands(self):
        super()._compute_commands()
        reference = self.reference_commands.lookup(self.motion_times, validate=False)
        self._velocity_command, _ = self.command_schedule.command(reference)
        self._build_task_goals()

    def _critic_obs(self):
        return interface_critic_input(self.rep_state, self._wm_goal_obs, self._velocity_command,
                                      self.sim_ctx, self.velocity_rows)

    def get_observations(self):
        obs = build_actor_obs_td(self._task_goal_obs, self._sensor_history(), critic=self._critic_obs())
        obs["velocity_reference_mask"] = self.command_schedule.reference
        return obs

    def _wm_step(self, rep_state, aux_state, action, last_action, sim_ctx):
        next_aux, next_rep = self.dynamics(rep_state, action, aux_state, sim_ctx)
        joints = self.representation.joint_state(next_rep)
        bodies = self.representation.world_frame_bodies(next_aux, next_rep)
        motion_reward = tracking_reward(next_aux, *bodies, joints, action, last_action,
            *self._ref_body_state, self.representation._soft_joint_pos_limits, self.representation.dt)
        motion_term = self._compute_term_prob(next_aux, *self._ref_body_state[:2])
        if self.velocity_mode == "distill":
            return next_aux, next_rep, joints, motion_term, motion_reward
        full = self.representation.reconstruct_full_state(next_aux, next_rep)
        torso_up = bodies[1][:, self.torso_index, :, 2]
        torso_angular = bodies[3][:, self.torso_index]
        velocity_value = full_state_velocity_reward(full, torso_up, torso_angular,
            self._velocity_command, action, last_action, self.representation._soft_joint_pos_limits,
            dt=self.representation.dt)
        height_margin = (0.35 - full[:, 2]) / 0.35
        tilt_margin = (1 + root_gravity_z(full) - 0.7) / 0.7
        fall_probability = torch.sigmoid(torch.maximum(height_margin, tilt_margin) / self._termination_softness)
        return (next_aux, next_rep, joints,
                torch.where(self.velocity_rows, fall_probability, motion_term),
                torch.where(self.velocity_rows, velocity_value, motion_reward))
