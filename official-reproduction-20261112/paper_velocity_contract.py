"""Explicit velocity-task rules from Table A.6 and Appendix G.

This module supplies commands, rewards, and step-test accounting for the local
velocity adapter. It is not a complete training runner. The reference window
clamps within a clip, an explicit boundary choice absent from the paper text.
"""

import math

import numpy as np
import torch


COMMAND_LOW = (-1.5, -1.32, -3.35)
COMMAND_HIGH = (2.0, 1.32, 3.35)


def heading_velocity(root_quat_w, linear_velocity_w, angular_velocity_w):
    """Yaw-only planar linear velocity and world yaw rate; quaternions are wxyz."""
    w, x, y, z = root_quat_w.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))
    c, s = torch.cos(yaw), torch.sin(yaw)
    vx, vy = linear_velocity_w[..., 0], linear_velocity_w[..., 1]
    return torch.stack((c * vx + s * vy, -s * vx + c * vy, angular_velocity_w[..., 2]), -1)


class ReferenceVelocityCommands:
    """Centered mean of framewise reference-heading commands, without range clipping.

    Projection before averaging is the local reading of Appendix G. Clip-edge
    repetition is an explicit implementation choice, not an author-confirmed rule.
    """

    def __init__(self, motion, window_frames=25):
        if window_frames < 1 or window_frames % 2 != 1:
            raise ValueError("A positive odd centered reference window is required")
        self.motion, self.window_frames = motion, window_frames
        ends = motion.motion_ends
        if (ends.ndim != 1 or ends.dtype != torch.long or not len(ends)
                or not bool(torch.all(torch.diff(ends, prepend=ends.new_zeros(1)) > 0))):
            raise ValueError("Positive increasing int64 motion ends are required")
        n = int(ends[-1])
        if (motion.root_quat_w.shape != (n, 4) or motion.root_lin_vel_w.shape != (n, 3)
                or motion.root_ang_vel_w.shape != (n, 3)):
            raise ValueError("Reference velocity/quaternion arrays disagree with motion ends")
        self.starts = torch.cat((ends.new_zeros(1), ends[:-1]))

    def lookup(self, times, *, validate=True):
        if times.ndim != 1 or times.dtype != torch.long:
            raise ValueError("Reference times must be an int64 vector")
        ends = self.motion.motion_ends
        if validate and not bool(torch.all((times >= 0) & (times < ends[-1]))):
            raise ValueError("Reference times outside complete clip membership")
        clip = torch.searchsorted(ends, times, right=True)
        offsets = torch.arange(self.window_frames, device=times.device) - self.window_frames // 2
        idx = times[:, None] + offsets
        idx = torch.minimum(torch.maximum(idx, self.starts[clip, None]), ends[clip, None] - 1)
        command = heading_velocity(self.motion.root_quat_w[idx], self.motion.root_lin_vel_w[idx],
                                   self.motion.root_ang_vel_w[idx])
        return command.mean(dim=1)


class VelocityCommandSchedule:
    """Reference or held random command, resampled on reset and every 250 steps.

    ``command`` is called once before each control step. Reference commands are
    refreshed every call and are not clipped to the uniform sampling ranges.
    Distillation always uses reference commands. State must be logged separately
    for replay; this object does not implement full training resume.
    """

    def __init__(self, num_envs, *, mode="rl", device="cpu", period=250,
                 reference_probability=0.5, standing_probability=0.1):
        if mode not in ("rl", "distill") or num_envs < 1 or period < 1:
            raise ValueError("Invalid velocity command schedule")
        if not 0 <= reference_probability <= 1 or not 0 <= standing_probability <= 1:
            raise ValueError("Invalid command probabilities")
        self.mode, self.period = mode, period
        self.reference_probability = reference_probability
        self.standing_probability = standing_probability
        self.age = torch.full((num_envs,), period, dtype=torch.long, device=device)
        self.reference = torch.ones(num_envs, dtype=torch.bool, device=device)
        self.held = torch.zeros(num_envs, 3, device=device)
        self.low = torch.tensor(COMMAND_LOW, device=device)
        self.high = torch.tensor(COMMAND_HIGH, device=device)

    def command(self, reference_command, reset_mask=None):
        if reference_command.shape != self.held.shape:
            raise ValueError("One 3D reference command per environment is required")
        reset = torch.zeros_like(self.reference) if reset_mask is None else reset_mask
        if reset.shape != self.reference.shape or reset.dtype != torch.bool:
            raise ValueError("Reset mask must be one boolean per environment")
        due = reset | (self.age >= self.period)
        if self.mode == "distill":
            source = torch.ones_like(self.reference)
            sampled = self.held
        else:
            source = torch.rand_like(self.held[:, 0]) < self.reference_probability
            sampled = self.low + torch.rand_like(self.held) * (self.high - self.low)
            standing = torch.rand_like(self.held[:, 0]) < self.standing_probability
            sampled = torch.where(standing[:, None], torch.zeros_like(sampled), sampled)
        self.reference = torch.where(due, source, self.reference)
        self.held = torch.where(due[:, None], sampled, self.held)
        self.age = torch.where(due, torch.zeros_like(self.age), self.age) + 1
        result = torch.where(self.reference[:, None], reference_command.detach(), self.held)
        return result, self.reference.clone()


def velocity_reward(velocity_heading, command, action, last_action, joint_pos,
                    joint_limits, torso_sideways_sine, torso_roll_pitch_rates_heading,
                    *, dt=0.02):
    """Table A.6; callers supply the torso geometry under an explicit convention."""
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("A positive finite control period is required")
    linear = torch.exp(-(velocity_heading[..., :2] - command[..., :2]).square().sum(-1) / 0.5**2)
    yaw = torch.exp(-(velocity_heading[..., 2] - command[..., 2]).square() / 0.5)
    smooth = (action - last_action).square().sum(-1)
    limits = ((joint_pos - joint_limits[..., 1]).clamp_min(0)
              + (joint_limits[..., 0] - joint_pos).clamp_min(0)).sum(-1)
    lean = torso_sideways_sine.square()
    roll_pitch = torso_roll_pitch_rates_heading.square().sum(-1)
    return dt * (2 * linear + 2 * yaw - 0.3 * smooth - 2 * limits - 4 * lean - 0.05 * roll_pitch)


def velocity_fall(root_height, root_gravity_z):
    """Table A.6 hard fall condition; equality alone is not a fall."""
    return (root_height < 0.35) | (1 + root_gravity_z > 0.7)


def step_command_grid():
    """672 plane/grid identities; coincident axis commands are intentionally retained."""
    axes = [np.linspace(lo, hi, 15, dtype=np.float64) for lo, hi in zip(COMMAND_LOW, COMMAND_HIGH)]
    rows = []
    for name, first, second in (("vx-vy", 0, 1), ("vx-wz", 0, 2), ("vy-wz", 1, 2)):
        for i, a in enumerate(axes[first]):
            for j, b in enumerate(axes[second]):
                command = np.zeros(3, dtype=np.float64)
                command[first], command[second] = a, b
                if np.linalg.norm(command) < 1e-12:
                    continue
                command[np.abs(command) < 1e-12] = 0
                rows.append({"plane": name, "row": i, "column": j, "command": command.tolist()})
    assert len(rows) == 672
    return rows


def aggregate_step_trials(command, measured_velocity, fallen, *, fps=50, cap=0.9):
    """Four 5s trials, mean vector error over final 2s, median of non-falling trials.

    Measured velocities are in the heading/world-yaw convention, after each
    control step. Capping is display-only. All-fall has no finite raw error.
    """
    command = np.asarray(command, dtype=np.float64)
    measured = np.asarray(measured_velocity, dtype=np.float64)
    fallen = np.asarray(fallen)
    if fps < 1 or measured.shape != (4, 5 * fps, 3) or fallen.shape != (4,) or fallen.dtype != np.bool_:
        raise ValueError("Exactly four five-second trials and four boolean fall flags are required")
    if command.shape != (3,) or not np.isfinite(command).all() or np.linalg.norm(command) <= 0:
        raise ValueError("A finite non-standing command is required")
    if not math.isfinite(cap) or cap <= 0:
        raise ValueError("A positive finite display cap is required")
    errors = []
    for velocities, fell in zip(measured, fallen):
        if fell:
            errors.append(None)
            continue
        if not np.isfinite(velocities).all():
            raise ValueError("A non-falling trial contains nonfinite velocity data")
        mean_error = (velocities[-2 * fps:] - command).mean(axis=0, dtype=np.float64)
        errors.append(float(np.linalg.norm(mean_error) / np.linalg.norm(command)))
    valid = [error for error in errors if error is not None]
    median = float(np.median(valid)) if valid else None
    return {"trial_errors": errors, "non_falling_trials": len(valid), "all_trials_fell": not valid,
            "median_relative_error": median, "display_relative_error": min(median, cap) if valid else cap}
