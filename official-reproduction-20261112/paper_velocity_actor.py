"""Velocity actor/critic components for the local Appendix G adapter.

The history feature is explicitly chosen as flattened normalized proprioceptive
history. The paper does not specify its feature extractor. Simulator command
collection, interface-specific replay starts, and reward routing need a separate
runner; these components alone do not reproduce the velocity experiment.
"""

import torch
from torch import nn
from torch.func import functional_call

from supermimic.models import MultiEncoderAE, build_mlp


class PaperVelocityActor(MultiEncoderAE):
    """Motion-trained decoder; downstream return gradients update only its encoder.

    Detaching decoder parameters preserves the derivative with respect to the
    latent and observation inputs. It also freezes LayerNorm affine parameters.
    Reference-only latent alignment is a separate encoder-only gradient path.
    """

    def __init__(self, obs_dim, action_dim, task_specs, slot_mix, *, history_length=10,
                 latent_dim=32, motion_hidden_dims=(2048, 1024, 512, 256),
                 velocity_hidden_dims=(256, 256), latent_bc_weight=0.1):
        if [task.name for task in task_specs] != ["motion", "velocity"]:
            raise ValueError("Explicit ordered motion and velocity task specs are required")
        if task_specs[1].goal_dim != 3 or history_length < 1:
            raise ValueError("Velocity requires a 3D command and nonempty observation history")
        super().__init__(obs_dim, action_dim, task_specs, slot_mix, latent_dim,
                         motion_hidden_dims, latent_bc_weight, history_length)
        self.encoders["velocity"] = build_mlp(3 + history_length * obs_dim,
                                               velocity_hidden_dims, latent_dim)
        self.history_feature = "flattened_normalized_proprioception"
        self.history_feature_author_confirmed = False

    def _velocity_latent(self, goal, history):
        return torch.tanh(self.encoders["velocity"](torch.cat((goal, history.flatten(1)), -1)))

    def forward(self, obs_td):
        slots = self.slot_ranges(obs_td.shape[0])
        history = self.obs_normalizer(obs_td["history"])
        outputs = []
        for name, (lo, hi) in zip(self._task_names, slots, strict=True):
            if lo == hi:
                continue
            goal = self.goal_normalizers[name](obs_td[f"goal_{name}"][lo:hi])
            z = (torch.tanh(self.encoders[name](goal)) if name == "motion"
                 else self._velocity_latent(goal, history[lo:hi]))
            decoder_input = torch.cat((z, history[lo:hi].flatten(1)), -1)
            if name == "motion":
                outputs.append(self.decoder(decoder_input))
            else:
                frozen = {key: value.detach() for key, value in self.decoder.named_parameters()}
                outputs.append(functional_call(self.decoder, frozen, (decoder_input,)))
        self._aux_loss = self._reference_alignment(obs_td, slots, history)
        return torch.cat(outputs)

    def _reference_alignment(self, obs_td, slots, normalized_history):
        result = torch.zeros(obs_td.shape[0], device=obs_td.device)
        lo, hi = slots[1]
        if not self.training or self.latent_bc_weight == 0 or lo == hi:
            return result
        mask = obs_td["velocity_reference_mask"]
        if mask.shape != (obs_td.shape[0],) or mask.dtype != torch.bool:
            raise ValueError("Velocity reference mask must identify each sampled command source")
        command = self.goal_normalizers["velocity"](obs_td["goal_velocity"][lo:hi]).detach()
        latent = self._velocity_latent(command, normalized_history[lo:hi].detach())
        with torch.no_grad():
            goal = self.goal_normalizers["motion"](obs_td["goal_motion"][lo:hi])
            teacher = torch.tanh(self.encoders["motion"](goal))
        result[lo:hi] = self.latent_bc_weight * (latent - teacher).square().mean(-1) * mask[lo:hi]
        return result


def interface_critic_input(rep_state, motion_goal, velocity_command, sim_context, velocity_mask):
    """Pack the task goal and an explicit route bit that survives replay shuffling."""
    if motion_goal.shape[-1] < 3 or velocity_command.shape[-1] != 3:
        raise ValueError("Motion goal must accommodate the 3D velocity command")
    if velocity_mask.dtype != torch.bool or velocity_mask.shape != rep_state.shape[:-1]:
        raise ValueError("Explicit per-sample boolean interface identity is required")
    padded_velocity = torch.cat((velocity_command, torch.zeros_like(motion_goal[..., 3:])), -1)
    goal = torch.where(velocity_mask[..., None], padded_velocity, motion_goal)
    return torch.cat((rep_state, goal, sim_context, velocity_mask[..., None].to(rep_state.dtype)), -1)


class InterfaceCritic(nn.Module):
    """Independent value networks with task-specific goals, robust to time/batch shuffle.

    The route bit selects the network and is not a network input. Velocity sees
    only its three command values; it cannot read the padded motion-goal tail.
    """

    def __init__(self, rep_state_dim, motion_goal_dim, sim_context_dim, hidden_dims=(256, 128)):
        super().__init__()
        if min(rep_state_dim, motion_goal_dim, sim_context_dim) < 1 or motion_goal_dim < 3:
            raise ValueError("Positive state/context and at least three motion-goal dimensions required")
        self.rep_state_dim, self.motion_goal_dim = rep_state_dim, motion_goal_dim
        self.sim_context_dim = sim_context_dim
        self.input_dim = rep_state_dim + motion_goal_dim + sim_context_dim + 1
        self.motion = build_mlp(rep_state_dim + motion_goal_dim + sim_context_dim, hidden_dims, 1)
        self.velocity = build_mlp(rep_state_dim + 3 + sim_context_dim, hidden_dims, 1)

    def forward(self, observation):
        if observation.shape[-1] != self.input_dim:
            raise ValueError("Critic input does not include the explicit interface route bit")
        state_end, goal_end = self.rep_state_dim, self.rep_state_dim + self.motion_goal_dim
        velocity_input = torch.cat((observation[..., :state_end],
                                    observation[..., state_end:state_end + 3],
                                    observation[..., goal_end:-1]), -1)
        motion_value = self.motion(observation[..., :-1])
        velocity_value = self.velocity(velocity_input)
        return torch.where(observation[..., -1:] > 0.5, velocity_value, motion_value)
