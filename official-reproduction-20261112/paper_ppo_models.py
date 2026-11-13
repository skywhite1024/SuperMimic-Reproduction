"""RSL model interfaces around the exact native motion actor and critic.

Keeps the native goal encoder, latent, history decoder, LayerNorm/GELU and
normalizers. Adds a Gaussian distribution for PPO sampling. This module alone
does not supply a paper-equivalent PPO run, termination convention or sampler.
"""

import math

import torch
from torch import nn
from tensordict import TensorDict
from rsl_rl.models import MLPModel

from supermimic.tasks.task_spec import build_actor_obs_td


class FlatMotionActor(nn.Module):
    """Undo BaseRunner's [flattened sensor history, motion goal] layout."""

    def __init__(self, actor):
        super().__init__()
        active = [spec.name for spec, fraction in zip(actor.task_specs, actor._slot_fractions, strict=True)
                  if fraction != 0.]
        if active != ['motion']:
            raise ValueError('Only the native motion-only actor is supported')
        self.actor = actor
        self.goal_width = next(spec.goal_dim for spec in actor.task_specs if spec.name == 'motion')
        self.history_width = actor.history_length * actor.obs_dim
        self.input_width = self.history_width + self.goal_width

    def as_native_observation(self, flat):
        if flat.ndim != 2 or flat.shape[-1] != self.input_width:
            raise ValueError('Expected a flat batch with the exact native history/goal width')
        history = flat[:, :self.history_width].reshape(-1, self.actor.history_length, self.actor.obs_dim)
        goal = flat[:, self.history_width:]
        return build_actor_obs_td({'motion': goal}, history)

    def forward(self, flat):
        return self.actor(self.as_native_observation(flat))


class NativeMotionPpoActor(MLPModel):
    def __init__(self, actor, *, action_dim, device='cpu', init_std=1.):
        if not math.isfinite(init_std) or init_std <= 0:
            raise ValueError('PPO standard deviation must be positive and finite')
        if action_dim != actor.decoder[-1].out_features:
            raise ValueError('Distribution width must match the native actor output')
        bridge = FlatMotionActor(actor)
        template = TensorDict({'actor': torch.zeros(1, bridge.input_width, device=device)}, batch_size=[1])
        # Installed RSL requires a nonempty hidden_dims list. The tiny temporary
        # network is discarded, and its initialization must not advance RNG.
        with torch.random.fork_rng(devices=[]):
            super().__init__(template, {'actor':['actor']}, 'actor', action_dim,
                hidden_dims=(1,), obs_normalization=False,
                distribution_cfg={'class_name':'rsl_rl.modules.distribution.GaussianDistribution',
                                  'init_std':init_std})
        # Replace the temporary RSL MLP; the native actor is the only mean network.
        self.mlp = bridge
        self.to(device)

    @property
    def native_actor(self):
        return self.mlp.actor

    def update_normalization(self, obs):
        self.native_actor.update_normalization(self.mlp.as_native_observation(obs['actor']))


class NativePpoCritic(MLPModel):
    def __init__(self, critic, *, input_dim, device='cpu'):
        template = TensorDict({'critic': torch.zeros(1, input_dim, device=device)}, batch_size=[1])
        with torch.random.fork_rng(devices=[]):
            super().__init__(template, {'critic':['critic']}, 'critic', 1,
                             hidden_dims=(1,), obs_normalization=False)
        # Main-method critic consumes raw privileged state/goal/context without
        # an additional RSL input normalizer. Preserve that exact function.
        self.mlp = critic
        self.to(device)

    @property
    def native_critic(self):
        return self.mlp
