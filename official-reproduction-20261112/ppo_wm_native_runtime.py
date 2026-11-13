"""Native-network PPO-WM integration with explicit termination diagnostics.

Leaves vendor files unchanged. Neither storage choice establishes the paper's
termination convention. Reset sampling and PPO hyperparameters are recorded
separately from the main A5 runner.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from paper_ppo_models import NativeMotionPpoActor, NativePpoCritic


def state_sha256(module):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f'Non-tensor state: {name}')
        digest.update(f'{name}|{tensor.dtype}|{tuple(tensor.shape)}'.encode())
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class TerminationTrace:
    def __init__(self, source):
        self.source = source
        self.probabilities = []

    def reset(self):
        self.probabilities.clear()
        return self.source.reset()

    def step(self, actions):
        result = self.source.step(actions)
        probability = result[2].detach().clone()
        if not torch.isfinite(probability).all() or not ((probability >= 0) & (probability <= 1)).all():
            raise ValueError('World-model termination must be a finite probability')
        self.probabilities.append(probability)
        return result


def make_native_ppo_runner(native_cfg, termination_mode, trace_folder):
    from supermimic.models import build_mlp
    from supermimic.runners.ppo_runner import PpoRunner
    from supermimic.runners.supermimic_runner import build_actor
    if termination_mode not in ('released_uint8', 'diagnostic_float'):
        raise ValueError('Explicit termination mode required')
    trace_folder = Path(trace_folder)

    class NativeNetworkPpoRunner(PpoRunner):
        def _build_mlp(self, name, input_dim, output_dim, hidden_dims, **kwargs):
            if name == 'actor':
                model = NativeMotionPpoActor(build_actor(native_cfg, self._wm, self.device),
                    action_dim=output_dim, device=self.device, init_std=self.cfg.init_noise_std)
                if model.mlp.input_width != input_dim:
                    raise ValueError('Actual full motion-goal/history width mismatch')
                return model
            if name == 'critic':
                return NativePpoCritic(build_mlp(input_dim, native_cfg.critic_hidden_dims, output_dim).to(self.device),
                    input_dim=input_dim, device=self.device)
            raise ValueError(name)

        def _init_policy(self):
            super()._init_policy()
            if termination_mode == 'diagnostic_float':
                self.alg.storage.dones = torch.zeros_like(self.alg.storage.dones, dtype=torch.float32)
            expected = torch.uint8 if termination_mode == 'released_uint8' else torch.float32
            assert self.alg.storage.dones.dtype == expected
            self.wm_env = TerminationTrace(self.wm_env)
            self.integration_updates = []

        def imagine(self):
            actor_before = state_sha256(self.alg.actor.native_actor)
            critic_before = state_sha256(self.alg.critic.native_critic)
            losses = super().imagine()
            if not losses or not all(np.isfinite(float(value)) for value in losses.values()):
                raise ValueError('Nonfinite/empty PPO update losses')
            for module in (self.alg.actor, self.alg.critic):
                if not all(torch.isfinite(p).all() for p in module.parameters()):
                    raise ValueError('Nonfinite PPO parameter')
            probabilities = torch.stack(self.wm_env.probabilities)
            stored = self.alg.storage.dones[..., 0]
            assert probabilities.shape == stored.shape == (self.cfg.num_steps_per_env, self.cfg.batch_size)
            assert torch.equal(stored, probabilities.to(stored.dtype))
            actor_after = state_sha256(self.alg.actor.native_actor)
            critic_after = state_sha256(self.alg.critic.native_critic)
            if actor_after == actor_before or critic_after == critic_before:
                raise ValueError('Native actor and critic must both change in each PPO update')
            trace_folder.mkdir(parents=True,exist_ok=True)
            trace = trace_folder / f'update-{len(self.integration_updates)+1:03d}.npz'
            np.savez_compressed(trace, probabilities=probabilities.cpu().numpy(),
                stored_termination=stored.cpu().numpy())
            with np.load(trace, allow_pickle=False) as saved:
                assert np.array_equal(saved['probabilities'], probabilities.cpu().numpy())
                assert np.array_equal(saved['stored_termination'], stored.cpu().numpy())
            fractional = (probabilities > 0) & (probabilities < 1)
            row = dict(update=len(self.integration_updates)+1, termination_mode=termination_mode,
                storage_dtype=str(stored.dtype), imagined_steps=int(probabilities.numel()),
                fractional_probability_positions=int(fractional.sum()),
                fractional_positions_truncated_to_zero=int((fractional & (stored == 0)).sum()),
                stored_matches_declared_conversion=True,
                probability_min=float(probabilities.min()), probability_max=float(probabilities.max()),
                probability_mean=float(probabilities.mean()),
                actor_before=actor_before, actor_after=actor_after,
                critic_before=critic_before, critic_after=critic_after,
                actor_and_critic_changed=True,
                losses={k:float(v) for k,v in losses.items()},
                trace=dict(path=str(trace), bytes=trace.stat().st_size,
                    sha256=hashlib.sha256(trace.read_bytes()).hexdigest(), readback_verified=True))
            self.integration_updates.append(row)
            print('PPO_WM_NATIVE_UPDATE ' + json.dumps(row),flush=True)
            return losses

    return NativeNetworkPpoRunner
