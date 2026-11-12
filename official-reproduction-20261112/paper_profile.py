"""Explicit single-GPU adapter for paper Tables A.4/A.5, outside vendor code.

This is our interpretation of the published tables, not an author-provided
experiment configuration. The schedule counts simulator collection steps; the
paper's iteration convention and complete benchmark manifest remain unresolved.
"""

import math

import torch


PROFILE_NAME = "paper-a5-single-gpu"
INITIAL_LR = 2e-4
DECAY_STEPS = 250_000
FINAL_LR_RATIO = 0.03


def profile_description():
    return {
        "name": PROFILE_NAME,
        "provenance": "local interpretation of paper Tables A.4/A.5",
        "optimizer": "Adam for every trainable network parameter; no Muon",
        "actor_betas": [0.9, 0.95],
        "wm_and_critic_betas": [0.9, 0.999],
        "initial_lr": INITIAL_LR,
        "schedule": "cosine to 0.03 times initial by collection step 250000; then constant",
        "schedule_unit": "zero-based simulator collection step, before its learner updates",
        "schedule_unit_author_confirmed": False,
        "actor_observation": "proprio",
        "history_length": 10,
        "gpus": 1,
        "paper_training_length": 320_000,
        "limitations": [
            "Does not supply the missing paper dataset filters or surviving-clip manifest",
            "Does not implement multi-GPU gradient synchronization",
            "Does not implement full training-state resume",
            "Motion training only; velocity encoder architecture not aligned here",
            "Uses the released simulator, normalization and return-sampler implementation",
        ],
        "paper_equivalence_verified": False,
    }


def configure_runner(cfg):
    """Set the unambiguous single-GPU motion-training table entries."""
    cfg.actor_obs = "proprio"
    cfg.history_length = 10
    cfg.buffer_size = 1000
    cfg.dynamics_hidden_dims = (2048, 2048, 2048)
    cfg.actor_hidden_dims = (2048, 1024, 512, 256)
    cfg.critic_hidden_dims = (256, 128)
    cfg.latent_dim = 32
    cfg.slot_mix = {"motion": 1.0, "velocity": 0.0}
    cfg.wm.lr = cfg.policy.lr = INITIAL_LR
    cfg.wm.num_batches, cfg.wm.batch_size, cfg.wm.horizon = 10, 2048, 8
    cfg.policy.num_batches, cfg.policy.batch_size, cfg.policy.horizon = 1, 4096, 32
    cfg.policy.beta2, cfg.policy.gamma, cfg.policy.lam = 0.95, 0.99, 0.95
    cfg.policy.noise, cfg.policy.grad_clip_value = 0.1, 0.1
    # Author's ratio convention: 4096 / (10 * 2048 * 8) * 16 == 0.4.
    cfg.replay_ratio = 16


def build_adam_optimizers(modules, lr, betas=(0.9, 0.999), device="cpu"):
    """Cover each parameter once, with CUDA-graph-safe tensor learning rates."""
    params, seen = [], set()
    for module in modules:
        for parameter in module.parameters():
            if id(parameter) not in seen:
                seen.add(id(parameter))
                params.append(parameter)
    cuda = torch.device(device).type == "cuda"
    return [torch.optim.Adam(
        params, lr=torch.tensor(lr, dtype=torch.float32, device=device),
        betas=betas, fused=cuda, capturable=cuda,
    )]


def cosine_lr(collection_step):
    if not isinstance(collection_step, int) or collection_step < 0:
        raise ValueError("collection_step must be a nonnegative integer")
    progress = min(collection_step, DECAY_STEPS) / DECAY_STEPS
    return INITIAL_LR * (
        FINAL_LR_RATIO + (1.0 - FINAL_LR_RATIO) * (1.0 + math.cos(math.pi * progress)) / 2
    )


def set_learning_rate_in_place(optimizers, lr):
    """Do not replace a tensor referenced by an already captured CUDA graph."""
    with torch.no_grad():
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                current = group["lr"]
                if not isinstance(current, torch.Tensor) or current.ndim != 0:
                    raise TypeError("Paper adapter requires scalar tensor learning rates")
                current.fill_(lr)


def install_paper_runner(base_runner):
    """Install both imported optimizer bindings; return an isolated runner class."""
    import supermimic.learners.policy_learner as policy_module
    import supermimic.learners.wm_learner as wm_module

    policy_module.build_optimizers = build_adam_optimizers
    wm_module.build_optimizers = build_adam_optimizers

    class PaperRunner(base_runner):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._paper_step = 0
            self._paper_optimizers = [
                *self.world_model_learner.optimizers,
                *self.policy_learner.actor_optimizers,
                *self.policy_learner.critic_optimizers,
            ]
            if len(self._paper_optimizers) != 3 or any(
                type(opt) is not torch.optim.Adam for opt in self._paper_optimizers
            ):
                raise RuntimeError("Expected exactly three all-Adam learners")

        def step(self):
            step = self._paper_step
            lr = cosine_lr(step)
            set_learning_rate_in_place(self._paper_optimizers, lr)
            if step % 1000 == 0:
                print(f"PAPER_PROFILE collection_step={step} lr={lr:.12g}", flush=True)
            result = super().step()
            self._paper_step += 1
            return result

    return PaperRunner
