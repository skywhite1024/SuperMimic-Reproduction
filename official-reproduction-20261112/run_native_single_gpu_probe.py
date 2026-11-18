"""Measure the one-H100 eager control for the two-rank native BONES timing."""

import argparse
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

from push_recovery_contract import sha256, write_json
from run_native_two_gpu_gate import STEADY_WINDOW, parameter_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motions", type=Path, required=True)
    parser.add_argument("--membership-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    motions = args.motions.resolve()
    pack = json.loads(motions.with_name("full-duration-pack.json").read_bytes())
    if (pack["clips"], pack["frames"], pack["membership_sha256"]) != (129543, 47374271, args.membership_sha256):
        raise ValueError("The complete verified BONES proxy pack is required")
    if {p.name: sha256(p) for p in sorted(motions.iterdir()) if p.is_file()} != pack["files"]:
        raise ValueError("Motion files differ from the two-rank input pack")
    os.environ.update(MUJOCO_GL="disable", WANDB_MODE="offline")

    import numpy as np
    import torch
    import supermimic.learners.policy_learner as policy_module
    import supermimic.learners.wm_learner as wm_module
    import supermimic.representations as representations
    from supermimic.runners.supermimic_runner import Ratio, SuperMimicRunner
    from supermimic.scripts.train import TrainConfig
    from supermimic.scripts.train_utils import create_env, setup_device
    from paper_profile import build_adam_optimizers, configure_runner, cosine_lr, set_learning_rate_in_place

    if torch.cuda.device_count() != 1 or "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Exactly one H100 is required")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.chdir(output)
    task, num_envs = "SuperMimic-GenCoord-G1", 4096
    warmup, steps = STEADY_WINDOW
    lr_multiplier = math.sqrt(2)
    cfg = TrainConfig.from_task(task)
    configure_runner(cfg.runner)
    cfg.runner.max_iterations = steps
    cfg.runner.wm.lr *= lr_multiplier
    cfg.runner.policy.lr *= lr_multiplier
    cfg.env.seed = 42
    cfg.env.scene.num_envs = num_envs
    cfg.env.commands["motion"].motion_file = str(motions)
    cfg.env.commands["motion"].sampling_mode = "external"
    representations._compiled_assemble_goal = representations._assemble_goal
    policy_module.build_optimizers = build_adam_optimizers
    wm_module.build_optimizers = build_adam_optimizers
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    device = setup_device("cuda:0")
    env = create_env(cfg.env, num_envs, device)
    try:
        torch.manual_seed(42)
        runner = SuperMimicRunner(env, cfg.runner, task, device=device)
        wm, policy = runner.world_model_learner, runner.policy_learner
        for learner in (wm, policy):
            learner._captured_update = learner._update_from
        models = {"dynamics": runner._dynamics, "actor": policy.actor, "critic": policy.critic}
        original = {name: parameter_hash(model) for name, model in models.items()}
        optimizers = [*wm.optimizers, *policy.actor_optimizers, *policy.critic_optimizers]
        assert len(optimizers) == 3 and all(type(item) is torch.optim.Adam for item in optimizers)
        torch.manual_seed(42)
        env.reset()
        runner._seed_sim_history()
        wm_cfg = cfg.runner.wm
        ratio = Ratio(num_envs / wm_cfg.num_batches / wm_cfg.batch_size / wm_cfg.horizon * cfg.runner.replay_ratio)
        updates = transitions = 0
        steady_start = None
        for step in range(steps):
            set_learning_rate_in_place(optimizers, lr_multiplier * cosine_lr(step))
            transitions += runner.step()
            for _ in range(ratio(step)):
                if runner.replay_buffer.num_transitions < wm_cfg.horizon:
                    continue
                samples = (runner.replay_buffer.sample(wm_cfg.horizon, wm_cfg.batch_size)
                           for _ in range(wm_cfg.num_batches))
                wm_logs = wm.update(samples)
                policy_logs = policy.update()
                assert all(torch.isfinite(value).all() for value in (*wm_logs.values(), *policy_logs.values()))
                updates += 1
            if step + 1 == warmup:
                print("NATIVE_SINGLE_GPU_STEADY_WINDOW_START", flush=True)
                torch.cuda.synchronize(device)
                steady_start = time.monotonic()
        torch.cuda.synchronize(device)
        elapsed = time.monotonic() - steady_start
        final = {name: parameter_hash(model) for name, model in models.items()}
        assert all(final[name] != original[name] for name in models)
        assert transitions == num_envs * steps and updates == 398
        assert int(policy.actor.obs_normalizer.count) == transitions
        report = {"status": "native_single_H100_eager_control_completed", "gpu": torch.cuda.get_device_name(0),
                  "seed": 42, "num_envs": num_envs, "local_collections": steps, "real_transitions": transitions,
                  "outer_learner_updates": updates, "learner_CUDA_graphs_disabled": True,
                  "learning_rate_multiplier": lr_multiplier, "membership_sha256": args.membership_sha256,
                  "native_models_changed": True, "first_excluded_collection": warmup,
                  "last_included_collection": steps, "seconds": elapsed,
                  "steady_real_transitions": num_envs * (steps - warmup),
                  "steady_transitions_per_second": num_envs * (steps - warmup) / elapsed,
                  "quality_evaluated": False, "paper_performance_equivalence": False}
        receipt = output / "result.json"
        write_json(receipt, report)
        script = Path(__file__).resolve().with_name("emit_json_receipts.py")
        subprocess.run([sys.executable, str(script), str(receipt)], check=True, timeout=60)
        print("NATIVE_SINGLE_GPU_EAGER_COMPLETED " + json.dumps(report), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
