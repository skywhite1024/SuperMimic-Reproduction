"""Finite CUDA pilot of the unchanged official default learner on raw LAFAN40.

This verifies execution and checkpoint creation, not the paper's full benchmark.
"""

import argparse
import dataclasses
import hashlib
import json
import os
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1001)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--goal-backend", choices=("official", "eager"), default="official",
                        help="Diagnostic: eager runs the author's same goal math without torch.compile")
    args = parser.parse_args()
    if args.steps < 2:
        parser.error("At least two simulator collection steps are required")
    motions = args.motions.resolve()
    output = args.output.resolve()
    if not (motions / "motion_ends.npy").is_file():
        raise FileNotFoundError("Prepared packed motions are required")
    output.mkdir(parents=True, exist_ok=False)
    os.environ["WANDB_MODE"] = "offline"
    os.environ["MUJOCO_GL"] = "disable"
    os.chdir(output)

    import numpy as np
    import torch
    import wandb
    import supermimic.runners.supermimic_runner as runner_module
    from supermimic.runners.supermimic_runner import SuperMimicRunner
    from supermimic.scripts.train import TrainConfig
    from supermimic.scripts.train_utils import create_env, setup_device

    if args.goal_backend == "eager":
        import supermimic.representations as representations
        representations._compiled_assemble_goal = representations._assemble_goal
        print("DIAGNOSTIC_OVERRIDE goal assembly uses author eager function", flush=True)

    # The author's artifact helper contacts the online API to prune versions
    # even in offline mode. Keep its local artifact logging, with no API call.
    def log_offline_checkpoint(files):
        assert wandb.run is not None
        artifact = wandb.Artifact(f"model-{wandb.run.id}", type="model")
        for path in files:
            artifact.add_file(path)
        wandb.run.log_artifact(artifact)

    runner_module.log_model_artifact = log_offline_checkpoint

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Pilot requires exactly one allocated CUDA GPU")
    if len(np.load(motions / "motion_ends.npy")) != 40:
        raise RuntimeError("Pilot denominator must be the frozen 40 clips")
    task = "SuperMimic-GenCoord-G1"
    config = TrainConfig.from_task(task)
    config.env.seed = args.seed
    config.env.commands["motion"].motion_file = str(motions)
    config.env.commands["motion"].sampling_mode = "external"
    config.runner.max_iterations = args.steps
    config.runner.save_interval = args.steps - 1  # save both initial and final step
    config.runner.run_name = f"official-lafan40-pilot-seed{args.seed}"
    config.runner.log_interval = 10
    resolved = {
        "task": task, "seed": args.seed, "num_envs": args.num_envs,
        "collection_steps": args.steps, "motion_directory": str(motions),
        "runner": dataclasses.asdict(config.runner),
        "environment": dataclasses.asdict(config.env),
        "source_algorithm_modified": False,
        "goal_assembly_backend": args.goal_backend,
        "runtime_execution_modified": args.goal_backend != "official",
        "io_adapter": "offline checkpoint artifact logging; online pruning disabled",
        "is_full_paper_reproduction": False,
    }
    (output / "resolved-config.json").write_text(
        json.dumps(resolved, indent=2, default=str) + "\n", encoding="utf-8"
    )
    device = setup_device("cuda:0")
    env = create_env(config.env, args.num_envs, device)
    runner = SuperMimicRunner(env=env, cfg=config.runner, task_id=task,
                             registry_name=None, device=device)
    started = time.monotonic()
    runner.run()
    wandb.finish()
    artifacts = []

    def assert_finite(value):
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            if not torch.isfinite(value).all().item():
                raise RuntimeError("NonfiniteCheckpoint")
        elif isinstance(value, dict):
            for child in value.values():
                assert_finite(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                assert_finite(child)

    for name in ("world_model.pt", "actor.pt", "critic.pt"):
        paths = list((output / "logs" / task).glob(f"*/{name}"))
        if len(paths) != 1:
            raise RuntimeError(f"ExpectedOneCheckpoint_{name}")
        path = paths[0]
        assert_finite(torch.load(path, map_location="cpu", weights_only=True))
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        artifacts.append({"path": str(path), "bytes": path.stat().st_size, "sha256": digest})
    result = {
        "status": "pilot_completed", "collection_steps": args.steps,
        "seed": args.seed, "num_envs": args.num_envs,
        "elapsed_training_seconds": time.monotonic() - started,
        "gpu": torch.cuda.get_device_name(0), "checkpoints": artifacts,
        "full_clip_evaluation_completed": False, "paper_results_reproduced": False,
        "goal_assembly_backend": args.goal_backend,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("OFFICIAL_LAFAN_PILOT_PASS " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
