"""Finite CUDA training on raw LAFAN40 or the verified unfiltered BONES pack.

This verifies execution and checkpoint creation, not the paper's full benchmark.
"""

import argparse
import dataclasses
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path


def main():
    entry_started = time.monotonic()
    entry_utc = datetime.now(timezone.utc).isoformat()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1001)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-interval", type=int,
                        help="Native zero-based save interval; must divide steps-1. Default saves initial/final only")
    parser.add_argument("--keep-checkpoint-series", action="store_true",
                        help="Retain finite, hash-identified model snapshots; does not enable training resume")
    parser.add_argument("--record-checkpoint-times", action="store_true",
                        help="Record asynchronous checkpoint availability times separately from model manifests")
    parser.add_argument("--initial-native-snapshot", type=Path,
                        help="Explicit model-weight warm start; optimizer/replay/RNG are reset")
    parser.add_argument("--learning-rate-offset", type=int, default=0,
                        help="Inherited native collection count for the A5 cosine schedule")
    parser.add_argument("--policy-horizon", type=int, choices=(8, 16, 32, 64, 128),
                        help="Explicit local critic-horizon ablation after applying the A5 profile")
    parser.add_argument("--without-critic", action="store_true",
                        help="Use the author's native BPTT learner; save exactly world_model and actor")
    parser.add_argument("--corpus", choices=("lafan40", "bones-raw"), default="lafan40",
                        help="bones-raw is a capacity/training probe, not the filtered paper benchmark")
    parser.add_argument("--training-profile", choices=("public-default", "paper-a5-single-gpu"),
                        default="public-default")
    parser.add_argument("--goal-backend", choices=("official", "eager"), default="official",
                        help="Diagnostic: eager runs the author's same goal math without torch.compile")
    args = parser.parse_args()
    if (args.initial_native_snapshot is None) != (args.learning_rate_offset == 0):
        parser.error("Warm start requires both a native snapshot and a positive learning-rate offset")
    if args.initial_native_snapshot is not None and (args.learning_rate_offset < 1 or args.training_profile != "paper-a5-single-gpu"):
        parser.error("Warm start requires the explicit A5 profile and positive inherited count")
    if (args.policy_horizon is not None or args.without_critic) and args.training_profile != "paper-a5-single-gpu":
        parser.error("Critic/horizon ablation requires the explicit A5 profile")
    if args.steps < 2:
        parser.error("At least two simulator collection steps are required")
    if args.record_checkpoint_times and not args.keep_checkpoint_series:
        parser.error("record-checkpoint-times requires keep-checkpoint-series")
    save_interval = args.steps - 1 if args.checkpoint_interval is None else args.checkpoint_interval
    if save_interval < 1 or (args.steps - 1) % save_interval:
        parser.error("checkpoint-interval must be positive and divide steps-1 so the final step is saved")
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
    checkpoint_receipts = []
    checkpoint_times = []

    def clock_observation():
        now = time.monotonic()
        return {"utc": datetime.now(timezone.utc).isoformat(),
                "seconds_since_program_entry": now - entry_started,
                "seconds_since_runner_run_entry": now - started}

    def log_offline_checkpoint(files):
        callback_entry = clock_observation() if args.record_checkpoint_times else None
        assert wandb.run is not None
        if args.keep_checkpoint_series:
            from checkpoint_series import preserve_checkpoint
            # The native runner serializes uploads and invokes this exactly once
            # at each zero-based save interval, after that iteration's updates.
            completed_steps = len(checkpoint_receipts) * save_interval + 1
            receipt = preserve_checkpoint(files, output / "checkpoint-series", completed_steps,
                                          without_critic=args.without_critic)
            checkpoint_receipts.append(receipt)
            preserved = clock_observation() if args.record_checkpoint_times else None
            print("MODEL_SNAPSHOT_PRESERVED " + json.dumps(receipt), flush=True)
        artifact = wandb.Artifact(f"model-{wandb.run.id}", type="model")
        for path in files:
            artifact.add_file(path)
        wandb.run.log_artifact(artifact)
        if args.record_checkpoint_times:
            checkpoint_times.append({"collection_steps": completed_steps,
                "callback_entry_after_native_save": callback_entry,
                "retained_snapshot_complete": preserved,
                "offline_artifact_logging_returned": clock_observation()})
            timing = {"schema_version": 1, "program_entry_utc": entry_utc,
                "runner_run_entry_utc": run_entry_utc,
                "time_definition": "Monotonic elapsed wall time. Program entry includes imports, config and environment/model initialization; runner.run entry excludes that setup. Both include native warmup, graph capture and snapshot overhead. Evaluation runs separately after training.",
                "asynchronous_callback_caveat": "Callback entry is after native serialization and may be delayed by thread scheduling. Retained-snapshot completion is an observed availability upper bound, not an exact collection/update completion timestamp.",
                "checkpoints": checkpoint_times}
            (output / "checkpoint-times.json").write_text(
                json.dumps(timing, indent=2) + "\n", encoding="utf-8", newline="\n")
            print("CHECKPOINT_TIME_OBSERVED " + json.dumps(checkpoint_times[-1]), flush=True)

    runner_module.log_model_artifact = log_offline_checkpoint

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Pilot requires exactly one allocated CUDA GPU")
    ends = np.load(motions / "motion_ends.npy")
    expected_clips = {"lafan40": 40, "bones-raw": 142220}[args.corpus]
    if ends.ndim != 1 or len(ends) != expected_clips or not np.all(np.diff(ends, prepend=0) > 0):
        raise RuntimeError(f"Expected {expected_clips} nonempty clips for {args.corpus}")
    corpus = {"name": args.corpus, "clips": len(ends), "frames": int(ends[-1]),
              "paper_name_filter_applied": False, "paper_feasibility_filter_applied": False}
    print("TRAINING_CORPUS " + json.dumps(corpus), flush=True)
    task = "SuperMimic-GenCoord-G1"
    config = TrainConfig.from_task(task)
    profile = None
    ablation = None
    runner_class = SuperMimicRunner
    if args.training_profile == "paper-a5-single-gpu":
        if args.num_envs != 4096:
            raise ValueError("The single-GPU table profile requires 4096 environments")
        from paper_profile import configure_runner, install_paper_runner, profile_description
        configure_runner(config.runner)
        runner_class = install_paper_runner(SuperMimicRunner)
        profile = profile_description()
        if args.policy_horizon is not None or args.without_critic:
            from paper_profile import configure_critic_ablation
            ablation = configure_critic_ablation(config.runner, args.policy_horizon or 32,
                                                 args.without_critic)
    config.env.seed = args.seed
    config.env.commands["motion"].motion_file = str(motions)
    config.env.commands["motion"].sampling_mode = "external"
    config.runner.max_iterations = args.steps
    config.runner.save_interval = save_interval
    config.runner.run_name = f"official-{args.corpus}-pilot-seed{args.seed}"
    config.runner.log_interval = 10
    initial_payloads, initialization = None, None
    if args.initial_native_snapshot is not None:
        from native_warm_start import read_snapshot
        initial_payloads, initialization = read_snapshot(args.initial_native_snapshot,
            args.learning_rate_offset, dataclasses.asdict(config.runner), args.corpus)
    resolved = {
        "task": task, "seed": args.seed, "num_envs": args.num_envs,
        "collection_steps": args.steps, "motion_directory": str(motions),
        "corpus": corpus,
        "runner": dataclasses.asdict(config.runner),
        "environment": dataclasses.asdict(config.env),
        "source_algorithm_modified": profile is not None,
        "vendor_files_modified": False,
        "training_profile": args.training_profile,
        "paper_profile": profile,
        "policy_ablation": ablation,
        "goal_assembly_backend": args.goal_backend,
        "runtime_execution_modified": args.goal_backend != "official",
        "io_adapter": "offline checkpoint artifact logging; online pruning disabled",
        "keep_checkpoint_series": args.keep_checkpoint_series,
        "record_checkpoint_times": args.record_checkpoint_times,
        "full_training_resume_available": False,
        "is_full_paper_reproduction": False,
        "initialization": initialization,
    }
    (output / "resolved-config.json").write_text(
        json.dumps(resolved, indent=2, default=str) + "\n", encoding="utf-8"
    )
    device = setup_device("cuda:0")
    torch.cuda.reset_peak_memory_stats()
    env = create_env(config.env, args.num_envs, device)
    runner = runner_class(env=env, cfg=config.runner, task_id=task,
                             registry_name=None, device=device)
    if initial_payloads is not None:
        from native_warm_start import restore_snapshot, learned_parameter_hashes
        initialization = {**initialization, **restore_snapshot(runner, initial_payloads)}
        runner._paper_step = args.learning_rate_offset
        initialization["initial_learned_parameter_hashes"] = learned_parameter_hashes(runner)
        (output / "initialization-readback.json").write_text(
            json.dumps(initialization, indent=2) + "\n", encoding="utf-8")
        del initial_payloads
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    initial_memory = {"free_bytes": free_bytes, "total_bytes": total_bytes,
                      "torch_allocated_bytes": torch.cuda.memory_allocated(),
                      "torch_reserved_bytes": torch.cuda.memory_reserved()}
    print("CUDA_MEMORY_AFTER_INITIALIZATION " + json.dumps(initial_memory), flush=True)
    started = time.monotonic()
    run_entry_utc = datetime.now(timezone.utc).isoformat()
    runner.run()
    if initialization is not None:
        initialization["final_learned_parameter_hashes"] = learned_parameter_hashes(runner)
        if any(initialization["initial_learned_parameter_hashes"][key] == value
               for key, value in initialization["final_learned_parameter_hashes"].items()):
            raise RuntimeError("WarmStartDidNotUpdateEveryNativeLearnedNetwork")
        initialization["all_three_learned_networks_changed"] = True
    runner_returned = clock_observation() if args.record_checkpoint_times else None
    wandb.finish()
    if args.keep_checkpoint_series:
        expected = list(range(1, args.steps + 1, save_interval))
        if [item["collection_steps"] for item in checkpoint_receipts] != expected:
            raise RuntimeError("IncompleteIntermediateCheckpointSeries")
        if args.record_checkpoint_times and [x["collection_steps"] for x in checkpoint_times] != expected:
            raise RuntimeError("IncompleteCheckpointTimingLedger")
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

    model_names = ("world_model.pt", "actor.pt") if args.without_critic else ("world_model.pt", "actor.pt", "critic.pt")
    if args.without_critic and list((output / "logs" / task).glob("*/critic.pt")):
        raise RuntimeError("Unexpected critic checkpoint in no-critic experiment")
    for name in model_names:
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
        "training_profile": args.training_profile,
        "critic_enabled": not args.without_critic,
        "policy_ablation": ablation,
        "corpus": corpus,
        "device_memory_after_initialization": initial_memory,
        "peak_torch_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_torch_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
        "memory_scope": "PyTorch allocator only; excludes Warp and other CUDA allocations",
        "retained_checkpoint_series": checkpoint_receipts,
        "checkpoint_timing_file": "checkpoint-times.json" if args.record_checkpoint_times else None,
        "runner_returned_observation": runner_returned,
        "full_training_resume_available": False,
        "initialization": initialization,
        "inherited_collection_steps": args.learning_rate_offset,
        "combined_collection_steps_including_warm_start": args.learning_rate_offset + args.steps,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    marker = "OFFICIAL_LAFAN_PILOT_PASS" if args.corpus == "lafan40" else "OFFICIAL_BONES_RAW_PILOT_PASS"
    print(marker + " " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
