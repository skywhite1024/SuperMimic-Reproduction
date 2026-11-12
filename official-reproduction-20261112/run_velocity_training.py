"""Finite local velocity experiment using the fixed official simulator and WM.

Requires a verified walking/running library for 100000-step experiments. The
short integration purpose accepts an explicitly counted control library. This
entry has model snapshots, not full optimizer/replay/simulator-state resume.
"""

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import time


PACKED_FILES = {"joint_names.json", "joint_pos.npy", "joint_vel.npy", "motion_ends.npy",
                "motion_names.json", "root_ang_vel_w.npy", "root_lin_vel_w.npy",
                "root_pos_w.npy", "root_quat_w.npy"}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def final_model_snapshot(runner, paths, output, steps):
    """Save live final weights when the native zero-based interval misses the end."""
    import torch
    from checkpoint_series import preserve_checkpoint
    states = {"world_model.pt": runner._wm.bundle(),
              "actor.pt": runner.policy_learner.actor.state_dict(),
              "critic.pt": runner.policy_learner.critic.state_dict()}
    if {path.name for path in paths} != set(states) or len(paths) != 3:
        raise ValueError("A complete three-model destination set is required")
    for path in paths:
        torch.save(states[path.name], path)
    result = preserve_checkpoint(paths, output, steps)
    result["source"] = "post_loop_final_save"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("rl", "distill"), required=True)
    parser.add_argument("--purpose", choices=("experiment", "integration"), default="experiment")
    parser.add_argument("--motions", type=Path, required=True)
    parser.add_argument("--library-manifest", type=Path)
    parser.add_argument("--expected-clips", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--checkpoint-interval", type=int, default=4000)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.num_envs != 4096 or args.steps < 21 or args.checkpoint_interval < 1:
        parser.error("Use4096 environments, at least21 finite steps and a positive save interval")
    if args.purpose == "integration" and args.steps > 101:
        parser.error("Integration is bounded to101 collection steps")
    if args.purpose == "experiment" and (args.steps != 100000 or args.library_manifest is None):
        parser.error("The100000-step experiment requires an explicit verified walking/running library")
    motions, output = args.motions.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError("A fresh output folder is required")
    os.environ.update(WANDB_MODE="offline", MUJOCO_GL="disable")

    import numpy as np
    import torch
    import wandb
    import supermimic.runners.supermimic_runner as native
    from supermimic.scripts.train import TrainConfig
    from supermimic.scripts.train_utils import create_env, setup_device
    from checkpoint_series import preserve_checkpoint
    from paper_profile import configure_runner, profile_description
    from paper_velocity_runner import make_velocity_runner
    from paper_velocity_runtime import configure_mixed_environment

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Velocity training requires exactly one allocated CUDA GPU")
    ends = np.load(motions / "motion_ends.npy")
    names = json.loads((motions / "motion_names.json").read_text(encoding="utf-8"))
    if (args.expected_clips < 1 or len(names) != args.expected_clips or len(set(names)) != len(names)
            or ends.shape != (len(names),) or not np.issubdtype(ends.dtype, np.integer)
            or not np.all(np.diff(ends, prepend=0) > 0)):
        raise ValueError("Complete nonempty unique library membership is required")
    files = [{"file": name, "bytes": (motions / name).stat().st_size,
              "sha256": digest(motions / name)} for name in sorted(PACKED_FILES)]
    library = None
    if args.library_manifest is not None:
        library = json.loads(args.library_manifest.read_text(encoding="utf-8"))
        if (library["status"] != "verified_walking_running_library" or library["clips"] != len(names)
                or library["frames"] != int(ends[-1]) or library["fps"] != 50
                or library["packed_files"] != files):
            raise ValueError("Walking/running manifest does not identify these exact packed bytes")
    output.mkdir(parents=True, exist_ok=False)
    os.chdir(output)
    import supermimic.representations as representations
    representations._compiled_assemble_goal = representations._assemble_goal

    snapshots = []
    def offline_checkpoint(paths):
        steps = len(snapshots) * args.checkpoint_interval + 1
        receipt = preserve_checkpoint(paths, output / "checkpoint-series", steps)
        receipt["source"] = "native_callback"
        if wandb.run is None:
            raise RuntimeError("Native offline artifact run is missing")
        artifact = wandb.Artifact(f"model-{wandb.run.id}", type="model")
        for path in paths:
            artifact.add_file(path)
        wandb.run.log_artifact(artifact)
        receipt["offline_artifact_logging_completed"] = True
        snapshots.append(receipt)
        print("VELOCITY_MODEL_SNAPSHOT " + json.dumps(receipt), flush=True)
    native.log_model_artifact = offline_checkpoint

    task = "SuperMimic-GenCoord-G1"
    config = TrainConfig.from_task(task)
    configure_runner(config.runner)
    config.runner.slot_mix = {"motion": .5, "velocity": .5}
    config.runner.max_iterations, config.runner.save_interval = args.steps, args.checkpoint_interval
    config.runner.run_name = f"local-paper-velocity-{args.mode}-{args.purpose}-seed{args.seed}"
    config.runner.log_interval = 10
    config.env.seed = args.seed
    config.env.scene.num_envs = args.num_envs
    config.env.commands["motion"].motion_file = str(motions)
    config.env.commands["motion"].sampling_mode = "external"
    configure_mixed_environment(config.env, args.mode)
    profile = profile_description()
    profile["name"] = "paper-a5-velocity-local"
    profile["limitations"] = [item for item in profile["limitations"]
                              if not item.startswith("Motion training only")]
    profile["velocity_mode"] = args.mode
    profile["history_feature"] = "flattened_normalized_proprioception; author feature extractor unspecified"
    profile["reference_window"] = "Project each reference frame, then centered25-frame mean; clip-edge repetition unconfirmed"
    profile["reset_weights"] = "Native return sampler scores motion simulator rows only; weights shared for reference resets"
    profile["velocity_fall_smoothing"] = "Native softness on normalized height/tilt margins; paper specifies only hard thresholds"
    resolved = {"task": task, "seed": args.seed, "num_envs": args.num_envs,
        "collection_steps": args.steps, "purpose": args.purpose, "velocity_mode": args.mode,
        "motion_directory": str(motions), "clips": len(names), "frames": int(ends[-1]),
        "packed_files": files, "library_manifest": library,
        "runner": dataclasses.asdict(config.runner), "environment": dataclasses.asdict(config.env),
        "paper_profile": profile, "goal_assembly_backend": "eager", "vendor_files_modified": False,
        "full_training_resume_available": False, "paper_results_reproduced": False,
        "step_test_completed": False, "onnx_export_available": False}
    (output / "resolved-config.json").write_text(json.dumps(resolved, indent=2, default=str) + "\n",
                                                 encoding="utf-8", newline="\n")
    print("VELOCITY_TRAINING_LIBRARY " + json.dumps({"mode": args.mode, "purpose": args.purpose,
        "clips": len(names), "frames": int(ends[-1]), "source_filter_author_confirmed": False}), flush=True)
    device = setup_device("cuda:0")
    env = create_env(config.env, args.num_envs, device)
    runner_class = make_velocity_runner(args.mode)
    runner = runner_class(env=env, cfg=config.runner, task_id=task, registry_name=None, device=device)
    update_counts = {"world_model": 0, "policy": 0}
    def observe_updates(learner, name):
        original = learner.update
        def update(*inputs, **kwargs):
            values = original(*inputs, **kwargs)
            update_counts[name] += 1
            if args.purpose == "integration":
                for key, value in values.items():
                    if isinstance(value, torch.Tensor) and not torch.isfinite(value).all().item():
                        raise RuntimeError("Nonfinite integration update metric: " + key)
            return values
        learner.update = update
    observe_updates(runner.world_model_learner, "world_model")
    observe_updates(runner.policy_learner, "policy")
    started = time.monotonic()
    runner.run()
    expected = list(range(1, args.steps + 1, args.checkpoint_interval))
    if [receipt["collection_steps"] for receipt in snapshots] != expected:
        raise RuntimeError("Incomplete native velocity snapshot sequence")
    paths = []
    for name in ("world_model.pt", "actor.pt", "critic.pt"):
        found = list((output / "logs" / task).glob(f"*/{name}"))
        if len(found) != 1:
            raise RuntimeError("Expected one native model path: " + name)
        paths.append(found[0])
    if expected[-1] != args.steps:
        receipt = final_model_snapshot(runner, paths, output / "checkpoint-series", args.steps)
        snapshots.append(receipt)
    final = snapshots[-1]
    if final["collection_steps"] != args.steps:
        raise RuntimeError("Final snapshot does not identify the complete training count")
    final_files = {entry["file"]: entry for entry in final["files"]}
    checkpoints = [{"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)} for path in paths]
    if any(item[key] != final_files[Path(item["path"]).name][key]
           for item in checkpoints for key in ("bytes", "sha256")):
        raise RuntimeError("Native final files differ from the finite-validated final snapshot")
    captured = {"world_model": runner.world_model_learner._captured_update._graph is not None,
                "policy": runner.policy_learner._captured_update._graph is not None}
    if not all(captured.values()) or min(update_counts.values()) < 4:
        raise RuntimeError("Both learners must execute updates including native CUDA graph capture")
    replay_evidence = None
    if args.purpose == "integration":
        replay = runner.replay_buffer
        stored = replay._storage[:, :replay.num_transitions]
        for key in ("state", "action", "prev_action", "sim_ctx", "actor_obs",
                    "velocity_command", "velocity_held"):
            if not torch.isfinite(stored[key]).all().item():
                raise RuntimeError("Nonfinite integration replay field: " + key)
        half = args.num_envs // 2
        expected = (torch.arange(args.num_envs, device=device) >= half)[:, None]
        if not torch.equal(stored["velocity_interface"], expected.expand_as(stored["velocity_interface"])):
            raise RuntimeError("Logged simulator interface identities disagree with the configured split")
        starts = runner.wm_env.sample_initial_states()
        if not torch.equal(starts["velocity_interface"], runner.wm_env.velocity_rows):
            raise RuntimeError("Imagined start states crossed simulator interfaces")
        if not ((stored["velocity_age"] >= 0) & (stored["velocity_age"] < 250)).all().item():
            raise RuntimeError("Invalid logged command age")
        reference_fraction = float(stored["velocity_reference"][half:].float().mean())
        if args.mode == "distill" and reference_fraction != 1.0:
            raise RuntimeError("Distillation used non-reference commands")
        if args.mode == "rl" and not 0 < reference_fraction < 1:
            raise RuntimeError("RL integration did not exercise both command sources")
        replay_evidence = {"finite_fields": 7, "stored_control_steps": replay.num_transitions,
            "simulator_interface_identity_verified": True, "own_interface_imagined_starts_verified": True,
            "command_ages_valid": True, "velocity_reference_fraction": reference_fraction}
    result = {"status": "velocity_training_completed", "purpose": args.purpose,
        "mode": args.mode, "collection_steps": args.steps, "num_envs": args.num_envs,
        "seed": args.seed, "clips": len(names), "frames": int(ends[-1]),
        "elapsed_training_seconds": time.monotonic() - started,
        "checkpoints": checkpoints,
        "retained_checkpoint_series": snapshots, "independent_critics": 2,
        "learner_update_counts": update_counts, "native_cuda_graphs_captured": captured,
        "all_update_metrics_checked_finite": args.purpose == "integration",
        "integration_replay_evidence": replay_evidence,
        "full_training_resume_available": False, "step_test_completed": False,
        "paper_results_reproduced": False}
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("VELOCITY_TRAINING_COMPLETED " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
