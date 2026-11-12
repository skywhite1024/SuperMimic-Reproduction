"""Finite 672-command, four-trial standing step test for the local velocity actor.

The exact standing joint pose and trial perturbations are absent from Appendix
G. This protocol uses the pinned robot's default knees-bent pose, zero velocity,
four explicit reset seeds and no physical randomization. Trials have nominally
identical initial conditions; numerical GPU repeatability is not established.
"""

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import time


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--motions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--purpose", choices=("experiment", "integration"), default="experiment")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    os.environ.update(MUJOCO_GL="disable", WANDB_MODE="offline")
    run, motions, output = args.run.resolve(), args.motions.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError("A fresh step-test output is required")

    import numpy as np
    import torch
    import supermimic.tasks
    import supermimic.representations as representations
    from mjlab.tasks.registry import load_env_cfg
    from supermimic.runners.run_config import load_runner_cfg
    from supermimic.runners.supermimic_runner import build_world_model
    from supermimic.scripts.train_utils import create_env, setup_device
    from supermimic.tasks.task_spec import TaskSpec, build_actor_obs_td, make_motion_task
    from paper_velocity_actor import PaperVelocityActor
    from paper_velocity_contract import aggregate_step_trials, heading_velocity, step_command_grid, velocity_fall
    from paper_velocity_runtime import root_gravity_z

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("The standing step test needs one allocated CUDA GPU")
    training = json.loads((run / "result.json").read_text(encoding="utf-8"))
    resolved = json.loads((run / "resolved-config.json").read_text(encoding="utf-8"))
    if training["status"] != "velocity_training_completed" or training["mode"] not in ("rl", "distill"):
        raise ValueError("A completed local velocity run is required")
    if args.purpose == "experiment" and (training["collection_steps"] != 100000
            or training["purpose"] != "experiment"):
        raise ValueError("The experiment purpose requires a completed100000-step actor")
    if resolved["velocity_mode"] != training["mode"] or resolved["task"] != "SuperMimic-GenCoord-G1":
        raise ValueError("Checkpoint task and velocity mode disagree")
    checkpoints = {}
    for record in training["checkpoints"]:
        path = Path(record["path"])
        if path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
            raise ValueError("Final checkpoint differs: " + path.name)
        checkpoints[path.name] = path
    if set(checkpoints) != {"world_model.pt", "actor.pt", "critic.pt"}:
        raise ValueError("The complete final three-model set is required")
    files = [{"file": name, "bytes": (motions / name).stat().st_size,
              "sha256": digest(motions / name)} for name in sorted(f["file"] for f in resolved["packed_files"])]
    if files != resolved["packed_files"]:
        raise ValueError("Evaluation motion bytes differ from the frozen training library")
    if resolved["goal_assembly_backend"] != "eager":
        raise ValueError("Review changed goal assembly before evaluating this adapter")
    representations._compiled_assemble_goal = representations._assemble_goal
    runner = load_runner_cfg({"runner_cfg": resolved["runner"]})
    if runner.actor_obs != "proprio" or runner.history_length != 10 or runner.representation != "gencoord":
        raise ValueError("Expected the verified93D/history10 generalized-coordinate actor")
    grid = step_command_grid()
    if len(grid) != 672 or len({tuple(r["command"]) for r in grid}) != 630:
        raise RuntimeError("Changed672 grid identities or630 unique physical commands")

    cfg = load_env_cfg(resolved["task"], play=True)
    cfg.seed, cfg.episode_length_s, cfg.auto_reset = args.seed, 1e9, False
    # Fall accounting below uses refreshed post-step qpos/qvel and never permits
    # a motion-end, timer or fall manager to reset a trial mid-response.
    cfg.terminations, cfg.rewards, cfg.metrics, cfg.curriculum = {}, {}, {}, {}
    if set(cfg.events) != {"foot_friction", "base_com", "encoder_bias"}:
        raise ValueError("Review changed event schemas before the standing test")
    cfg.events["foot_friction"].params["ranges"] = (1.0, 1.0)
    cfg.events["base_com"].params["ranges"] = {axis: (0.0, 0.0) for axis in range(3)}
    cfg.events["encoder_bias"].params["bias_range"] = (0.0, 0.0)
    for group in cfg.observations.values():
        group.enable_corruption = False
    command_cfg = cfg.commands["motion"]
    command_cfg.motion_file, command_cfg.sampling_mode = str(motions), "start"
    command_cfg.pose_range, command_cfg.velocity_range = {}, {}
    command_cfg.joint_position_range, command_cfg.resampling_time_range = (0.0, 0.0), (1e9, 1e9)
    robot_cfg = cfg.scene.entities["robot"]
    robot_cfg.collisions = tuple(dataclasses.replace(c, friction=(1.0,)) for c in robot_cfg.collisions)
    output.mkdir(parents=True, exist_ok=False)
    device = setup_device("cuda:0")
    started = time.monotonic()
    env = create_env(cfg, 672, device)
    try:
        wm = build_world_model(runner, env, device)
        wm.load_bundle(torch.load(checkpoints["world_model.pt"], map_location=device, weights_only=True))
        rep, base = wm.representation, env.unwrapped
        task_specs = [make_motion_task(rep, wm.num_goals), TaskSpec("velocity", 3, lambda *_: None)]
        actor = PaperVelocityActor(rep.proprio_obs_dim(wm.num_actions), wm.num_actions,
            task_specs, runner.slot_mix, history_length=runner.history_length,
            latent_dim=runner.latent_dim, motion_hidden_dims=runner.actor_hidden_dims,
            latent_bc_weight=runner.latent_bc_weight).to(device)
        actor.load_state_dict(torch.load(checkpoints["actor.pt"], map_location=device, weights_only=True))
        actor.set_slot_mix({"motion": 0.0, "velocity": 1.0})
        actor.eval()
        wm.dynamics.eval()
        if not abs(base.step_dt - .02) < 1e-9 or rep.full_state_dim != 71 or wm.num_actions != 29:
            raise ValueError("The fixed50Hz,71D-state,29D-action step contract changed")
        command = base.command_manager.get_term("motion")
        robot = base.scene["robot"]
        all_ids = torch.arange(672, device=device)
        commands = torch.tensor([row["command"] for row in grid], dtype=torch.float32, device=device)
        states = np.full((4, 672, 251, 71), np.nan, dtype=np.float32)
        actions_saved = np.full((4, 672, 250, 29), np.nan, dtype=np.float32)
        velocities = np.full((4, 672, 250, 3), np.nan, dtype=np.float32)
        steps = np.zeros((4, 672), dtype=np.int64)
        falls = np.zeros((4, 672), dtype=np.bool_)
        protocol = {"purpose": args.purpose, "mode": training["mode"], "commands": grid,
            "grid_identities": 672, "unique_physical_commands": 630, "trials": 4, "fps": 50,
            "seconds": 5, "steady_state_seconds": 2, "trial_seeds": [args.seed + i for i in range(4)],
            "initial_pose": "Pinned robot default knees-bent joint/root pose, zero velocity",
            "standing_root_state": robot.data.default_root_state[0].cpu().tolist(),
            "standing_joint_positions": robot.data.default_joint_pos[0].cpu().tolist(),
            "joint_names": list(robot.joint_names),
            "initial_pose_author_confirmed": False, "trial_perturbations": "None; nominal initial states identical",
            "domain_randomization": False, "observation_noise": False, "friction": 1.0,
            "fall": "Refreshed post-step root height<0.35 OR1+body-gravity-z>0.7",
            "motion_manager": "Reference index held away from total-pack wrap; no state resets during trials",
            "trace": "Initial state then post-step states; pre-step actions and post-step heading velocities; truncate at first fall; unused padding is NaN and excluded with steps",
            "repeatability_verified": False, "training": training, "packed_files": files,
            "paper_results_reproduced": False}
        write_json(output / "protocol.json", protocol)
        with torch.inference_mode():
            for trial in range(4):
                env.seed(args.seed + trial)
                env.reset()
                root = robot.data.default_root_state.clone()
                root[:, :3] += base.scene.env_origins
                root[:, 7:] = 0
                robot.write_root_state_to_sim(root)
                robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(),
                                                torch.zeros_like(robot.data.default_joint_pos))
                command.set_time_steps(all_ids, torch.zeros(672, device=device, dtype=torch.long))
                base.scene.write_data_to_sim()
                base.sim.forward()
                command.update_relative_body_poses()
                base.sim.sense()
                obs = env.get_observations()
                states[trial, :, 0] = obs["full_state"].cpu().numpy()
                expected_standing = torch.cat((robot.data.default_root_state[:, :7],
                    robot.data.default_joint_pos, torch.zeros(672, wm.num_actions + 6, device=device)), -1)
                if not torch.allclose(obs["full_state"], expected_standing, atol=1e-5, rtol=0):
                    raise RuntimeError("Observed initial state differs from the declared standing pose")
                if not torch.isfinite(obs["full_state"]).all() or velocity_fall(
                        obs["full_state"][:, 2], root_gravity_z(obs["full_state"])).any():
                    raise RuntimeError("The explicit standing initialization is nonfinite or already fallen")
                history, fallen = None, torch.zeros(672, device=device, dtype=torch.bool)
                for step in range(250):
                    active = ~fallen
                    sensor = rep.compute_actor_sensor(rep.represent(obs["full_state"]),
                        wm.sim_context.read(base), sim_actor_obs=obs["actor"], noisy=False)
                    frame = sensor[:, None]
                    history = (frame.repeat(1, 10, 1) if history is None
                               else torch.cat((history[:, 1:], frame), dim=1))
                    actor_obs = build_actor_obs_td({"motion": torch.zeros(672, task_specs[0].goal_dim,
                        device=device), "velocity": commands}, history)
                    actions = actor(actor_obs)
                    if not torch.isfinite(actions[active]).all() or not torch.isfinite(sensor[active]).all():
                        raise RuntimeError("Nonfinite active step-test action or observation")
                    actions = torch.where(active[:, None], actions, torch.zeros_like(actions))
                    next_obs, _, dones, _ = env.step(actions)
                    full = next_obs["full_state"]
                    if dones.bool().any() or not torch.isfinite(full[active]).all():
                        raise RuntimeError("Unexpected simulator reset or nonfinite active post-step state")
                    measured = heading_velocity(full[:, 3:7], full[:, 36:39], full[:, 39:42])
                    keep = active.cpu().numpy()
                    states[trial, keep, step + 1] = full[active].cpu().numpy()
                    actions_saved[trial, keep, step] = actions[active].cpu().numpy()
                    velocities[trial, keep, step] = measured[active].cpu().numpy()
                    steps[trial, keep] += 1
                    fallen |= velocity_fall(full[:, 2], root_gravity_z(full))
                    obs = next_obs
                    if fallen.all():
                        break
                falls[trial] = fallen.cpu().numpy()
                if not np.all(falls[trial] | (steps[trial] == 250)):
                    raise RuntimeError("A surviving trial lacks the full five-second response")
                print(f"VELOCITY_STEP_TRIAL_COMPLETED trial={trial} fallen={falls[trial].sum()} steps={steps[trial].sum()}", flush=True)
        records = [dict(row, index=index, **aggregate_step_trials(row["command"],
            velocities[:, index], falls[:, index])) for index, row in enumerate(grid)]
        trace = output / "trajectories.npz"
        np.savez_compressed(trace, states=states, actions=actions_saved,
                            heading_velocity=velocities, steps=steps, fallen=falls)
        del states, actions_saved, velocities
        with np.load(trace, allow_pickle=False) as data:
            recomputed_heading = None
            for key, count in (("states", 251), ("actions", 250), ("heading_velocity", 250)):
                values = data[key]
                valid = np.arange(count)[None, None, :] < (steps[..., None] + (key == "states"))
                if not np.isfinite(values[valid]).all() or not np.isnan(values[~valid]).all():
                    raise RuntimeError("Saved trace finite values or unused padding disagree with first-fall lengths")
                if key == "states":
                    post = values[:, :, 1:]
                    gravity_z = 2 * np.square(post[..., 4:6]).sum(-1, dtype=np.float32) - 1
                    fell = ((post[..., 2] < .35) | (1 + gravity_z > .7)) & valid[..., 1:]
                    if (not np.array_equal(fell.any(-1), falls)
                            or not np.array_equal(np.where(falls, fell.argmax(-1) + 1, 250), steps)):
                        raise RuntimeError("Saved states do not reproduce first-fall flags and trace lengths")
                    w, x, y, z = np.moveaxis(post[..., 3:7].astype(np.float64), -1, 0)
                    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
                    c, sine = np.cos(yaw), np.sin(yaw)
                    vx, vy = post[..., 36].astype(np.float64), post[..., 37].astype(np.float64)
                    recomputed_heading = np.stack((c * vx + sine * vy,
                        -sine * vx + c * vy, post[..., 41]), -1)
                elif key == "heading_velocity":
                    if not np.allclose(values[valid], recomputed_heading[valid], rtol=1e-5, atol=1e-5):
                        raise RuntimeError("Saved heading velocities disagree with independent float64 state projection")
                    for index, row in enumerate(grid):
                        recalculated = aggregate_step_trials(row["command"], values[:, index], falls[:, index])
                        if any(records[index][name] != value for name, value in recalculated.items()):
                            raise RuntimeError("Saved velocity traces do not reproduce the command score")
            if not np.array_equal(data["steps"], steps) or not np.array_equal(data["fallen"], falls):
                raise RuntimeError("Saved trial accounting differs")
        raw = [r["median_relative_error"] for r in records if r["median_relative_error"] is not None]
        result = {"status": "velocity_step_test_completed", "purpose": args.purpose,
            "mode": training["mode"], "commands": 672, "trials_per_command": 4,
            "total_trials": 2688, "fallen_trials": int(falls.sum()),
            "all_declared_initial_poses_verified": True,
            "all_fall_commands": sum(r["all_trials_fell"] for r in records),
            "median_defined_command_error": float(np.median(raw)) if raw else None,
            "median_display_command_error": float(np.median([r["display_relative_error"] for r in records])),
            "records": records, "elapsed_seconds": time.monotonic() - started,
            "trajectory": {"file": trace.name, "bytes": trace.stat().st_size, "sha256": digest(trace),
                           "saved_arrays_readback_verified": True, "saved_falls_recomputed": True,
                           "saved_heading_velocity_recomputed_float64": True,
                           "heading_projection_tolerance": {"rtol": 1e-5, "atol": 1e-5},
                           "command_scores_recomputed_from_saved_traces": True,
                           "padding": "NaN outside first-fall length"},
            "protocol_sha256": digest(output / "protocol.json"), "paper_results_reproduced": False}
        write_json(output / "result.json", result)
        print("VELOCITY_STEP_TEST_COMPLETED " + json.dumps({k: v for k, v in result.items() if k != "records"}), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
