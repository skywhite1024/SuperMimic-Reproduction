"""Offline full-clip evaluation of all 40 frozen LAFAN motions.

Uses the author's actor construction, observations and native termination rules.
This is an explicit local evaluation protocol, not a claim of paper equivalence.
"""

import argparse
import dataclasses
import hashlib
import json
import os
import time
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--motions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    os.environ["MUJOCO_GL"] = "disable"
    os.environ["WANDB_MODE"] = "offline"

    import numpy as np
    import torch
    import supermimic.tasks  # registers author tasks
    from mjlab.tasks.registry import load_env_cfg
    from supermimic.representations import extract_actor_goal
    from supermimic.runners.run_config import check_representation_matches, load_runner_cfg
    from supermimic.runners.supermimic_runner import build_actor, build_world_model
    from supermimic.scripts.train_utils import create_env, setup_device
    from supermimic.tasks.task_spec import build_actor_obs_td

    run, motions, output = args.run.resolve(), args.motions.resolve(), args.output.resolve()
    # Refuse an initial/partial checkpoint while the training process is active.
    training = json.loads((run / "result.json").read_text(encoding="utf-8"))
    if training["status"] != "pilot_completed":
        raise RuntimeError("A completed and validated pilot is required")
    resolved = json.loads((run / "resolved-config.json").read_text(encoding="utf-8"))
    goal_backend = resolved.get("goal_assembly_backend", "official")
    if goal_backend == "eager":
        import supermimic.representations as representations
        representations._compiled_assemble_goal = representations._assemble_goal
    elif goal_backend != "official":
        raise RuntimeError("Unknown goal assembly backend")
    task = resolved["task"]
    runner_cfg = load_runner_cfg({"runner_cfg": resolved["runner"]})
    check_representation_matches(task, runner_cfg.representation)
    names = json.loads((motions / "motion_names.json").read_text(encoding="utf-8"))
    ends = np.load(motions / "motion_ends.npy")
    starts = np.concatenate(([0], ends[:-1]))
    lengths = ends - starts
    if len(names) != 40 or len(ends) != 40 or not np.all(lengths > 0):
        raise RuntimeError("All 40 nonempty motions are required")
    checkpoints = {}
    for record in training["checkpoints"]:
        path = Path(record["path"])
        if digest(path) != record["sha256"]:
            raise RuntimeError("Checkpoint hash mismatch: " + path.name)
        checkpoints[path.name] = path
    output.mkdir(parents=True, exist_ok=False)

    cfg = load_env_cfg(task, play=True)
    cfg.seed = args.seed
    cfg.episode_length_s = 1e9
    # Keep the event schemas: SimContext derives its 33 input columns from
    # these entries. Collapse their sampling ranges to fixed physical values;
    # load_bundle below restores the training normalization buffers.
    if set(cfg.events) != {"foot_friction", "base_com", "encoder_bias"}:
        raise RuntimeError("Review newly introduced randomization events first")
    cfg.events["foot_friction"].params["ranges"] = (1.0, 1.0)
    cfg.events["base_com"].params["ranges"] = {axis: (0.0, 0.0) for axis in range(3)}
    cfg.events["encoder_bias"].params["bias_range"] = (0.0, 0.0)
    for group in cfg.observations.values():
        group.enable_corruption = False
    command_cfg = cfg.commands["motion"]
    command_cfg.motion_file = str(motions)
    command_cfg.sampling_mode = "start"
    command_cfg.pose_range = {}
    command_cfg.velocity_range = {}
    command_cfg.joint_position_range = (0.0, 0.0)
    command_cfg.resampling_time_range = (1e9, 1e9)
    robot = cfg.scene.entities["robot"]
    robot.collisions = tuple(dataclasses.replace(c, friction=(1.0,)) for c in robot.collisions)
    protocol = {
        "task": task, "seed": args.seed, "clips": 40,
        "start": "exact first reference frame via author reset_to_frame",
        "success": "first termination is native motion_end with no failure term",
        "termination_terms": {k: {"time_out": v.time_out, "params": v.params}
                              for k, v in cfg.terminations.items()},
        "domain_randomization": False, "observation_noise": False,
        "robot_collision_friction": 1.0, "physics": "unchanged author defaults",
        "repeatability_verified": False,
        "native_joint_limit_clipping_on_reset": True,
        "training": training, "paper_results_reproduced": False,
        "trajectory_state": "pre-action full_state; rewards and terminations are post-step",
        "motion_files": {p.name: digest(p) for p in sorted(motions.iterdir()) if p.is_file()},
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, default=str) + "\n")
    device = setup_device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation required")
    started = time.monotonic()
    env = create_env(cfg, 40, device)
    try:
        wm = build_world_model(runner_cfg, env, device)
        wm.load_bundle(torch.load(checkpoints["world_model.pt"], map_location=device, weights_only=True))
        wm.dynamics.eval()
        actor = build_actor(runner_cfg, wm, device)
        actor.load_state_dict(torch.load(checkpoints["actor.pt"], map_location=device, weights_only=True))
        actor.eval()
        actor.set_slot_mix({"motion": 1.0})
        env.reset()
        base = env.unwrapped
        command = base.command_manager.get_term("motion")
        if not np.array_equal(command.motion.motion_ends.cpu().numpy(), ends):
            raise RuntimeError("Loaded motion boundaries differ from the frozen pack")
        for index, start in enumerate(starts):
            command.reset_to_frame(torch.tensor([index], device=device), int(start))
        base.sim.forward()
        command.update_relative_body_poses()
        base.sim.sense()
        obs = env.get_observations()
        rep = wm.representation
        sim_ctx = wm.sim_context.read(base)
        active = np.ones(40, dtype=bool)
        records = [None] * 40
        trajectories = [{"full_state": [], "actions": [], "rewards": []} for _ in names]
        history = None
        with torch.inference_mode():
            for step in range(int(lengths.max()) + 1):
                state = rep.represent(obs["full_state"])
                sensor = rep.compute_actor_sensor(state, sim_ctx,
                    sim_actor_obs=obs["actor"] if rep.actor_obs_mode == "proprio" else None,
                    noisy=False)
                frame = sensor.unsqueeze(1)
                history = (frame.repeat(1, actor.history_length, 1) if history is None
                           else torch.cat((history[:, 1:], frame), dim=1))
                actions = actor(build_actor_obs_td({"motion": extract_actor_goal(obs["goal"])}, history))
                state_cpu = obs["full_state"].cpu().numpy().copy()
                action_cpu = actions.cpu().numpy().copy()
                if not np.isfinite(state_cpu[active]).all() or not np.isfinite(action_cpu[active]).all():
                    raise RuntimeError("Nonfinite active state or action; evaluation is incomplete")
                next_obs, rewards, dones, _ = env.step(actions)
                reward_cpu = rewards.cpu().numpy()
                done_cpu = dones.bool().cpu().numpy()
                tm = base.termination_manager
                terms = {name: tm.get_term(name).cpu().numpy().copy() for name in tm.active_terms}
                failed = tm.terminated.cpu().numpy().copy()
                for index in np.flatnonzero(active):
                    trace = trajectories[index]
                    trace["full_state"].append(state_cpu[index])
                    trace["actions"].append(action_cpu[index])
                    trace["rewards"].append(float(reward_cpu[index]))
                    if done_cpu[index]:
                        reasons = [name for name, values in terms.items() if values[index]]
                        success = bool(terms["motion_end"][index] and not failed[index])
                        duration = len(trace["rewards"])
                        records[index] = {
                            "index": int(index), "name": names[index], "success": success,
                            "reference_frames": int(lengths[index]), "steps": duration,
                            "seconds": duration * base.step_dt, "termination_terms": reasons,
                            "reward_sum": float(np.sum(trace["rewards"])),
                            "trajectory": f"trajectory-{index:02d}.npz",
                        }
                        if success and duration != int(lengths[index]):
                            raise RuntimeError("Unexpected motion-end duration; protocol needs correction")
                        np.savez_compressed(output / records[index]["trajectory"], **{
                            key: np.asarray(value, dtype=np.float32) for key, value in trace.items()})
                        trajectories[index] = None
                        active[index] = False
                        print("CLIP_RESULT " + json.dumps(records[index]), flush=True)
                obs = next_obs
                if not active.any():
                    break
                if step % 1000 == 0:
                    print(f"EVAL_PROGRESS step={step} remaining={active.sum()}", flush=True)
        if active.any():
            raise RuntimeError("Some clips did not reach a native termination")
        result = {
            "status": "full40_evaluation_completed", "clips": 40,
            "successes": sum(row["success"] for row in records),
            "failures": sum(not row["success"] for row in records),
            "elapsed_seconds": time.monotonic() - started, "records": records,
            "paper_results_reproduced": False,
            "goal_assembly_backend": goal_backend,
        }
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print("OFFICIAL_LAFAN_FULL40_EVAL_PASS " + json.dumps(result), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
