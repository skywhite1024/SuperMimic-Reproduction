"""Corrected manual-reset-v2 full-clip evaluation of a frozen packed corpus.

Uses the author's actor construction, observations and native termination rules.
This is an explicit local evaluation protocol, not a claim of paper equivalence.
"""

import argparse
import dataclasses
import hashlib
import json
import os
import time
from contextlib import ExitStack
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
    parser.add_argument("--expected-clips", type=int, required=True)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--order", choices=("index", "length"), default="length")
    args = parser.parse_args()
    if args.num_envs < 1 or args.expected_clips < 1:
        parser.error("positive environment and clip counts are required")
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
    from manual_reset_observations import refresh_manual_reset_observations

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
    count = len(names)
    if (count != args.expected_clips or len(ends) != count or len(set(names)) != count
            or not np.issubdtype(ends.dtype, np.integer) or not np.all(lengths > 0)):
        raise RuntimeError("Expected complete corpus with unique names and positive integer boundaries")
    num_envs = min(args.num_envs, count)
    order = (np.arange(count) if args.order == "index"
             else np.argsort(-lengths, kind="stable"))
    assert set(order.tolist()) == set(range(count))
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
        "task": task, "seed": args.seed, "clips": count,
        "num_envs": num_envs, "batch_order": args.order,
        "all_motions_included": True, "subset_filter_applied": False,
        "batch_reset": "fresh env reset; exact first frame; history filled with first sensor frame",
        "inference_scope": "all resets, observations and steps across all batches after model construction",
        "incomplete_result_policy": "completed batch receipts retained; never label an interrupted evaluation complete",
        "start": "exact first reference frame via author reset_to_frame",
        "success": "first termination is native motion_end with no failure term",
        "termination_terms": {k: {"time_out": v.time_out, "params": v.params}
                              for k, v in cfg.terminations.items()},
        "domain_randomization": False, "observation_noise": False,
        "robot_collision_friction": 1.0, "physics": "unchanged author defaults",
        "repeatability_verified": False,
        "native_joint_limit_clipping_on_reset": True,
        "manual_reset_version": 2,
        "observation_cache_refreshed_after_pose_write": True,
        "actuator_histories_backfilled_after_pose_write": True,
        "training": training, "paper_results_reproduced": False,
        "trajectory_state": "pre-action full_state; rewards and terminations are post-step",
        "motion_files": {p.name: digest(p) for p in sorted(motions.iterdir()) if p.is_file()},
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, default=str) + "\n")
    device = setup_device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation required")
    started = time.monotonic()
    env = create_env(cfg, num_envs, device)
    with ExitStack() as evaluation:
        evaluation.callback(env.close)
        wm = build_world_model(runner_cfg, env, device)
        wm.load_bundle(torch.load(checkpoints["world_model.pt"], map_location=device, weights_only=True))
        wm.dynamics.eval()
        actor = build_actor(runner_cfg, wm, device)
        actor.load_state_dict(torch.load(checkpoints["actor.pt"], map_location=device, weights_only=True))
        actor.eval()
        actor.set_slot_mix({"motion": 1.0})
        # ContactSensor.update replaces history buffers with torch.roll results.
        # Keep their later in-place reset inside the same inference context.
        # Construct/load model parameters first, outside inference mode.
        evaluation.enter_context(torch.inference_mode())
        env.reset()
        base = env.unwrapped
        command = base.command_manager.get_term("motion")
        if not np.array_equal(command.motion.motion_ends.cpu().numpy(), ends):
            raise RuntimeError("Loaded motion boundaries differ from the frozen pack")
        rep = wm.representation
        records = [None] * count
        for batch_id, offset in enumerate(range(0, count, num_envs)):
            indices = order[offset:offset + num_envs]
            # All environment managers and previous actions reset between batches.
            # The final short batch has inactive padding environments; they never
            # contribute records or denominator entries.
            if batch_id:
                env.reset()
            for slot, index in enumerate(indices):
                command.reset_to_frame(torch.tensor([slot], device=device), int(starts[index]))
            obs, reset_audit = refresh_manual_reset_observations(base)
            print("CORPUS_MANUAL_RESET_REFRESH " + json.dumps(dict(batch=batch_id, **reset_audit)), flush=True)
            sim_ctx = wm.sim_context.read(base)
            active = np.zeros(num_envs, dtype=bool)
            active[:len(indices)] = True
            trajectories = [{"full_state": [], "actions": [], "rewards": []} for _ in indices]
            history = None
            with torch.inference_mode():
                for step in range(int(lengths[indices].max()) + 1):
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
                    if not np.isfinite(reward_cpu[active]).all():
                        raise RuntimeError("Nonfinite active reward; evaluation is incomplete")
                    done_cpu = dones.bool().cpu().numpy()
                    tm = base.termination_manager
                    terms = {name: tm.get_term(name).cpu().numpy().copy() for name in tm.active_terms}
                    failed = tm.terminated.cpu().numpy().copy()
                    for slot in np.flatnonzero(active):
                        index = int(indices[slot])
                        trace = trajectories[slot]
                        trace["full_state"].append(state_cpu[slot])
                        trace["actions"].append(action_cpu[slot])
                        trace["rewards"].append(float(reward_cpu[slot]))
                        if done_cpu[slot]:
                            reasons = [name for name, values in terms.items() if values[slot]]
                            success = bool(terms["motion_end"][slot] and not failed[slot])
                            duration = len(trace["rewards"])
                            if records[index] is not None:
                                raise RuntimeError("Duplicate corpus record")
                            records[index] = {
                                "index": index, "name": names[index], "success": success,
                                "reference_frames": int(lengths[index]), "steps": duration,
                                "seconds": duration * base.step_dt, "termination_terms": reasons,
                                "reward_sum": float(np.sum(trace["rewards"])),
                                "trajectory": f"trajectory-{index:06d}.npz", "batch": batch_id,
                                "slot": int(slot),
                            }
                            if success and duration != int(lengths[index]):
                                raise RuntimeError("Unexpected motion-end duration; protocol needs correction")
                            np.savez_compressed(output / records[index]["trajectory"], **{
                                key: np.asarray(value, dtype=np.float32) for key, value in trace.items()})
                            trajectories[slot] = None
                            active[slot] = False
                            print("CORPUS_CLIP_RESULT " + json.dumps(records[index]), flush=True)
                    obs = next_obs
                    if not active.any():
                        break
                    if step % 1000 == 0:
                        print(f"CORPUS_PROGRESS batch={batch_id} step={step} remaining={active.sum()}", flush=True)
            if active.any():
                raise RuntimeError("Some clips did not reach a native termination")
            batch_records = [records[int(index)] for index in indices]
            (output / f"batch-{batch_id:04d}.json").write_text(
                json.dumps(batch_records, indent=2) + "\n", encoding="utf-8")
            print("BATCH_COMPLETED " + json.dumps({"batch": batch_id,
                "clips": len(indices), "completed": offset + len(indices),
                "successes": sum(row["success"] for row in batch_records)}), flush=True)
        if any(row is None for row in records):
            raise RuntimeError("Corpus evaluation has missing records")
        result = {
            "status": "full_corpus_evaluation_completed", "clips": count,
            "manual_reset_version": 2,
            "num_envs": num_envs, "batch_order": args.order,
            "successes": sum(row["success"] for row in records),
            "failures": sum(not row["success"] for row in records),
            "elapsed_seconds": time.monotonic() - started, "records": records,
            "paper_results_reproduced": False,
            "goal_assembly_backend": goal_backend,
        }
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print("OFFICIAL_CORPUS_EVAL_PASS " + json.dumps(
            {key: value for key, value in result.items() if key != "records"}), flush=True)


if __name__ == "__main__":
    main()
