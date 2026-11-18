"""Released SONIC G1 full-clip inference in the corrected-v2 common CUDA simulator.

Uses released SONIC original G1 weights with its own control parameters.
Keeps the native common reward, termination, physical scene and corrected reset.
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
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--batched", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--motions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-clips", type=int, required=True)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--order", choices=("index", "length"), default="length")
    parser.add_argument("--onnx-provider", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.num_envs < 1 or args.expected_clips < 1:
        parser.error("positive environment and clip counts are required")
    os.environ["MUJOCO_GL"] = "disable"
    os.environ["WANDB_MODE"] = "offline"

    import numpy as np
    import torch
    import supermimic.tasks  # registers author tasks
    from mjlab.tasks.registry import load_env_cfg
    from supermimic.scripts.train_utils import create_env, setup_device
    from manual_reset_observations import refresh_manual_reset_observations

    from sonic_g1_runtime import configure_control,verify_compiled_control,SonicG1Controller
    models,batched=args.models.resolve(),args.batched.resolve()
    motions,output=args.motions.resolve(),args.output.resolve()
    contract=json.loads(args.contract.read_bytes())
    task="SuperMimic-GenCoord-G1"
    import supermimic.representations as representations
    representations._compiled_assemble_goal=representations._assemble_goal
    goal_backend="eager"
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
    preserved_armatures=configure_control(cfg,contract)
    provider = {"cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider"}[args.onnx_provider]
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
        "initial_reference_state_check": "Full71D matches native clipped reference; atol=1e-5,rtol=1e-5 before first action",
        "training": "None: released pretrained SONIC original G1-mode weights",
        "paper_results_reproduced": False,"training_algorithm": "pretrained-SONIC",
        "SONIC_source_commit": contract["source_commit"],
        "SONIC_control_contract_sha256": digest(args.contract),
        "SONIC_ONNX_provider": provider,
        "physics_provider": "CUDA MuJoCo Warp, pinned native common G1 scene",
        "SONIC_control": "Own29joint gains/defaults/scales/effort bounds; common armatures retained",
        "history_initialization": "First reference sensor repeated, matching corrected-v2; SONIC training reset unconfirmed",
        "reference_window": "G1 mode0;10 current/future frames spaced5 at50Hz, clamp to each clip end; full base-relative orientation",
        "author_exact_SONIC_benchmark_protocol_confirmed": False,
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
        # ContactSensor.update replaces history buffers with torch.roll results.
        # Keep their later in-place reset inside the same inference context.
        # Construct/load model parameters first, outside inference mode.
        evaluation.enter_context(torch.inference_mode())
        env.reset()
        base = env.unwrapped
        command = base.command_manager.get_term("motion")
        if not np.array_equal(command.motion.motion_ends.cpu().numpy(), ends):
            raise RuntimeError("Loaded motion boundaries differ from the frozen pack")
        robot = base.scene["robot"]
        controller=SonicG1Controller(models,batched,args.contract,motions,list(robot.joint_names),
            provider=provider, profile_dir=output / "onnx-profiles")
        protocol["compiled_SONIC_control_readback"]=verify_compiled_control(base.sim.mj_model,contract,preserved_armatures)
        action_term=base.action_manager.get_term(base.action_manager.active_terms[0])
        assert list(action_term.target_names)==list(robot.joint_names)
        for i,name in enumerate(robot.joint_names):
            row=next(row for row in contract["joints_in_policy_order"] if row["name"]==name)
            assert torch.allclose(action_term.scale[:,i],torch.full_like(action_term.scale[:,i],row["action_scale"]),atol=1e-6)
            assert torch.allclose(action_term.offset[:,i],torch.full_like(action_term.offset[:,i],row["default_joint_position"]),atol=1e-6)
        for field in ["actuator_gainprm","actuator_biasprm","actuator_forcerange","dof_armature"]:
            cpu=np.asarray(getattr(base.sim.mj_model,field))
            actual=getattr(base.sim.model,field).detach().cpu().numpy()
            assert np.allclose(actual,cpu,atol=1e-5,rtol=1e-6),(field,actual.shape,cpu.shape)
        protocol["all_compiled_CPU_and_live_CUDA_control_fields_verified"]=True
        print("SONIC_COMPILED_AND_CUDA_CONTROL_READBACK_PASS",flush=True)
        limits = robot.data.soft_joint_pos_limits
        if not torch.equal(limits, limits[:1].expand_as(limits)):
            raise RuntimeError("Record per-world limits before evaluating heterogeneous joint limits")
        protocol["native_joint_names"] = list(robot.joint_names)
        protocol["native_soft_joint_pos_limits"] = limits[0].cpu().tolist()
        (output / "protocol.json").write_text(json.dumps(protocol, indent=2, default=str) + "\n", encoding="utf-8")
        max_first_state_error = 0.0
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
            active_count = len(indices)
            times = command.time_steps[:active_count]
            expected_joint_pos = torch.clamp(command.joint_pos[:active_count],
                limits[:active_count, :, 0], limits[:active_count, :, 1])
            expected_state = torch.cat((command.motion.root_pos_w[times],
                command.motion.root_quat_w[times], expected_joint_pos,
                command.motion.root_lin_vel_w[times], command.motion.root_ang_vel_w[times],
                command.joint_vel[:active_count]), -1)
            actual_state = obs["full_state"][:active_count]
            if not torch.allclose(actual_state, expected_state, atol=1e-5, rtol=1e-5):
                raise RuntimeError("First actor observation does not match the intended native clipped reference")
            first_error = float((actual_state - expected_state).abs().max())
            max_first_state_error = max(max_first_state_error, first_error)
            reset_audit.update(reference_initial_state_verified=True,
                               reference_first_state_max_abs_error=first_error)
            print("CORPUS_MANUAL_RESET_REFRESH " + json.dumps(dict(batch=batch_id, **reset_audit)), flush=True)
            active = np.zeros(num_envs, dtype=bool)
            active[:len(indices)] = True
            trajectories = [{"full_state": [], "actions": [], "rewards": []} for _ in indices]
            controller.reset()
            with torch.inference_mode():
                for step in range(int(lengths[indices].max()) + 1):
                    state_cpu=obs["full_state"].cpu().numpy().copy()
                    times_cpu=command.time_steps.cpu().numpy().copy()
                    actions=torch.from_numpy(controller(state_cpu,times_cpu)).to(device=device)
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
            "initial_reference_state_verified_for_all_clips": True,
            "first_reference_state_max_abs_error": max_first_state_error,
            "num_envs": num_envs, "batch_order": args.order,
            "successes": sum(row["success"] for row in records),
            "failures": sum(not row["success"] for row in records),
            "elapsed_seconds": time.monotonic() - started, "records": records,
            "paper_results_reproduced": False,
            "goal_assembly_backend": goal_backend,
            "SONIC_calls": controller.calls, "ONNX_provider": provider,
            "ONNX_runtime_compute_profiles": controller.profile_records,
            "original_batch1_vs_batched_observed_state_comparisons": controller.original_comparisons,
            "maximum_original_token_difference": controller.maximum_token_difference,
            "maximum_original_action_difference": controller.maximum_action_difference,
            "physical_robot_deployment": False,"SONIC_training_reset_confirmed": False,
        }
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print("OFFICIAL_CORPUS_EVAL_PASS " + json.dumps(
            {key: value for key, value in result.items() if key != "records"}), flush=True)


if __name__ == "__main__":
    main()
