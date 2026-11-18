"""Evaluate saved SuperMimic checkpoints under one-shot root velocity perturbations."""

import argparse
import dataclasses
import json
import os
import time
from pathlib import Path

import numpy as np

from push_recovery_contract import (
    CLIP_FRAMES, DT, METRICS, POSES, PUSH_STEP, STEPS, THRESHOLDS,
    derive_metrics, prepare_reference, pulse_grid, score_from_pre_push_pose, score_trials,
    sha256, write_json,
)
from select_completed_checkpoint import select_completed_checkpoint


def evaluate_pose(env, actor, model, command, pose, frame, seed):
    import torch
    from mjlab.utils.lab_api.math import quat_apply
    from supermimic.representations import extract_actor_goal
    from supermimic.tasks.task_spec import build_actor_obs_td

    from manual_reset_observations import refresh_manual_reset_observations

    base = env.unwrapped
    robot = base.scene["robot"]
    rep = model.representation
    labels, vectors = pulse_grid()
    n = len(labels)
    env.seed(seed)
    env.reset()
    ids = torch.arange(n, device=env.device)
    command.reset_to_frame(ids, frame)
    obs, reset_audit = refresh_manual_reset_observations(base)
    full_state = obs["full_state"]
    xyz = command.motion.root_pos_w[frame]
    q = command.motion.joint_pos[frame]
    reference = torch.cat((xyz, command.motion.root_quat_w[frame], q,
                           torch.zeros_like(full_state[0, 36:])))
    initial = full_state.clone()
    reference_xy_square = reference[4:6].square().sum()
    # GenCoord subtracts the environment origin from the root position.
    expected = reference.expand(n, -1).clone()
    if not torch.allclose(initial, expected, rtol=1e-5, atol=1e-5):
        raise RuntimeError(f"Refreshed reset state differs from {pose} reference: "
                           f"max error={(initial - expected).abs().max().item():.6g}")
    reset_audit["first_state_max_abs_error"] = (initial - expected).abs().max().item()
    print("PUSH_RESET " + json.dumps(dict(pose=pose, **reset_audit)), flush=True)
    state_bank = torch.full((n, STEPS + 1, 71), float("nan"), device=env.device)
    action_bank = torch.full((n, STEPS, 29), float("nan"), device=env.device)
    state_bank[:, 0] = full_state
    alive = torch.ones(n, device=env.device, dtype=torch.bool)
    observed = torch.zeros(n, device=env.device, dtype=torch.long)
    applied = torch.zeros_like(alive)
    impulse_fields = torch.full((n, 2, 6), float("nan"), device=env.device)
    delta = torch.as_tensor(vectors, device=env.device)
    # Per-environment fixed context; the event ranges are degenerate at evaluation.
    sim_context = model.sim_context.read(base)
    history = None
    for step in range(STEPS):
        sensor = rep.compute_actor_sensor(
            rep.represent(obs["full_state"]), sim_context,
            sim_actor_obs=obs["actor"] if rep.actor_obs_mode == "proprio" else None,
            noisy=False,
        )
        new_frame = sensor[:, None]
        history = (new_frame.repeat(1, actor.history_length, 1) if history is None
                   else torch.cat((history[:, 1:], new_frame), dim=1))
        command_td = {"motion": extract_actor_goal(obs["goal"])}
        action = actor(build_actor_obs_td(command_td, history))
        if not torch.isfinite(action[alive]).all():
            raise RuntimeError("Policy produced a nonfinite action before a recorded fall")
        action_bank[alive, step] = action[alive]
        if step == PUSH_STEP:
            # Act on the current observation first. The next policy step sees the
            # disturbance through ordinary sensors without an artificial reset.
            applied = alive & (delta.norm(dim=-1) > 0)
            push_ids = ids[applied]
            if len(push_ids):
                adr = robot.indexing.free_joint_v_adr
                before = base.sim.data.qvel[push_ids[:, None], adr].clone()
                angular_world = quat_apply(obs["full_state"][applied, 3:7], before[:, 3:])
                command_velocity = torch.cat((before[:, :3] + delta[applied], angular_world), dim=-1)
                robot.write_root_link_velocity_to_sim(command_velocity, env_ids=push_ids)
                after = base.sim.data.qvel[push_ids[:, None], adr].clone()
                expected_change = torch.cat((delta[applied], torch.zeros_like(before[:, 3:])), dim=-1)
                if not torch.allclose(after - before, expected_change, rtol=0, atol=2e-5):
                    raise RuntimeError("Raw generalized root velocity did not receive the declared jump")
                impulse_fields[applied, 0] = before
                impulse_fields[applied, 1] = after
        obs, _, dones, _ = env.step(torch.where(alive[:, None], action, 0.0))
        next_state = obs["full_state"]
        if dones.bool().any() or not torch.isfinite(next_state[alive]).all():
            raise RuntimeError("Unexpected automatic reset or nonfinite active post-step state")
        state_bank[alive, step + 1] = next_state[alive]
        observed += alive
        # In GenCoord the anchor is the pelvis. Use the fresh post-physics state.
        bad_z = (next_state[:, 2] - initial[:, 2]).abs() > .25
        bad_tilt = (2.0 * (next_state[:, 4:6].square().sum(-1) - reference_xy_square).abs()) > .8
        alive &= ~(bad_z | bad_tilt)
        if step in (49, PUSH_STEP, 199, 299, STEPS - 1) or not alive.any():
            print(f"PUSH_PROGRESS pose={pose} step={step + 1} survivors={int(alive.sum())}"
                  f" impulses={int(applied.sum())}", flush=True)
        if not alive.any():
            break
    arrays = {
        "states": state_bank.cpu().numpy(), "actions": action_bank.cpu().numpy(),
        "reference": reference.cpu().numpy(), "observed_steps": observed.cpu().numpy(),
        "applied": applied.cpu().numpy(), "vectors": vectors,
        "impulse_qvel": impulse_fields.cpu().numpy(), "fallen": ~alive.cpu().numpy(),
    }
    return arrays, reset_audit


def audit_pose(path):
    labels, expected_vectors = pulse_grid()
    with np.load(path, allow_pickle=False) as data:
        steps = data["observed_steps"]
        applied = data["applied"]
        vectors = data["vectors"]
        if len(steps) != len(labels) or not np.array_equal(vectors, expected_vectors):
            raise RuntimeError("Saved trial grid differs")
        if not np.array_equal(applied, (steps > PUSH_STEP) & (np.linalg.norm(vectors, axis=1) > 0)):
            raise RuntimeError("Unexpected omission or application of an impulse")
        for name, count, extra in (("states", STEPS + 1, 1), ("actions", STEPS, 0)):
            array = data[name]
            mask = np.arange(count) < (steps[:, None] + extra)
            if not np.isfinite(array[mask]).all() or not np.isnan(array[~mask]).all():
                raise RuntimeError(f"Invalid trajectory samples or censoring for {name}")
        qvel = data["impulse_qvel"]
        jumps = qvel[applied, 1] - qvel[applied, 0]
        target = np.pad(vectors[applied], ((0, 0), (0, 3)))
        error = np.max(np.abs(jumps - target), initial=0.0)
        if error > 2e-5 or not np.isnan(qvel[~applied]).all():
            raise RuntimeError("Saved impulse samples disagree with the declared velocity jumps")
        states = data["states"]
        metrics = derive_metrics(states, data["reference"])
        # Recompute first fall, including at the final sample and after the impulse.
        z_error = np.abs(states[..., 2].astype(np.float64) - states[:, :1, 2])
        ref_squared = np.square(data["reference"][4:6].astype(np.float64)).sum()
        gravity_z_error = 2.0 * np.abs(np.square(states[..., 4:6].astype(np.float64)).sum(-1) - ref_squared)
        failures = (z_error > .25) | (gravity_z_error > .8)
        fall = failures.any(axis=1)
        if not (np.array_equal(fall, data["fallen"])
                and np.array_equal(np.where(fall, failures.argmax(1), STEPS), steps)):
            raise RuntimeError("Stored traces do not reproduce first-fall times")
        records = score_trials(labels, vectors, metrics, steps, applied, 1 - 2 * ref_squared)
        if [r["fell"] for r in records] != fall.tolist():
            raise RuntimeError("Metric-based final fall disagrees with the state traces")
        control = records[0]
        eligible = (not control["fell"] and control["prepush_stable_final_1s"]
                    and control["final_1s_stable"])
        control_metrics = {}
        if not control["fell"]:
            for label, region in (("prepush_last_second_max", metrics[0, PUSH_STEP-50:PUSH_STEP+1]),
                                  ("last_second_max", metrics[0, STEPS-50:])):
                control_metrics[label] = dict(zip(METRICS, region.max(0).tolist(), strict=True))
        return {
            "cases": len(records), "impulses": int(applied.sum()),
            "prepush_failures": int((fall & (steps <= PUSH_STEP)).sum()),
            "falls_after_push": int((fall & applied).sum()),
            "pushed_survived": int((~fall & applied).sum()),
            "pushed_settled": sum(row["impulse_applied"] and row["recovery_seconds"] is not None
                                  for row in records),
            "zero_push_eligible": eligible,
            "control_metrics": control_metrics,
            "relative_to_prepush_posture": score_from_pre_push_pose(data, metrics),
            "max_root_qvel_jump_error": float(error),
            "records": records,
            "trace": {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path),
                      "readback_recomputed": True},
        }


def run_evaluation(plan, packed, output):
    os.environ.update(MUJOCO_GL="disable", WANDB_MODE="offline")
    import supermimic.tasks  # noqa: F401
    import torch
    from mjlab.tasks.registry import load_env_cfg
    from supermimic.runners.run_config import check_representation_matches, load_runner_cfg
    from supermimic.runners.supermimic_runner import build_actor, build_world_model
    from supermimic.scripts.train_utils import create_env, setup_device

    seed = 42
    labels, _ = pulse_grid()
    device = setup_device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("The push experiment requires the native CUDA simulator")
    start = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    run = Path(plan["parent"])
    _, files, selection = select_completed_checkpoint(run, plan["collection"])
    resolved_path = run / "resolved-config.json"
    selection["resolved_config_sha256"] = sha256(resolved_path)
    if selection != plan["selection"]:
        raise RuntimeError("The selected native models no longer match the frozen full40 evaluation")
    resolved = json.loads(resolved_path.read_bytes())
    task = resolved["task"]
    runner_cfg = load_runner_cfg({"runner_cfg": resolved["runner"]})
    check_representation_matches(task, runner_cfg.representation)
    if (task != "SuperMimic-GenCoord-G1"
            or runner_cfg.policy.horizon != plan["horizon"]
            or ("critic.pt" in files) != plan["critic_enabled"]):
        raise RuntimeError("Model configuration does not match the stated ablation")
    goal_backend = resolved.get("goal_assembly_backend", "official")
    if goal_backend == "eager":
        import supermimic.representations as representations
        representations._compiled_assemble_goal = representations._assemble_goal
    elif goal_backend != "official":
        raise RuntimeError("Unknown native goal assembly mode")

    cfg = load_env_cfg(task, play=True)
    cfg.seed = seed
    cfg.auto_reset = False
    cfg.episode_length_s = 1e9
    height_fall = cfg.terminations["anchor_pos"]
    orientation_fall = cfg.terminations["anchor_ori"]
    if (height_fall.func.__name__ != "bad_anchor_pos_z_only"
            or height_fall.params["threshold"] != .25
            or orientation_fall.func.__name__ != "bad_anchor_ori"
            or orientation_fall.params["threshold"] != .8):
        raise RuntimeError("Native GenCoord fall rules changed")
    cfg.terminations = {}
    cfg.curriculum = {}
    cfg.metrics = {}
    if set(cfg.events) != {"foot_friction", "base_com", "encoder_bias"}:
        raise RuntimeError("Unexpected events in the native play configuration")
    cfg.events["foot_friction"].params["ranges"] = (1.0, 1.0)
    cfg.events["base_com"].params["ranges"] = {axis: (0., 0.) for axis in range(3)}
    cfg.events["encoder_bias"].params["bias_range"] = (0., 0.)
    cfg.scene.entities["robot"].collisions = tuple(
        dataclasses.replace(c, friction=(1.0,)) for c in cfg.scene.entities["robot"].collisions)
    for group in cfg.observations.values():
        group.enable_corruption = False
    motion = cfg.commands["motion"]
    if motion.anchor_body_name != "pelvis":
        raise RuntimeError("Fall and tilt metrics require the native pelvis anchor")
    motion.motion_file = str(packed)
    motion.sampling_mode = "start"
    motion.pose_range = {}
    motion.velocity_range = {}
    motion.joint_position_range = (0., 0.)
    motion.resampling_time_range = (1e9, 1e9)
    pose_names = tuple(json.loads((packed / "motion_names.json").read_bytes()))
    if pose_names != tuple(plan.get("poses", POSES)):
        raise RuntimeError("The planned static-pose names and packed clips differ")
    if np.load(packed / "motion_ends.npy", allow_pickle=False).tolist() != [
            CLIP_FRAMES * i for i in range(1, len(pose_names) + 1)]:
        raise RuntimeError("Unexpected static-pose clip boundaries")
    protocol = {
        "task": task, "policy": plan, "seed": seed, "dt_s": DT, "cases_per_pose": len(labels),
        "elapsed_steps": STEPS, "impulse_at_step": PUSH_STEP, "reference_poses": pose_names,
        "physics": "native defaults; fixed friction coefficient 1, no noise or randomization",
        "fall": "first fresh state with absolute pelvis height error > .25 m or projected gravity z error > .8",
        "input": "one additive horizontal world root velocity jump after computing the action at t=2 s",
        "recovery": "first post-impulse time within thresholds for the rest of the six-second response, at least 0.5 seconds",
        "zero_push_criterion": "survival and all four thresholds for the last second before nominal push and at the end",
        "thresholds": THRESHOLDS, "state_metric_order": METRICS,
        "goal_backend": goal_backend, "auto_reset": False, "weights_frozen": True,
    }
    write_json(output / "protocol.json", protocol)
    env = create_env(cfg, len(labels), device)
    try:
        if not np.isclose(env.unwrapped.step_dt, DT, atol=1e-8, rtol=0):
            raise RuntimeError("Native policy rate is not 50 Hz")
        model = build_world_model(runner_cfg, env, device)
        model.load_bundle(torch.load(files["world_model.pt"], map_location=device, weights_only=True))
        model.dynamics.eval()
        actor = build_actor(runner_cfg, model, device)
        actor.load_state_dict(torch.load(files["actor.pt"], map_location=device, weights_only=True))
        actor.eval()
        actor.set_slot_mix({"motion": 1.0})
        command = env.unwrapped.command_manager.get_term("motion")
        for i, pose in enumerate(pose_names):
            with torch.inference_mode():
                arrays, reset_audit = evaluate_pose(env, actor, model, command, pose, i * CLIP_FRAMES, seed)
            trace = output / f"{pose}.npz"
            np.savez_compressed(trace, **arrays)
            result = audit_pose(trace)
            result.update(status="pose_sweep_completed", policy=plan["label"], pose=pose,
                          reset=reset_audit, elapsed_s=time.monotonic() - start,
                          protocol_sha256=sha256(output / "protocol.json"))
            write_json(output / f"{pose}.json", result)
            print("PUSH_POSE_RESULT " + json.dumps({k: v for k, v in result.items() if k != "records"}),
                  flush=True)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--xml", type=Path, required=True)
    prepare.add_argument("--joints", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--policy", required=True)
    run.add_argument("--packed", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        receipt = prepare_reference(args.xml, json.loads(args.joints.read_bytes()), args.output / "packed")
        write_json(args.output / "reference.json", receipt)
        print(json.dumps(receipt), flush=True)
    else:
        plan = json.loads(args.plan.read_bytes())
        run_evaluation(plan["policies"][args.policy], args.packed, args.output)


if __name__ == "__main__":
    main()
