"""Native per-clip physical evaluation with fixed clocks and retained failures.

The native first failure stops that clip. Errors are reported on its observed
prefix and explicitly censored; missing future errors are never filled with zero.
Returns always use the planned full-reference denominator, including failures.
Completion under the native gate is NOT strict contact/quality acceptance.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from sports_runtime import (STUDY, extra_columns, load_manifest, make_env,
                            migrate_checkpoint, sha)
from unisupermimic.rl.algos.supermimic.geometry import unflatten_state, yaw_mat
from unisupermimic.rl.contracts.supermimic import SUPERMIMIC_STATE_KEY, SUPERMIMIC_TRANSITION_KEY
from unisupermimic.tasks.motion_tracking.g1.supermimic.evaluate import load_policy
from unisupermimic.tasks.motion_tracking.g1.supermimic.mdp import MAX_STEP_REWARD
from unisupermimic.tasks.motion_tracking.g1.supermimic.motion import CONTROL_DT


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare_checkpoint(checkpoint, feedback):
    dimension = checkpoint["pi"]["obs_norm.mean"].numel()
    if dimension == 99:
        return migrate_checkpoint(checkpoint, feedback), "common_99d_parent_zero_column_migration"
    if dimension != 101:
        raise ValueError(f"Unsupported policy observation dimension {dimension}")
    variant = checkpoint.get("variant", {})
    if variant.get("study") != "sports-video-xy" or variant.get("feedback") != feedback:
        raise ValueError("101D checkpoint must declare the matching sports-video-xy F0/F1 protocol")
    if variant.get("xy_frame") != "reference_yaw" or variant.get("reward_changed") is not False:
        raise ValueError("Checkpoint has incompatible coordinate/reward semantics")
    if variant.get("termination_changed") is not False:
        raise ValueError("Checkpoint has incompatible termination semantics")
    if feedback == "F0":
        cols = extra_columns(checkpoint["args"]["hist"], checkpoint["args"]["z_dim"])
        if bool((checkpoint["pi"]["dec.0.weight"][:, cols] != 0).any()):
            raise ValueError("F0 checkpoint contains nonzero new decoder columns")
    return checkpoint, "trained_101d_checkpoint"


def snapshot(mdp, current, frame):
    state = unflatten_state(current["s"])
    reference_p, reference_r = mdp.motion.root_p[frame], mdp.motion.root_R[frame]
    bodies = mdp.kin.forward(*state)[0]
    delta_rotation = yaw_mat(state[1] @ reference_r.transpose(-1, -2))
    anchor = torch.cat((state[0][..., :2], reference_p[..., 2:3]), -1)
    local_reference = anchor[:, None] + (
        delta_rotation[:, None] @ (mdp.motion.body_p[frame]-reference_p[:, None])[..., None]).squeeze(-1)
    yaw_delta = torch.atan2(state[1][:, 1, 0], state[1][:, 0, 0]) - torch.atan2(reference_r[:, 1, 0], reference_r[:, 0, 0])
    angle = torch.acos(((torch.einsum("bij,bij->b", state[1], reference_r)-1)/2).clamp(-1, 1))
    values = {
        "state": current["s"][0],
        "reference": torch.cat((reference_p[0], reference_r[0].flatten(), mdp.motion.q[frame][0])),
        "root_xy_error": (state[0]-reference_p)[0, :2],
        "root_error_m": (state[0]-reference_p).norm(dim=-1)[0],
        "yaw_error_rad": torch.atan2(torch.sin(yaw_delta), torch.cos(yaw_delta)).abs()[0],
        "root_orientation_error_rad": angle[0],
        "body_global_mean_m": (bodies-mdp.motion.body_p[frame]).norm(dim=-1).mean(-1)[0],
        "body_local_mean_m": (bodies-local_reference).norm(dim=-1).mean(-1)[0],
    }
    return {key: value.detach().cpu().numpy().copy() for key, value in values.items()}


def summarize_trace(trace, row):
    if not trace["state"]:
        return {"tracking_error_scope": "no_observed_states", "tracking_errors_censored": True}
    xy = np.asarray(trace["root_xy_error"], dtype=np.float64)
    squared = np.square(xy).sum(-1)
    return {
        "root_xy_rmse_m": float(np.sqrt(squared.mean())),
        "root_xy_peak_m": float(np.sqrt(squared.max())),
        "root_xy_final_error_m": float(np.sqrt(squared[-1])),
        "yaw_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(np.square(trace["yaw_error_rad"]))))),
        "root_orientation_geodesic_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(np.square(trace["root_orientation_error_rad"]))))),
        "body_global_mean_m": float(np.mean(trace["body_global_mean_m"])),
        "body_local_mean_m": float(np.mean(trace["body_local_mean_m"])),
        "tracking_error_scope": "full_reference" if row["full_reference_completed"] else "observed_prefix_only",
        "tracking_errors_censored": not row["full_reference_completed"],
        "tracking_observed_frames": len(trace["state"]),
    }


def append_snapshot(trace, mdp, current, frame):
    values = snapshot(mdp, current, frame)
    if not all(np.isfinite(value).all() for value in values.values()):
        raise FloatingPointError("Nonfinite physical state/reference tracking metrics")
    for key, value in values.items():
        trace[key].append(value)


def evaluate_clip(source, saved, *, device, feedback, seed, action_mode, output,
                  max_steps=None, deadline=None):
    """Evaluate exactly one clip; keep any observed trace even after exceptions."""
    output.mkdir(parents=True, exist_ok=False)
    row = {"clip": source["path"].stem, "group": source["group"], "path": str(source["path"]),
           "source_sha256": source["sha256"], "source_fps": source["fps"], "status": "not_started",
           "observed_steps": 0, "planned_steps": None, "first_native_hard_failure_s": None,
           "full_reference_completed": False, "native_hard_failure": False,
           "completed_without_native_failure": False, "post_initialization_resets": 0,
           "reward_sum_until_stop": 0., "norm_return_full_reference": None,
           "strict_contact_quality_acceptance": None}
    state_keys = ("state", "reference", "root_xy_error", "root_error_m", "yaw_error_rad",
                  "root_orientation_error_rad", "body_global_mean_m", "body_local_mean_m")
    trace = {key: [] for key in (*state_keys, "action_raw", "action_applied", "reward", "native_hard", "native_done")}
    env, started = None, time.perf_counter()
    try:
        if deadline is not None and time.perf_counter() >= deadline:
            row["status"] = "not_run_engineering_deadline"
            return row
        env = make_env([source["path"]], device, 1, seed, feedback, source["fps"], evaluation=True)
        mdp, motion = env.supermimic_mdp, env.supermimic_mdp.motion
        policy = load_policy(env, saved, device=device)
        planned = int(motion.clip_len[0]-1)
        row.update(planned_steps=planned, planned_seconds=planned*CONTROL_DT,
                   normalization_denominator=planned*MAX_STEP_REWARD)
        _, info = env.reset_to_frames(motion.clip_start.clone())
        current, history = info[SUPERMIMIC_STATE_KEY], None
        def forbidden_reset(*args, **kwargs):
            row["post_initialization_resets"] += 1
            raise RuntimeError("Evaluation forbids resets after the initial frame")
        env.reset = forbidden_reset
        with torch.inference_mode():
            for step in range(planned+1):
                expected_frame = motion.clip_start + step
                if not torch.equal(current["f"], expected_frame):
                    raise RuntimeError("Reference clock was reset, skipped or changed")
                if not bool(torch.isfinite(current["s"]).all()):
                    raise FloatingPointError("Nonfinite physical state")
                append_snapshot(trace, mdp, current, expected_frame)
                if step == planned:
                    row["status"] = "completed"
                    row["full_reference_completed"] = True
                    break
                if max_steps is not None and step >= max_steps:
                    row["status"] = "truncated_engineering_steps"
                    break
                if deadline is not None and time.perf_counter() >= deadline:
                    row["status"] = "truncated_engineering_deadline"
                    break
                obs = mdp.obs(unflatten_state(current["s"]), current["a_prev"], current["xi"], current["f"])
                hist = int(saved["args"]["hist"])
                history = obs[:, None].repeat(1, hist, 1) if history is None else torch.cat((history[:, 1:], obs[:, None]), 1)
                action_raw = policy(history, mdp.goal(current["f"], int(saved["args"]["goal_frames"])))
                if not bool(torch.isfinite(action_raw).all()):
                    raise FloatingPointError("Nonfinite policy action")
                action = action_raw if action_mode == "raw" else action_raw.clamp(-20, 20)
                result = env.step(action)
                transition = result.info[SUPERMIMIC_TRANSITION_KEY]
                reward = float(transition["r"][0])
                if not math.isfinite(reward):
                    raise FloatingPointError("Nonfinite native reward")
                hard, done = bool(transition["term"][0]), bool(transition["done"][0])
                trace["action_raw"].append(action_raw[0].cpu().numpy().copy())
                trace["action_applied"].append(action[0].cpu().numpy().copy())
                trace["reward"].append(reward); trace["native_hard"].append(hard); trace["native_done"].append(done)
                row["reward_sum_until_stop"] += reward
                row["observed_steps"] = step+1
                current = result.info[SUPERMIMIC_STATE_KEY]
                if hard or (done and step+1 < planned):
                    if not torch.equal(current["f"], motion.clip_start+step+1):
                        raise RuntimeError("Terminated transition changed the reference clock")
                    append_snapshot(trace, mdp, current, current["f"])
                    row.update(status="native_hard_failure" if hard else "unexpected_native_truncation",
                               native_hard_failure=hard,
                               first_native_hard_failure_s=(step+1)*CONTROL_DT if hard else None,
                               full_reference_completed=step+1 == planned)
                    break
        row["completed_without_native_failure"] = row["full_reference_completed"] and not row["native_hard_failure"]
    except Exception as error:
        row.update(status="error", error_type=type(error).__name__, error=str(error),
                   completed_without_native_failure=False)
    finally:
        if row["planned_steps"] is not None:
            row["norm_return_full_reference"] = row["reward_sum_until_stop"] / row["normalization_denominator"]
            row["observed_reference_fraction"] = row["observed_steps"] / row["planned_steps"]
        row["observed_seconds"] = row["observed_steps"]*CONTROL_DT
        row["survival_seconds"] = row["first_native_hard_failure_s"] if row["native_hard_failure"] else row["observed_seconds"]
        row["survival_is_censored_lower_bound"] = not row["native_hard_failure"] and not row["full_reference_completed"]
        row["wall_seconds"] = time.perf_counter()-started
        row.update(summarize_trace(trace, row))
        np.savez_compressed(output / "trajectory.npz", frame_index=np.arange(len(trace["state"])),
                            **{key: np.asarray(value) for key, value in trace.items()})
        row["trajectory_sha256"] = sha(output / "trajectory.npz")
        write_json(output / "metrics.json", row)
        if env is not None:
            try:
                env.close()
            finally:
                env.backend.close()
    return row


def aggregate(rows):
    measured = [row for row in rows if row.get("norm_return_full_reference") is not None]
    complete_metrics = len(measured) == len(rows)
    outcomes = ("completed", "native_hard_failure", "unexpected_native_truncation")
    return {"denominator": len(rows), "native_completed_without_failure": sum(row["completed_without_native_failure"] for row in rows),
            "native_completion_rate": sum(row["completed_without_native_failure"] for row in rows)/len(rows) if rows else None,
            "native_hard_failed": sum(row["native_hard_failure"] for row in rows),
            "errors": sum(row["status"] == "error" for row in rows),
            "budget_truncated_or_unrun": sum("engineering" in row["status"] for row in rows),
            "native_terminal_outcomes": sum(row["status"] in outcomes for row in rows),
            "return_metric_rows": len(measured),
            "macro_return_full_denominator": float(np.mean([row["norm_return_full_reference"] for row in measured])) if complete_metrics and rows else None,
            "frame_weighted_return_full_denominator": sum(row["reward_sum_until_stop"] for row in measured) / sum(row["normalization_denominator"] for row in measured) if complete_metrics and measured else None,
            "strict_contact_quality_pass_count": None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feedback", choices=["F0", "F1"], required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--action-mode", choices=["raw", "bound20"], default="raw")
    parser.add_argument("--label", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--engineering", action="store_true")
    parser.add_argument("--max-steps", type=int, help="Engineering-only per-clip truncation")
    parser.add_argument("--max-seconds", type=float, help="Engineering-only block deadline")
    args = parser.parse_args(argv)
    if (args.max_steps is not None or args.max_seconds is not None) and not args.engineering:
        parser.error("Truncated budgets require --engineering")
    if (args.max_steps is not None and args.max_steps <= 0) or (args.max_seconds is not None and args.max_seconds <= 0):
        parser.error("Budget values must be positive")
    output = (STUDY / "runs" / args.label).resolve()
    if not output.is_relative_to(STUDY / "runs") or output.exists():
        raise ValueError("Use a fresh evaluation label in the study runs directory")
    checkpoint_path, manifest_path = args.checkpoint.resolve(), args.manifest.resolve()
    manifest, sources, fps = load_manifest(manifest_path)
    torch.set_num_threads(2 if args.device == "cpu" else 4)
    if args.device == "cuda" and torch.cuda.device_count() != 1:
        raise ValueError("Expose one CUDA device per independent evaluator")
    saved, preparation = prepare_checkpoint(torch.load(checkpoint_path, map_location="cpu", weights_only=True), args.feedback)
    output.mkdir(parents=True)
    started = time.perf_counter()
    deadline = started+args.max_seconds if args.max_seconds is not None else None
    report = {"status": "running", "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha(checkpoint_path),
              "manifest": str(manifest_path), "manifest_sha256": sha(manifest_path),
              "reference_variant": manifest.get("reference_variant", "unspecified"),
              "training_reference_variant": saved.get("variant", {}).get("reference_variant"),
              "device": args.device, "backend": "mujoco" if args.device == "cpu" else "mjwarp",
              "feedback": args.feedback, "xy_frame": "reference_yaw", "checkpoint_preparation": preparation,
              "action_mode": args.action_mode, "action_calibration_applied": args.action_mode != "raw",
              "source_fps": fps, "control_dt": CONTROL_DT, "seed": args.seed,
              "engineering_only": args.engineering, "max_steps_per_clip": args.max_steps,
              "maximum_seconds": args.max_seconds, "observation_noise": False, "action_noise": False,
              "auto_reset": False, "reward_changed": False, "termination_changed": False,
              "evaluation_protocol": "one native rollout per clip from frame zero; stop at first native failure; include failing transition reward; fixed full-length denominator; no phase edits or DTW",
              "tracking_metrics_protocol": "unmodified world XY; local-body metric separately labelled; prefix errors censored after failure; contact acceptance not implemented",
              "trajectory_schema": {"state": "71D: xyz, wxyz, world linear/angular velocity, 29 joint positions, 29 joint velocities",
                  "reference": "41D: xyz, row-major 3x3 rotation, 29 joint positions",
                  "root_xy_error": "robot world XY minus reference world XY; no spatial alignment",
                  "frame_index": "control frame from t0, seconds = frame_index * 0.02",
                  "actions_rewards": "one fewer row than states; failing transition included"},
              "source_sha256": {name: sha(STUDY/name) for name in ("evaluate.py", "sports_runtime.py")}, "clips": []}
    write_json(output/"summary.json", report)
    for index, source in enumerate(sources):
        row = evaluate_clip(source, saved, device=args.device, feedback=args.feedback, seed=args.seed,
                            action_mode=args.action_mode, output=output/f"clip-{index:04d}-{source['path'].stem}",
                            max_steps=args.max_steps, deadline=deadline)
        report["clips"].append(row)
        report["aggregate"] = aggregate(report["clips"])
        write_json(output/"summary.json", report)
        print("SPORTS_EVAL_CLIP " + json.dumps({key: row[key] for key in
              ("clip", "status", "observed_steps", "planned_steps", "norm_return_full_reference")}), flush=True)
    report["groups"] = {group: aggregate([row for row in report["clips"] if row["group"] == group])
                        for group in sorted({row["group"] for row in report["clips"]})}
    report["wall_seconds"] = time.perf_counter()-started
    report["all_clips_reached_native_outcome"] = report["aggregate"]["native_terminal_outcomes"] == len(sources)
    report["full_protocol_evaluation_complete"] = report["all_clips_reached_native_outcome"] and not args.engineering
    report["status"] = "engineering_results" if args.engineering else "completed" if report["full_protocol_evaluation_complete"] else "incomplete_with_errors"
    write_json(output/"summary.json", report)
    print("SPORTS_EVAL_COMPLETE " + json.dumps({"status": report["status"], "aggregate": report["aggregate"]}), flush=True)
    return report


if __name__ == "__main__":
    main()
