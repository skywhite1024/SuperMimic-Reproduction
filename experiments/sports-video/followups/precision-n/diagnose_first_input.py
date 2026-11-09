"""Prepared post-N diagnostic: first policy inputs only, no physical steps.

Reuse N's SHA-verified frozen J inputs and initialize only the three fixed single
environments. Capture native inputs, cross history/goal, and compare repeating
before versus after input construction. This is not a parity or performance
acceptance test. Original I/J/N code, precision protocols and gates stay intact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import signal
import sys
import time

N_RECEIPT_SHA = "c75e0de0b723050b99d44445d36117708a3bd0b54fe0820cca165ac0a7622d1a"
INDICES = (0, 20, 39)
OBS_PARTS = {"joint_pos": (0, 29), "joint_vel": (29, 58),
             "local_angular_vel": (58, 61), "gravity": (61, 64),
             "previous_action": (64, 93), "relative_rotation6": (93, 99),
             "F0_zero_XY": (99, 101)}
GOAL_PARTS = {"position_delta": (0, 3), "rotation6": (3, 9),
              "local_linear_vel": (9, 12), "local_angular_vel": (12, 15),
              "joint_pos": (15, 44), "joint_vel_scaled_0p1": (44, 73)}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked(path, expected):
    actual = sha(path)
    if actual != expected:
        raise ValueError(f"Immutable input SHA mismatch: {path}")
    return actual


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def flags(torch, networks):
    return {"torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0), "AMP_DTYPE": str(networks.AMP_DTYPE),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            "allow_fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}


def apply_recorded_flags(torch, networks, recorded):
    torch.set_float32_matmul_precision(recorded["float32_matmul_precision"])
    torch.backends.cuda.matmul.allow_tf32 = recorded["cuda_matmul_allow_tf32"]
    torch.backends.cudnn.allow_tf32 = recorded["cudnn_allow_tf32"]
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = recorded["allow_bf16_reduced_precision_reduction"]
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = recorded["allow_fp16_reduced_precision_reduction"]
    torch.use_deterministic_algorithms(recorded["deterministic_algorithms"])
    networks.AMP_DTYPE = torch.bfloat16


def difference(left, right):
    import torch
    if left.shape != right.shape or not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise ValueError("Input differences require matching finite tensors")
    delta = (left.double() - right.double()).abs()
    return {"shape": list(left.shape), "max_abs": float(delta.max()),
            "rms": float(delta.square().mean().sqrt()),
            "nonzero_count": int((delta != 0).sum())}


def input_differences(history, goal, reference_history, reference_goal):
    return {"history": difference(history, reference_history),
            "history_components": {name: difference(history[..., lo:hi], reference_history[..., lo:hi])
                                   for name, (lo, hi) in OBS_PARTS.items()},
            "goal": difference(goal, reference_goal),
            "goal_components": {name: difference(goal.reshape(-1, 73)[..., lo:hi],
                                                 reference_goal.reshape(-1, 73)[..., lo:hi])
                                for name, (lo, hi) in GOAL_PARTS.items()}}


def repeat_rows(value, count=40):
    return value.repeat(count, *([1] * (value.ndim - 1)))


def build_inputs(mdp, current, hist, goal_frames, count=1):
    from unisupermimic.rl.algos.supermimic.geometry import unflatten_state
    state, aprev, context, frames = [repeat_rows(current[key], count) for key in ("s", "a_prev", "xi", "f")]
    obs = mdp.obs(unflatten_state(state), aprev, context, frames)
    return obs[:, None].repeat(1, hist, 1), mdp.goal(frames, goal_frames)


def forward_crosses(policy, frozen, native, preconstructed, index, check_deadline):
    """Same policy instance, preserving final policy batch shape per comparison."""
    fh, fg = frozen["history"][index:index+1], frozen["goal"][index:index+1]
    nh, ng = native
    target = frozen["J_action"][index:index+1]
    actual = policy(frozen["history"], frozen["goal"])[index:index+1]
    report = {"fresh_policy_frozen_actual40_vs_J": difference(actual, target),
              "fresh_policy_frozen_actual40_action": actual[0].cpu().tolist(), "crosses": {}}
    for label, history, goal in (("frozen_history_frozen_goal", fh, fg),
                                  ("native_history_frozen_goal", nh, fg),
                                  ("frozen_history_native_goal", fh, ng),
                                  ("native_history_native_goal", nh, ng)):
        check_deadline()
        one = policy(history, goal)
        forty = policy(repeat_rows(history), repeat_rows(goal))[index:index+1]
        report["crosses"][label] = {
            "batch1_vs_recorded_J": difference(one, target),
            "repeat40_vs_recorded_J": difference(forty, target),
            "repeat40_vs_fresh_frozen_actual40": difference(forty, actual),
            "batch1_vs_repeat40": difference(one, forty),
            "batch1_action": one[0].cpu().tolist(), "repeat40_action": forty[0].cpu().tolist()}
    ph, pg = preconstructed
    prediction = policy(ph, pg)[index:index+1]
    report["preconstruction_repeat40_vs_recorded_J"] = difference(prediction, target)
    report["preconstruction_repeat40_action"] = prediction[0].cpu().tolist()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i-spec", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--j-root", type=Path, required=True)
    parser.add_argument("--n-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=250)
    args = parser.parse_args(argv)
    if not 0 < args.max_seconds <= 290:
        parser.error("Soft budget must be in (0,290]; external TERM295 + KILL5 <=300s")
    for name in ("i_spec", "runtime_dir", "j_root", "n_root", "output"):
        setattr(args, name, getattr(args, name).resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = {"status": "starting", "protocol": "post-N first-input isolation; prepared new protocol",
              "script_sha256": sha(__file__), "model": "parent", "device": "cuda",
              "fixed_indices": list(INDICES), "max_seconds": args.max_seconds,
              "physical_steps": 0, "training_performed": False, "performance_evaluation": False,
              "original_J_N_gate_modified": False, "acceptance_gate": None, "clips": []}
    receipt = args.output / "first-input-receipt.json"
    write(receipt, report)

    def check_deadline():
        if time.perf_counter() - started >= args.max_seconds:
            raise TimeoutError("Finite first-input diagnostic deadline")

    def interrupted(signum, frame):
        raise TimeoutError("Finite first-input diagnostic interrupted")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        n_path = args.n_root / "precision-receipt.json"
        checked(n_path, N_RECEIPT_SHA)
        n = json.loads(n_path.read_text())
        frozen_path = args.n_root / "frozen-real-J-inputs.pt"
        checked(frozen_path, n["reconstruction"]["frozen_inputs_sha256"])
        for name, digest in n["runtime_sha256"].items():
            checked(args.runtime_dir / name, digest)
        summary_path = args.j_root / "parent/summary.json"
        checked(summary_path, n["J_summary_sha256"])
        checked(args.j_root / "runtime/evaluate_batch.py", n["J_evaluator_sha256"])
        summary = json.loads(summary_path.read_text())
        sys.path.insert(0, str(args.runtime_dir))
        import numpy as np
        import torch
        import evaluate as single_eval
        import sports_runtime as runtime
        from unisupermimic.rl.algos.supermimic import networks
        from unisupermimic.rl.contracts.supermimic import SUPERMIMIC_STATE_KEY
        torch.set_num_threads(4)
        if torch.cuda.device_count() != 1:
            raise ValueError("Exactly one visible CUDA device required")
        model = next(row for row in json.loads(args.i_spec.read_text())["models"] if row["id"] == "parent")
        if not model["available"] or model["feedback"] != "F0" or model["sha256"] != n["checkpoint_sha256"]:
            raise ValueError("Expected unchanged parent checkpoint and F0 migration")
        checked(model["path"], n["checkpoint_sha256"])
        saved, preparation = single_eval.prepare_checkpoint(torch.load(model["path"], map_location="cpu", weights_only=True), "F0")
        if saved["args"].get("amp") != "bf16":
            raise ValueError("Expected I/J BF16 parent")
        recorded_flags = n["bf16_runtime_flags"]
        apply_recorded_flags(torch, networks, recorded_flags)
        report["actual_start_flags"] = flags(torch, networks)
        report["flags_differ_from_N"] = {key: [recorded_flags[key], report["actual_start_flags"][key]]
                                          for key in recorded_flags if recorded_flags[key] != report["actual_start_flags"][key]}
        if report["flags_differ_from_N"]:
            raise ValueError("Runtime/hardware differs from N; do not mix protocols")
        frozen = next(row for row in torch.load(frozen_path, map_location="cuda", weights_only=True) if row["step"] == 0)
        if tuple(frozen["history"].shape) != (40, 10, 101) or tuple(frozen["goal"].shape) != (40, 730):
            raise ValueError("Frozen first-frame shape changed")
        _, sources, _ = runtime.load_manifest(args.j_root / "old40-manifest.json")
        if len(sources) != 40 or any(row["group"] != "old" for row in sources):
            raise ValueError("Expected fixed old40 manifest")
        report.update(status="running", checkpoint_preparation=preparation,
                      source_sha256={"N_receipt": N_RECEIPT_SHA, "N_frozen_inputs": sha(frozen_path),
                                     "J_summary": sha(summary_path), "I_spec": sha(args.i_spec),
                                     "checkpoint": model["sha256"], **n["runtime_sha256"]},
                      original_action_tolerance=n["frozen_input_replay_action_tolerance"],
                      strict_FP32_scope="Forward only; existing MotionLib caches are not recomputed")
        write(receipt, report)
        for index in INDICES:
            check_deadline()
            source = sources[index]
            if (summary["clips"][index]["clip"], summary["clips"][index]["source_sha256"]) != (source["path"].stem, source["sha256"]):
                raise ValueError("J reference ordering changed")
            row = {"env_id": index, "clip": source["path"].stem, "status": "initializing"}
            report["clips"].append(row)
            write(receipt, report)
            env = None
            stage = time.perf_counter()
            try:
                apply_recorded_flags(torch, networks, recorded_flags)
                env = runtime.make_env([source["path"]], "cuda", 1, 0, "F0", source["fps"], evaluation=True)
                mdp, motion = env.supermimic_mdp, env.supermimic_mdp.motion
                policy = single_eval.load_policy(env, saved, device="cuda")
                _, info = env.reset_to_frames(motion.clip_start.clone())
                current = info[SUPERMIMIC_STATE_KEY]
                row["initialization_seconds"] = time.perf_counter() - stage
                row["native_flags"] = flags(torch, networks)
                if row["native_flags"] != recorded_flags:
                    raise ValueError("Environment construction changed N precision flags")
                row["policy_tensors_equal_checkpoint"] = all(torch.equal(value.cpu(), saved["pi"][key])
                                                              for key, value in policy.state_dict().items())
                if not row["policy_tensors_equal_checkpoint"]:
                    raise ValueError("Loaded policy differs from checkpoint")
                # Fail explicitly if any accidental later code calls physical step.
                def forbidden_step(*unused_args, **unused_kwargs):
                    raise RuntimeError("First-input diagnostic forbids all physical steps")
                env.step = forbidden_step
                check_deadline()
                with torch.inference_mode():
                    native = build_inputs(mdp, current, saved["args"]["hist"], saved["args"]["goal_frames"])
                    before40 = build_inputs(mdp, current, saved["args"]["hist"], saved["args"]["goal_frames"], 40)
                    selected40 = tuple(value[index:index+1] for value in before40)
                    fh, fg = frozen["history"][index:index+1], frozen["goal"][index:index+1]
                    row["native_inputs_vs_frozen_J"] = input_differences(*native, fh, fg)
                    row["repeat_before_construction_vs_frozen_J"] = input_differences(*selected40, fh, fg)
                    row["repeat_before_vs_after_construction"] = input_differences(*selected40, *native)
                    n_context = torch.tensor(n["reconstruction"]["context"][index:index+1], device="cuda", dtype=current["xi"].dtype)
                    row["context_vs_N_J_reconstruction"] = difference(current["xi"], n_context)
                    n_pair = next(pair for pair in n["rollouts"] if pair["env_id"] == index)
                    trace_path = args.n_root / "original_policy1" / f"clip-{index:04d}-{source['path'].stem}" / "trajectory.npz"
                    checked(trace_path, n_pair["original_policy1"]["metrics"]["trajectory_sha256"])
                    with np.load(trace_path, allow_pickle=False) as trace:
                        n_state = torch.as_tensor(trace["state"][0:1].copy(), device="cuda")
                    row["initial_state_vs_N_single"] = difference(current["s"], n_state)
                    capture = {key: value.detach().cpu().numpy() for key, value in current.items() if torch.is_tensor(value)}
                    capture.update(history=native[0].cpu().numpy(), goal=native[1].cpu().numpy(),
                                   preconstruction_history40=before40[0].cpu().numpy(),
                                   preconstruction_goal40=before40[1].cpu().numpy(),
                                   frozen_history=fh.cpu().numpy(), frozen_goal=fg.cpu().numpy(),
                                   recorded_J_action=frozen["J_action"][index:index+1].cpu().numpy())
                    capture_path = args.output / f"first-input-env-{index:02d}.npz"
                    np.savez_compressed(capture_path, **capture)
                    row.update(status="native_inputs_captured_forward_checks_pending", capture_file=capture_path.name,
                               capture_sha256=sha(capture_path))
                    write(receipt, report)
                    row["BF16_cross_forwards"] = forward_crosses(policy, frozen, native, before40, index, check_deadline)
                    write(receipt, report)
                    check_deadline()
                    # A separate control. Cached root velocities remain the original
                    # single-clip values; this is not a fully FP32 MotionLib rebuild.
                    networks.AMP_DTYPE = None
                    torch.set_float32_matmul_precision("highest")
                    torch.backends.cuda.matmul.allow_tf32 = False
                    torch.backends.cudnn.allow_tf32 = False
                    strict_native = build_inputs(mdp, current, saved["args"]["hist"], saved["args"]["goal_frames"])
                    strict40 = build_inputs(mdp, current, saved["args"]["hist"], saved["args"]["goal_frames"], 40)
                    row["strict_FP32_flags"] = flags(torch, networks)
                    row["strict_construction_vs_original_native"] = input_differences(*strict_native, *native)
                    row["strict_repeat_before_vs_after_construction"] = input_differences(
                        *(value[index:index+1] for value in strict40), *strict_native)
                    row["strict_FP32_same_original_inputs_forwards"] = forward_crosses(policy, frozen, native, before40, index, check_deadline)
                    row["strict_FP32_rebuilt_inputs_forwards"] = forward_crosses(policy, frozen, strict_native, strict40, index, check_deadline)
                    capture.update(strict_history=strict_native[0].cpu().numpy(), strict_goal=strict_native[1].cpu().numpy(),
                                   strict_preconstruction_history40=strict40[0].cpu().numpy(),
                                   strict_preconstruction_goal40=strict40[1].cpu().numpy())
                    np.savez_compressed(capture_path, **capture)
                    row.update(status="first_input_capture_complete", capture_file=capture_path.name,
                               capture_sha256=sha(capture_path))
            finally:
                apply_recorded_flags(torch, networks, recorded_flags)
                row["restored_flags"] = flags(torch, networks)
                if env is not None:
                    try:
                        env.close()
                    finally:
                        env.backend.close()
                row["wall_seconds"] = time.perf_counter() - stage
                write(receipt, report)
        report["status"] = "all_three_first_input_captures_complete_not_a_parity_gate"
    except BaseException as error:
        report.update(status="incomplete_or_error", error_type=type(error).__name__, error=str(error))
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        write(receipt, report)
        print("FIRST_INPUT_DIAGNOSTIC " + json.dumps({key: report[key] for key in
              ("status", "physical_steps", "elapsed_seconds")}), flush=True)
    return 0 if report["status"] == "all_three_first_input_captures_complete_not_a_parity_gate" else 2


if __name__ == "__main__":
    raise SystemExit(main())
