"""Finite CUDA BF16 batch-shape isolation using immutable I/J artifacts.

No training, tolerance edits, source edits or writes to I/J. Reconstruct frozen
policy inputs from J's saved first 100 steps, then compare identical inputs under
batch1/repeat40/actual40. Run original and policy-repeat40 single physical envs
for the same fixed three 100-step clips. Original J parity remains unchanged.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import signal
import sys
import time
import numpy as np


class Repeat40Policy:
    """Change only policy GEMM batch shape; keep one physical environment."""
    def __init__(self, policy, target_row):
        if not 0 <= target_row < 40:
            raise ValueError("Original target row must be in [0,40)")
        self.policy, self.target_row = policy, target_row

    def __call__(self, history, goal):
        if history.shape[0] != 1 or goal.shape[0] != 1:
            raise ValueError("Repeat40 adapter is only for one physical env")
        result = self.policy(history.repeat(40, 1, 1), goal.repeat(40, 1))
        return result[self.target_row:self.target_row+1]


def write(path, value):
    raw = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n"
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(raw, encoding="utf-8"); temporary.replace(path)


def flags(torch, networks):
    return {"torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0), "AMP_DTYPE": str(networks.AMP_DTYPE),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            "allow_fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}


def freeze_inputs(sources, saved, j_model, summary, runtime, deadline, output):
    """One native reset, no physical steps; rebuild J histories on actual states."""
    import torch
    from unisupermimic.rl.algos.supermimic.geometry import unflatten_state
    from unisupermimic.rl.contracts.supermimic import SUPERMIMIC_STATE_KEY
    from unisupermimic.tasks.motion_tracking.g1.supermimic.evaluate import load_policy
    traces, file_evidence = [], []
    for index, source in enumerate(sources):
        if time.perf_counter() >= deadline:
            raise TimeoutError("Deadline reading frozen J prefixes")
        row = summary["clips"][index]
        if row["clip"] != source["path"].stem or row["source_sha256"] != source["sha256"]:
            raise ValueError("J fixed clip ordering or reference differs")
        path = j_model/f"clip-{index:04d}-{source['path'].stem}"/"trajectory.npz"
        digest = runtime.sha(path)
        if digest != row["trajectory_sha256"]:
            raise ValueError("J trajectory hash mismatch")
        with np.load(path, allow_pickle=False) as trace:
            traces.append({key: trace[key][:101].copy() for key in ("state", "action_raw", "action_applied")})
        file_evidence.append({"env_id": index, "trajectory": str(path), "sha256": digest})
    common = min(100, *(len(trace["action_raw"]) for trace in traces))
    if common < 1:
        raise ValueError("All forty J rows need at least the initial recorded action")
    sample_steps = [step for step in (0, 1, 10, 50, 99) if step < common]
    env = runtime.make_env([s["path"] for s in sources], "cuda", 40, 0, saved["variant"]["feedback"],
                           sources[0]["fps"], evaluation=True, full_trace=True)
    try:
        mdp, motion = env.supermimic_mdp, env.supermimic_mdp.motion
        _, info = env.reset_to_frames(motion.clip_start.clone())
        current = info[SUPERMIMIC_STATE_KEY]
        state = torch.as_tensor(np.stack([t["state"][:common+1] for t in traces]), device="cuda")
        previous = torch.as_tensor(np.stack([t["action_applied"][:common] for t in traces]), device="cuda")
        initial_error = float((state[:, 0]-current["s"]).abs().max())
        if initial_error > 1e-5:
            raise ValueError("New native reset differs from frozen J initial state")
        policy = load_policy(env, saved, device="cuda")
        history, frozen = None, []
        hist = int(saved["args"]["hist"])
        with torch.inference_mode():
            for step in range(max(sample_steps)+1):
                if time.perf_counter() >= deadline:
                    raise TimeoutError("Deadline reconstructing frozen inputs")
                frames = motion.clip_start+step
                aprev = torch.zeros_like(previous[:, 0]) if step == 0 else previous[:, step-1]
                obs = mdp.obs(unflatten_state(state[:, step]), aprev, current["xi"], frames)
                history = obs[:, None].repeat(1, hist, 1) if history is None else torch.cat((history[:, 1:], obs[:, None]), 1)
                if step in sample_steps:
                    frozen.append({"step": step, "history": history.clone(),
                                   "goal": mdp.goal(frames, int(saved["args"]["goal_frames"])).clone(),
                                   "J_action": torch.as_tensor(np.stack([t["action_raw"][step] for t in traces]), device="cuda")})
        torch.save([{key: value.cpu() if torch.is_tensor(value) else value for key, value in entry.items()} for entry in frozen],
                   output/"frozen-real-J-inputs.pt")
        return policy, frozen, {"maximum_common_recorded_actions": common, "fixed_sample_steps_available": sample_steps,
                                "initial_state_max_abs_vs_J": initial_error, "native_initializations": 1,
                                "physical_steps_in_reconstruction": 0,
                                "context": current["xi"].cpu().tolist(), "source_trajectories": file_evidence,
                                "frozen_inputs_sha256": runtime.sha(output/"frozen-real-J-inputs.pt")}
    finally:
        try:
            env.close()
        finally:
            env.backend.close()


def frozen_differences(policy, frozen, indices, torch):
    rows = []
    with torch.inference_mode():
        for sample in frozen:
            h, g = sample["history"], sample["goal"]
            actual = policy(h, g)
            for index in indices:
                single = policy(h[index:index+1], g[index:index+1])
                repeated = policy(h[index:index+1].repeat(40, 1, 1), g[index:index+1].repeat(40, 1))
                target = actual[index:index+1]
                repeat_target = repeated[index:index+1]
                rows.append({"step": sample["step"], "env_id": index,
                             "single_vs_actual40_max_abs": float((single-target).abs().max()),
                             "single_vs_repeat40_max_abs": float((single-repeat_target).abs().max()),
                             "repeat40_vs_actual40_max_abs": float((repeat_target-target).abs().max()),
                             "repeat40_row0_vs_target_max_abs": float((repeated[0:1]-repeat_target).abs().max()),
                             "actual40_vs_recorded_J_action_max_abs": float((target-sample["J_action"][index:index+1]).abs().max()),
                             "single_action": single[0].cpu().tolist(), "repeat40_action": repeat_target[0].cpu().tolist(),
                             "actual40_action": target[0].cpu().tolist(),
                             "recorded_J_action": sample["J_action"][index].cpu().tolist()})
    return rows


def short_prefix_gate(report, action_tolerance):
    """Require faithful recorded-input replay as well as all fixed physical checks."""
    frozen = report.get("bf16_identical_input_comparisons", [])
    reconstruction = report.get("reconstruction", {})
    replay_ok = (reconstruction.get("maximum_common_recorded_actions") == 100
                 and len(frozen) == 15 and all(
                     row["actual40_vs_recorded_J_action_max_abs"] <= action_tolerance for row in frozen))
    pairs = report.get("rollouts", [])
    rollout_ok = len(pairs) == 3 and all(
        pair.get("new_policyrepeat40", {}).get("comparison_with_original_J_batch40", {}).get("passed", False)
        for pair in pairs)
    return bool(replay_ok and rollout_ok)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i-spec", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--j-root", type=Path, required=True)
    parser.add_argument("--model", choices=("parent", "f0", "f1"), default="parent")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=110)
    args = parser.parse_args(argv)
    if not 0 < args.max_seconds <= 120:
        parser.error("Diagnostic soft budget must be in (0,120] seconds")
    args.output = args.output.resolve(); args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter(); deadline = start+args.max_seconds
    report = {"status": "starting", "model": args.model, "protocol": "new single-physics-policyrepeat40 diagnostic",
              "original_J_results_modified": False, "training_performed": False, "reward_changed": False,
              "termination_changed": False, "tolerances_changed": False, "max_seconds": args.max_seconds,
              "fixed_indices": [0, 20, 39], "rollout_steps_each": 100, "rollouts": [],
              "new_policy40_short_prefix_gate_passed": False, "full_old40_evaluation_run": False}
    receipt = args.output/"precision-receipt.json"
    write(receipt, report)
    def interrupted(signum, frame):
        raise TimeoutError("Finite diagnostic interrupted")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        sys.path.insert(0, str(args.runtime_dir.resolve()))
        import torch
        import evaluate as single_eval
        import sports_runtime as runtime
        from unisupermimic.rl.algos.supermimic import networks
        torch.set_num_threads(4)
        if torch.cuda.device_count() != 1:
            raise ValueError("Exactly one visible CUDA device required; no CPU equivalence claim")
        models = {row["id"]: row for row in json.loads(args.i_spec.read_text())["models"]}
        model = models[args.model]
        if not model["available"] or runtime.sha(model["path"]) != model["sha256"]:
            raise ValueError("Frozen I checkpoint unavailable or changed")
        saved, preparation = single_eval.prepare_checkpoint(torch.load(model["path"], map_location="cpu", weights_only=True), model["feedback"])
        if saved["args"].get("amp") != "bf16":
            raise ValueError("This diagnostic targets the fixed BF16 I/J protocol")
        _, sources, _ = runtime.load_manifest(args.j_root/"old40-manifest.json")
        if len(sources) != 40 or any(s["group"] != "old" for s in sources):
            raise ValueError("Expected frozen old40 ordering")
        j_model = args.j_root/args.model
        summary = json.loads((j_model/"summary.json").read_text())
        if summary["checkpoint_sha256"] != model["sha256"]:
            raise ValueError("J model does not match frozen I model")
        for name in ("sports_runtime.py", "evaluate.py"):
            if runtime.sha(args.runtime_dir/name) != summary["code_sha256"][name]:
                raise ValueError("Runtime differs from byte-frozen J/I: "+name)
        compare_path = args.j_root/"runtime/evaluate_batch.py"
        if runtime.sha(compare_path) != summary["batch_evaluator_sha256"]:
            raise ValueError("Use J's exact original comparison thresholds")
        module_spec = importlib.util.spec_from_file_location("frozen_J_batch", compare_path)
        batch = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(batch)
        original_validation = summary["batch_single_same_backend_validation"]
        action_tolerance = original_validation["clips"][0]["absolute_tolerances"]["action_raw"]
        report.update(status="running", checkpoint_sha256=model["sha256"], checkpoint_preparation=preparation,
                      checkpoint_amp=saved["args"].get("amp"),
                      checkpoint_policy_weight_dtypes=sorted({str(value.dtype) for value in saved["pi"].values() if torch.is_tensor(value)}),
                      frozen_input_replay_action_tolerance=action_tolerance,
                      runtime_sha256={name: runtime.sha(args.runtime_dir/name) for name in ("sports_runtime.py", "evaluate.py")},
                      J_summary_sha256=runtime.sha(j_model/"summary.json"),
                      J_evaluator_sha256=runtime.sha(compare_path), J_original_parity=summary.get("batch_single_same_backend_validation"))
        policy, frozen, reconstruction = freeze_inputs(sources, saved, j_model, summary, runtime, deadline, args.output)
        report["reconstruction"] = reconstruction
        report["bf16_runtime_flags"] = flags(torch, networks)
        report["bf16_identical_input_comparisons"] = frozen_differences(policy, frozen, report["fixed_indices"], torch)
        write(receipt, report)
        # Additional forward-only precision probe; it does not change either
        # native rollout protocol. Restore every flag before new rollouts.
        old_amp = networks.AMP_DTYPE
        old_matmul = torch.backends.cuda.matmul.allow_tf32
        old_cudnn = torch.backends.cudnn.allow_tf32
        old_precision = torch.get_float32_matmul_precision()
        try:
            networks.AMP_DTYPE = None
            torch.set_float32_matmul_precision("highest")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            report["strict_fp32_forward_only_runtime_flags"] = flags(torch, networks)
            report["strict_fp32_identical_input_comparisons"] = frozen_differences(policy, frozen, report["fixed_indices"], torch)
        finally:
            networks.AMP_DTYPE = old_amp
            torch.set_float32_matmul_precision(old_precision)
            torch.backends.cuda.matmul.allow_tf32 = old_matmul
            torch.backends.cudnn.allow_tf32 = old_cudnn
        report["restored_bf16_runtime_flags"] = flags(torch, networks)
        del policy, frozen
        write(receipt, report)
        native_loader = single_eval.load_policy
        for index in report["fixed_indices"]:
            source = sources[index]
            pair = {"env_id": index, "clip": source["path"].stem}
            report["rollouts"].append(pair)
            for protocol in ("original_policy1", "new_policyrepeat40"):
                if time.perf_counter() >= deadline:
                    raise TimeoutError("Deadline before all six fixed diagnostic rollouts")
                target = args.output/protocol/f"clip-{index:04d}-{source['path'].stem}"
                try:
                    if protocol == "new_policyrepeat40":
                        def padded_loader(env, checkpoint, device=None):
                            return Repeat40Policy(native_loader(env, checkpoint, device=device), index)
                        single_eval.load_policy = padded_loader
                    else:
                        single_eval.load_policy = native_loader
                    row = single_eval.evaluate_clip(source, saved, device="cuda", feedback=model["feedback"], seed=0,
                                                    action_mode="raw", output=target, max_steps=100, deadline=deadline)
                finally:
                    single_eval.load_policy = native_loader
                j_trace = j_model/f"clip-{index:04d}-{source['path'].stem}"/"trajectory.npz"
                comparison = {"passed": False, "status": "unavailable_or_incomplete"}
                try:
                    comparison.update(batch.compare_prefix(j_trace, target/"trajectory.npz", 100))
                except Exception as error:
                    comparison["error"] = repr(error)
                pair[protocol] = {"metrics": row, "comparison_with_original_J_batch40": comparison}
                write(receipt, report)
        report["new_policy40_short_prefix_gate_passed"] = short_prefix_gate(report, action_tolerance)
        report["status"] = "new_policy40_short_prefix_pass" if report["new_policy40_short_prefix_gate_passed"] else "new_protocol_failed_fixed_gate"
        report["interpretation_limit"] = "Any pass applies only to the new BF16 policy40 short-prefix route; original policy1 J parity remains as recorded; no full-path equivalence inferred."
    except BaseException as error:
        report.update(status="incomplete_or_error", error_type=type(error).__name__, error=str(error))
    finally:
        report["elapsed_seconds"] = time.perf_counter()-start
        write(receipt, report)
        print("PRECISION_DIAGNOSTIC "+json.dumps({"status": report["status"], "model": report["model"],
              "new_policy40_short_prefix_gate_passed": report["new_policy40_short_prefix_gate_passed"],
              "elapsed_seconds": report["elapsed_seconds"]}), flush=True)
    return 0 if report["new_policy40_short_prefix_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
