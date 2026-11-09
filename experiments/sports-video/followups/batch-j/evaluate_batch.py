"""Fixed env_id/clip native-prefix batch evaluator; no training or auto-reset.

Install beside the FROZEN I evaluate.py/sports_runtime.py, or specify
--runtime-dir. The existing full_trace transport suppresses manager done flags;
the original MDP hard gate and clip boundary are evaluated independently.
Inactive worlds still advance with zero actions and safe reference clocks. Their
later states/rewards are explicitly unscored and never exported as valid traces.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import sys
import time
import numpy as np


class PrefixLedger:
    """Stop each row exactly once, including its failing transition reward."""
    def __init__(self, planned):
        self.planned = np.asarray(planned, dtype=np.int64)
        if np.any(self.planned < 1):
            raise ValueError("Every clip needs at least one control transition")
        self.steps = np.zeros_like(self.planned)
        self.returns = np.zeros(len(self.planned), dtype=np.float64)
        self.hard = np.zeros(len(self.planned), dtype=bool)
        self.alive = np.ones(len(self.planned), dtype=bool)
        self.status = np.full(len(self.planned), "running", dtype=object)

    def advance(self, reward, hard):
        reward, hard = np.asarray(reward), np.asarray(hard, dtype=bool)
        active = self.alive.copy()
        if not np.isfinite(reward[active]).all():
            raise FloatingPointError("Nonfinite scored native reward")
        self.steps[active] += 1
        self.returns[active] += reward[active]
        self.hard |= active & hard
        complete = active & (self.steps == self.planned)
        self.status[complete] = "completed"
        self.status[active & hard] = "native_hard_failure"
        self.alive &= ~(complete | hard)
        return active

    def censor(self, reason):
        self.status[self.alive] = reason


def safe_frames(starts, ends, step, alive):
    # Reward is evaluated at f+1, so inactive input frames must be <= end-1.
    return np.where(alive, np.asarray(starts)+step, np.asarray(ends)-1)


def compare_prefix(batch_path, single_path, steps=100):
    """Predeclared CUDA numerical check, not a claim of full trajectory identity."""
    a, b = np.load(batch_path), np.load(single_path)
    n = min(steps, len(a["reward"]), len(b["reward"]))
    # qdot magnifies tiny contact differences; retain separate physical units.
    tolerances = {"root_position": 1e-3, "quaternion": 1e-3, "velocity": 2e-2,
                  "joint_position": 2e-3, "joint_velocity": 5e-2,
                  "action_raw": 5e-3, "reward": 2e-4, "reference": 2e-6}
    slices = {"root_position": slice(0, 3), "quaternion": slice(3, 7),
              "velocity": slice(7, 13), "joint_position": slice(13, 42),
              "joint_velocity": slice(42, 71)}
    differences = {key: float(np.max(np.abs(a["state"][:n+1, part]-b["state"][:n+1, part])))
                   for key, part in slices.items()}
    for key in ("action_raw", "reward", "reference"):
        size = n+1 if key == "reference" else n
        differences[key] = float(np.max(np.abs(a[key][:size]-b[key][:size]))) if size else None
    def first_event(trace, key):
        indices = np.flatnonzero(trace[key][:steps])
        return int(indices[0]+1) if len(indices) else None
    ha, hb = first_event(a, "native_hard"), first_event(b, "native_hard")
    da, db = first_event(a, "native_done"), first_event(b, "native_done")
    hard_match = ha is None and hb is None or ha is not None and hb is not None and abs(ha-hb) <= 1
    done_match = da is None and db is None or da is not None and db is not None and abs(da-db) <= 1
    expected = min(steps, da or steps, db or steps)
    sufficient = n >= expected and n > 0
    numeric = all(value is not None and value <= tolerances[key] for key, value in differences.items())
    initial_error = float(np.max(np.abs(a["state"][0]-b["state"][0])))
    return {"passed": bool(sufficient and numeric and hard_match and done_match and initial_error <= 1e-5), "compared_transitions": n,
            "initial_state_max_abs": initial_error, "initial_state_tolerance": 1e-5,
            "maximum_absolute_differences": differences, "absolute_tolerances": tolerances,
            "batch_first_hard_step_in_prefix": ha, "single_first_hard_step_in_prefix": hb,
            "hard_event_tolerance_steps": 1, "hard_event_agreement": bool(hard_match),
            "done_event_agreement": bool(done_match), "batch_first_done_step": da, "single_first_done_step": db,
            "full_termination_equivalence_verified": bool(ha is not None and hb is not None and hard_match),
            "scope": "fixed short prefix only; no inference about later chaotic divergence"}


def snapshot_batch(mdp, current):
    import torch
    from unisupermimic.rl.algos.supermimic.geometry import unflatten_state, yaw_mat
    state, frame = unflatten_state(current["s"]), current["f"]
    p, r = mdp.motion.root_p[frame], mdp.motion.root_R[frame]
    bodies = mdp.kin.forward(*state)[0]
    delta = yaw_mat(state[1] @ r.transpose(-1, -2))
    anchor = torch.cat((state[0][..., :2], p[..., 2:3]), -1)
    local_ref = anchor[:, None] + (delta[:, None] @ (mdp.motion.body_p[frame]-p[:, None])[..., None]).squeeze(-1)
    yaw = torch.atan2(state[1][:, 1, 0], state[1][:, 0, 0])-torch.atan2(r[:, 1, 0], r[:, 0, 0])
    values = {"state": current["s"], "reference": torch.cat((p, r.flatten(1), mdp.motion.q[frame]), -1),
              "root_xy_error": (state[0]-p)[:, :2], "root_error_m": (state[0]-p).norm(dim=-1),
              "yaw_error_rad": torch.atan2(torch.sin(yaw), torch.cos(yaw)).abs(),
              "root_orientation_error_rad": torch.acos(((torch.einsum("bij,bij->b", state[1], r)-1)/2).clamp(-1, 1)),
              "body_global_mean_m": (bodies-mdp.motion.body_p[frame]).norm(dim=-1).mean(-1),
              "body_local_mean_m": (bodies-local_ref).norm(dim=-1).mean(-1)}
    return {key: value.detach().cpu().numpy().copy() for key, value in values.items()}


def run_batch(sources, saved, *, device, feedback, output, deadline=None, max_steps=None):
    import torch
    from evaluate import aggregate, summarize_trace, write_json
    from sports_runtime import make_env, sha
    from unisupermimic.rl.algos.supermimic.geometry import unflatten_state
    from unisupermimic.rl.contracts.supermimic import SUPERMIMIC_STATE_KEY as STATE, SUPERMIMIC_TRANSITION_KEY as TRANS
    from unisupermimic.tasks.motion_tracking.g1.supermimic.evaluate import load_policy
    from unisupermimic.tasks.motion_tracking.g1.supermimic.mdp import MAX_STEP_REWARD
    from unisupermimic.tasks.motion_tracking.g1.supermimic.motion import CONTROL_DT
    output.mkdir(parents=True, exist_ok=False)
    started, env, ledger, error = time.perf_counter(), None, None, None
    states, transitions = [], []
    report = {"status": "running", "device": device, "backend": "mujoco" if device == "cpu" else "mjwarp",
              "num_envs": len(sources), "seed": 0, "feedback": feedback, "action_mode": "raw",
              "auto_reset": False, "post_initialization_resets": 0, "reward_changed": False,
              "native_scoring_termination_changed": False, "manager_termination_transport": "suppressed_full_trace",
              "native_gate": "original mdp.termination(s1, prior_frame+1) OR original clip_end",
              "inactive_worlds": "still simulated with zero action and reference end-1 input; no resets; all post-stop samples discarded",
              "batch_single_cuda_validation": "not_run", "strict_contact_quality_acceptance": None,
              "full_protocol_evaluation_complete": False, "clips": []}
    write_json(output/"summary.json", report)
    try:
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError("Deadline before batch initialization")
        env = make_env([s["path"] for s in sources], device, len(sources), 0, feedback,
                       sources[0]["fps"], evaluation=True, full_trace=True)
        command = env._supermimic_command
        mdp, motion = env.supermimic_mdp, env.supermimic_mdp.motion
        if motion.names != [s["path"].stem for s in sources]:
            raise ValueError("MotionLib reordered fixed env_id-to-clip allocation")
        planned = (motion.clip_len-1).cpu().numpy()
        ledger = PrefixLedger(planned)
        starts, ends = motion.clip_start.cpu().numpy(), motion.clip_end.cpu().numpy()
        report["env_id_mapping"] = [{"env_id": i, "clip": s["path"].stem, "source_sha256": s["sha256"],
                                     "library_start": int(starts[i]), "library_end": int(ends[i]),
                                     "local_initial_frame": 0} for i, s in enumerate(sources)]
        policy = load_policy(env, saved, device=device)
        _, info = env.reset_to_frames(motion.clip_start.clone())
        current, history = info[STATE], None
        if not torch.equal(current["f"], motion.clip_start):
            raise ValueError("Native reset did not start every clip at its frame zero")
        report["initial_context_max_abs"] = float(current["xi"].abs().max())
        def forbidden_reset(*args, **kwargs):
            report["post_initialization_resets"] += 1
            raise RuntimeError("Batch evaluation forbids resets after initialization")
        env.reset = forbidden_reset
        states.append(snapshot_batch(mdp, current))
        with torch.inference_mode():
            while ledger.alive.any():
                step = len(transitions)
                if max_steps is not None and step >= max_steps:
                    ledger.censor("truncated_engineering_steps"); break
                if deadline is not None and time.perf_counter() >= deadline:
                    ledger.censor("truncated_engineering_deadline"); break
                active = torch.as_tensor(ledger.alive, device=device)
                # Only inactive reference clocks are pinned. Active clocks must
                # remain exactly native start+step; physical state is never reset.
                expected = torch.as_tensor(starts+step, device=device)
                if not torch.equal(current["f"][active], expected[active]):
                    raise RuntimeError("An active reference clock skipped or reset")
                clocks = torch.as_tensor(safe_frames(starts, ends, step, ledger.alive), device=device)
                command.f.copy_(clocks)
                current["f"] = clocks.clone()
                if not bool(torch.isfinite(current["s"]).all()):
                    raise FloatingPointError("Nonfinite world; abort batch without inventing future samples")
                obs = mdp.obs(unflatten_state(current["s"]), current["a_prev"], current["xi"], clocks)
                hist = int(saved["args"]["hist"])
                history = obs[:, None].repeat(1, hist, 1) if history is None else torch.cat((history[:, 1:], obs[:, None]), 1)
                raw = policy(history, mdp.goal(clocks, int(saved["args"]["goal_frames"])))
                if not bool(torch.isfinite(raw[active]).all()):
                    raise FloatingPointError("Nonfinite active raw policy action")
                action = torch.where(active[:, None], raw, torch.zeros_like(raw))
                result = env.step(action)
                tr, current = result.info[TRANS], result.info[STATE]
                # Same arguments as manager_terms.supermimic_termination; no
                # threshold copies and no replacement reward computation.
                hard = mdp.termination(unflatten_state(tr["s1"]), tr["f"]+1)
                clip_end = tr["f"]+1 >= motion.clip_end[motion.clip_id[tr["f"]]]
                timeout = tr["ep_step"]+1 >= command.max_episode_steps
                if bool(timeout[active].any()):
                    raise RuntimeError("Unexpected native timeout in fixed reference evaluation")
                if bool(tr["term"].any()) or bool(tr["done"].any()):
                    raise RuntimeError("full_trace transport failed to suppress manager done")
                nxt = snapshot_batch(mdp, current)
                if any(not np.isfinite(value[ledger.alive]).all() for value in nxt.values()):
                    raise FloatingPointError("Nonfinite scored post-step state")
                values = {"action_raw": raw.cpu().numpy().copy(), "action_applied": action.cpu().numpy().copy(),
                          "reward": tr["r"].cpu().numpy().copy(), "native_hard": hard.cpu().numpy().copy(),
                          "native_done": (hard | clip_end | timeout).cpu().numpy().copy()}
                ledger.advance(values["reward"], values["native_hard"])
                transitions.append(values); states.append(nxt)
                if step % 250 == 0:
                    print("BATCH_PROGRESS "+json.dumps({"step": step+1, "active": int(ledger.alive.sum()),
                           "hard_failures": int(ledger.hard.sum())}), flush=True)
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        if ledger is not None:
            ledger.censor("truncated_engineering_deadline" if isinstance(exc, TimeoutError) else "error")
    finally:
        for i, source in enumerate(sources):
            count = int(ledger.steps[i]) if ledger is not None else 0
            planned_i = int(ledger.planned[i]) if ledger is not None else None
            hard = bool(ledger.hard[i]) if ledger is not None else False
            complete = planned_i is not None and count == planned_i
            trace = {key: [value[key][i] for value in states[:count+1]] for key in states[0]} if states else {"state": []}
            if transitions:
                trace.update({key: [value[key][i] for value in transitions[:count]] for key in transitions[0]})
            row = {"env_id": i, "clip": source["path"].stem, "path": str(source["path"]), "group": source["group"],
                   "source_sha256": source["sha256"], "source_fps": source["fps"],
                   "status": str(ledger.status[i]) if ledger is not None else "not_run_batch_initialization_error",
                   "planned_steps": planned_i, "observed_steps": count, "observed_seconds": count*CONTROL_DT,
                   "native_hard_failure": hard, "first_native_hard_failure_s": count*CONTROL_DT if hard else None,
                   "full_reference_completed": complete, "completed_without_native_failure": complete and not hard,
                   "survival_seconds": count*CONTROL_DT, "survival_is_censored_lower_bound": not hard and not complete,
                   "reward_sum_until_stop": float(ledger.returns[i]) if ledger is not None else 0.,
                   "normalization_denominator": planned_i*MAX_STEP_REWARD if planned_i else None,
                   "norm_return_full_reference": float(ledger.returns[i]/(planned_i*MAX_STEP_REWARD)) if planned_i else None,
                   "post_initialization_resets": 0, "strict_contact_quality_acceptance": None,
                   "post_stop_backend_steps_discarded": len(transitions)-count,
                   "remaining_reference_steps_without_valid_rollout": planned_i-count if planned_i else None,
                   "post_stop_samples_exported": 0}
            row.update(summarize_trace(trace, row))
            if error and (ledger is None or ledger.alive[i]):
                row["batch_error"] = error
            target = output/f"clip-{i:04d}-{source['path'].stem}"
            target.mkdir()
            np.savez_compressed(target/"trajectory.npz", frame_index=np.arange(len(trace["state"])),
                                **{key: np.asarray(value) for key, value in trace.items()})
            row["trajectory_sha256"] = sha(target/"trajectory.npz")
            write_json(target/"metrics.json", row)
            report["clips"].append(row)
        report.update(status="batch_native_prefixes_recorded_pending_validation" if error is None else "batch_error_or_budget",
                      aggregate=aggregate(report["clips"]), wall_seconds=time.perf_counter()-started,
                      backend_steps=len(transitions), batch_error=error)
        write_json(output/"summary.json", report)
        if env is not None:
            try:
                env.close()
            finally:
                env.backend.close()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feedback", choices=("F0", "F1"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--expect-count", type=int, default=40)
    parser.add_argument("--max-seconds", type=float, default=500)
    parser.add_argument("--validation-steps", type=int, default=100)
    args = parser.parse_args(argv)
    if args.max_seconds <= 0 or args.validation_steps < 1:
        parser.error("Positive finite budgets are required")
    sys.path.insert(0, str(args.runtime_dir.resolve()))
    def preserve_on_signal(signum, frame):
        raise TimeoutError("External finite-budget signal interrupted evaluator")
    signal.signal(signal.SIGTERM, preserve_on_signal)
    import torch
    from evaluate import evaluate_clip, prepare_checkpoint, write_json
    from sports_runtime import load_manifest, sha
    _, sources, _ = load_manifest(args.manifest)
    if len(sources) != args.expect_count or any(s["group"] != "old" for s in sources):
        raise ValueError("Require exactly the frozen old-only manifest")
    if args.device == "cuda" and torch.cuda.device_count() != 1:
        raise ValueError("Expose exactly one CUDA device")
    torch.set_num_threads(4 if args.device == "cuda" else 2)
    saved, preparation = prepare_checkpoint(torch.load(args.checkpoint, map_location="cpu", weights_only=True), args.feedback)
    deadline = time.perf_counter()+args.max_seconds
    # Selection is fixed by manifest position, independent of result or duration.
    selected = sorted(set((0, len(sources)//2, len(sources)-1)))
    report = run_batch(sources, saved, device=args.device, feedback=args.feedback, output=args.output,
                       deadline=deadline-min(90., args.max_seconds*.3))
    validation = {"status": "not_run", "device": args.device, "backend": report["backend"],
                  "selection": "manifest indices 0, floor(N/2), N-1, fixed before scores",
                  "indices": selected, "maximum_steps": args.validation_steps, "clips": []}
    for index in selected:
        source = sources[index]
        target = args.output/f"single-check-{index:04d}"
        row = evaluate_clip(source, saved, device=args.device, feedback=args.feedback, seed=0, action_mode="raw",
                            output=target, max_steps=args.validation_steps, deadline=deadline)
        comparison = {"passed": False, "status": "unavailable_or_incomplete", "single_row": row}
        try:
            batch_path = args.output/f"clip-{index:04d}-{source['path'].stem}"/"trajectory.npz"
            comparison.update(compare_prefix(batch_path, target/"trajectory.npz", args.validation_steps))
        except Exception as exc:
            comparison["comparison_error"] = repr(exc)
        comparison["env_id"] = index
        validation["clips"].append(comparison)
    validation["passed"] = all(item["passed"] for item in validation["clips"])
    validation["status"] = "short_prefix_pass" if validation["passed"] else "failed_or_incomplete"
    report.update(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=sha(args.checkpoint),
                  checkpoint_preparation=preparation, manifest_sha256=sha(args.manifest),
                  batch_single_cuda_validation=validation if args.device == "cuda" else "not_run_cpu_only",
                  batch_single_same_backend_validation=validation,
                  full_protocol_evaluation_complete=bool(validation["passed"] and
                      report["aggregate"]["native_terminal_outcomes"] == len(sources)),
                  scope="old40 diagnostic native rollout; short-prefix numerical check is not full-path equivalence or sports acceptance",
                  code_sha256={name: sha(args.runtime_dir/name) for name in ("sports_runtime.py", "evaluate.py")},
                  batch_evaluator_sha256=sha(__file__))
    write_json(args.output/"summary.json", report)
    print("BATCH_EVALUATION_RECEIPT "+json.dumps({"output": str(args.output), "aggregate": report["aggregate"],
          "prefix_validation": validation["status"], "device": args.device}), flush=True)
    return 0 if report["full_protocol_evaluation_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
