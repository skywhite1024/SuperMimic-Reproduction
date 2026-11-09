"""Finite sports-reference continuation with a common 99D parent and F0/F1 control."""
from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

import numpy as np
import torch
from sports_runtime import (ROOT, STUDY, PARENT_SHA256, FixedCollection, extra_columns,
    freeze_normalizers, initial_policy_identity, install_f0_column_freeze,
    load_manifest, make_env, migrate_checkpoint, sha)
from unisupermimic.rl.algos.supermimic.learner import SuperMimicConfig, SuperMimicLearner, _LOG_FIELDS

sys.path.insert(0, str(ROOT / "acro-library-20261027-b/scripts"))
from balanced_replay import BalancedReplay


class SportsReplay(BalancedReplay):
    def __init__(self, *args, require_all_clips=True, **kwargs):
        self.require_all_clips = require_all_clips
        super().__init__(*args, **kwargs)

    def probabilities(self, counts):
        if self.require_all_clips and bool((counts == 0).any()):
            missing = [name for name, count in zip(self.names, counts.tolist()) if not count]
            raise RuntimeError(f"Replay lacks eligible starts for frozen clips: {missing}")
        return super().probabilities(counts)


def digest(module):
    result = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        result.update(key.encode()); result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--parent-sha256", default=PARENT_SHA256)
    parser.add_argument("--feedback", choices=["F0", "F1"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--phase", choices=["engineering", "pilot", "main"], default="pilot")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--wm-warmup", type=int)
    parser.add_argument("--max-seconds", type=int)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--old-fraction", type=float, default=.5)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    engineering = args.phase == "engineering"
    if args.device == "cpu" and not engineering:
        raise ValueError("CPU runs are engineering checks, not GPU experiment results")
    steps = args.steps if args.steps is not None else (3 if engineering else 100 if args.phase == "pilot" else 2000)
    warmup = args.wm_warmup if args.wm_warmup is not None else (1 if engineering else 20 if args.phase == "pilot" else 200)
    maximum = args.max_seconds if args.max_seconds is not None else (600 if engineering else 1800 if args.phase == "pilot" else 7200)
    count = args.num_envs if args.num_envs is not None else (8 if engineering else 512)
    if not 0 <= warmup < steps or maximum <= 0 or count <= 0 or not 0 < args.old_fraction < 1:
        raise ValueError("Invalid finite update/warmup/time/environment/group budget")
    output = (STUDY / "runs" / args.label).resolve()
    if not output.is_relative_to(STUDY / "runs") or output.exists():
        raise ValueError("Use a fresh run label inside the study runs directory")
    parent_path, manifest_path = args.parent.resolve(), args.manifest.resolve()
    if sha(parent_path) != args.parent_sha256:
        raise ValueError("Common parent SHA256 mismatch")
    manifest, rows, fps = load_manifest(manifest_path)
    files = [row["path"] for row in rows]
    new_names = {row["path"].stem for row in rows if row["group"] == "new"}
    if not new_names or len(new_names) == len(files):
        raise ValueError("Frozen training manifest must include both new and retained clips")
    torch.set_num_threads(2 if args.device == "cpu" else 4)
    if args.device == "cuda":
        if torch.cuda.device_count() != 1:
            raise ValueError("Expose exactly one GPU per independent run via CUDA_VISIBLE_DEVICES")
        torch.cuda.reset_peak_memory_stats()
    parent = torch.load(parent_path, map_location="cpu", weights_only=True)
    initial = migrate_checkpoint(parent, args.feedback)
    identity = initial_policy_identity(parent, initial)
    parent_iteration = int(parent["iteration"])
    config = {field.name: parent["args"][field.name] for field in fields(SuperMimicConfig) if field.name in parent["args"]}
    for key in ("wm_hidden", "policy_hidden", "critic_hidden", "test_hidden_dims"):
        if config.get(key) is not None:
            config[key] = tuple(config[key])
    config.update(iterations=parent_iteration+steps, wm_only_iters=parent_iteration+warmup,
                  buffer_steps=80 if engineering else 300, wm_batches=1 if engineering else 4,
                  wm_batch=4 if engineering else 512, pi_batch=4 if engineering else 512,
                  critic_epochs=1 if engineering else 2, critic_minibatches=1 if engineering else 2,
                  lr=5e-5, ckpt_every=500, log_every=1 if engineering else 25,
                  seed=args.seed, compile=False, amp="none" if args.device == "cpu" else "bf16")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    collection = FixedCollection(new_names, args.old_fraction)
    output.mkdir(parents=True)
    initial_path = output / "initial-full.pt"
    torch.save(initial, initial_path)
    variant = {**initial["variant"], "phase": args.phase, "seed": args.seed,
               "reference_variant": manifest.get("reference_variant", "unspecified"),
               "parent_sha256": args.parent_sha256, "manifest_sha256": sha(manifest_path),
               "old_replay_fraction": args.old_fraction, "source_fps": fps}
    run = {"variant": variant, "initial_policy_identity": identity, "training_config": config,
           "num_envs": count, "maximum_seconds": maximum, "planned_updates": steps,
           "world_model_warmup_updates": warmup,
           "evaluation_config": {"feedback": args.feedback, "source_fps": fps,
               "backend": "mujoco" if args.device == "cpu" else "mjwarp",
               "observation_dim": 101, "history_frames": config["hist"],
               "goal_frames": config["goal_frames"], "reference_yaw_xy": True,
               "raw_policy_action_output": True, "training_noise_action_clip": [-20, 20],
               "auto_reset": False, "score_native_first_failure": True,
               "max_step_reward": .1, "evaluation_reference_not_selected_by_this_run": True},
           "source_sha256": {name: sha(STUDY / name) for name in ("sports_runtime.py", "train.py")},
           "motions": [{**row, "path": str(row["path"])} for row in rows],
           "torch_version": str(torch.__version__), "device": args.device,
           "cuda_device_name": torch.cuda.get_device_name(0) if args.device == "cuda" else None}
    for key, command in (("source_commit", ["git", "rev-parse", "HEAD"]),
                         ("source_status", ["git", "status", "--porcelain"])):
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        run[key] = result.stdout.strip() if result.returncode == 0 else None
    (output / "run.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    started, env, learner, deadline_hit = time.perf_counter(), None, None, False
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    try:
        env = make_env(files, args.device, count, args.seed, args.feedback, fps, sampler=collection)
        learner = SuperMimicLearner(env, SuperMimicConfig(**config), output, device=args.device,
                                   backend="mujoco" if args.device == "cpu" else "mjwarp")
        learner.load(initial_path)
        freeze_normalizers(learner)
        freeze_hooks = install_f0_column_freeze(learner.pi, learner.opt_pi) if args.feedback == "F0" else ()
        replay = SportsReplay(learner.buffer, learner.mdp, learner.cfg.horizon, new_names, {},
                              new_fraction=1-args.old_fraction, require_all_clips=not engineering)
        replay.install(learner)
        native_checkpoint = learner._checkpoint
        learner._checkpoint = lambda include_optimizers: {**native_checkpoint(include_optimizers), "variant": variant}
        before = {name: digest(getattr(learner, name)) for name in ("wm", "pi", "critic")}
        initial_sim_steps = learner.sim_steps
        native_collect = learner.collect
        def timed_collect(n):
            nonlocal deadline_hit
            if time.perf_counter() - started > maximum:
                deadline_hit = True
                raise TimeoutError("Finite sports run deadline reached")
            return native_collect(n)
        learner.collect = timed_collect
        def interrupted(signum, frame):
            raise InterruptedError(f"Received signal {signum}")
        signal.signal(signal.SIGTERM, interrupted)
        # The native resumed-run logger only adds a header for iteration zero.
        (output / "train_log.csv").write_text(",".join(_LOG_FIELDS) + "\n", encoding="utf-8")
        learner.train()
        destination = learner.save(output / "final-full.pt")
        saved = torch.load(destination, map_location="cpu", weights_only=True)
        normalizers_same = all(torch.equal(initial[name][key], saved[name][key])
            for name in ("wm", "pi", "critic") for key in initial[name]
            if "_norm." in key or key.startswith("norm."))
        if not normalizers_same:
            raise AssertionError("Frozen normalizer tensors changed")
        after = {name: digest(getattr(learner, name)) for name in before}
        for name in before:
            if before[name] == after[name]:
                raise AssertionError(f"No verified {name} update")
            for key, value in getattr(learner, name).state_dict().items():
                if not torch.isfinite(value).all() or not torch.equal(value.detach().cpu(), saved[name][key]):
                    raise AssertionError(f"Invalid checkpoint readback: {name}/{key}")
        cols = extra_columns(config["hist"], config["z_dim"])
        extra_max = float(saved["pi"]["dec.0.weight"][:, cols].abs().max())
        if args.feedback == "F0" and extra_max != 0:
            raise AssertionError("F0 new decoder columns changed")
        proof = {"status": "real_updates_and_checkpoint_readback_verified", "variant": variant,
                 "engineering_only": engineering, "physical_acceptance_inferred": False,
                 "new_updates": learner.iteration-parent_iteration,
                 "new_sim_transitions": learner.sim_steps-initial_sim_steps,
                 "wm_updates": steps*config["wm_batches"], "policy_updates": steps-warmup,
                 "wall_seconds": time.perf_counter()-started, "normalizer_tensors_unchanged": normalizers_same,
                 "initial_policy_identity": identity, "new_decoder_column_max_abs": extra_max,
                 "before": before, "after": after, "collection": collection.receipt(), "replay": replay.receipt(),
                 "checkpoint_sha256": sha(destination),
                 "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved() if args.device == "cuda" else None}
        (output / "proof.json").write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
        print("SPORTS_TRAIN_VERIFIED " + json.dumps({key: proof[key] for key in
              ("status", "variant", "new_updates", "wall_seconds", "new_decoder_column_max_abs")}), flush=True)
        return proof
    except BaseException as error:
        if learner is not None:
            learner.save(output / "interrupted-full.pt")
        failure = {"error": str(error), "new_updates": learner.iteration-parent_iteration if learner else 0,
                   "deadline_hit": deadline_hit, "wall_seconds": time.perf_counter()-started}
        (output / "failure.json").write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        if env is not None:
            try:
                env.close()
            finally:
                env.backend.close()


if __name__ == "__main__":
    main()
