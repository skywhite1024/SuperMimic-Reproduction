"""Separate deterministic first-frame full-clip evaluation with explicit denominators."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from unisupermimic.tasks.motion_tracking.g1.supermimic.evaluate import make_eval_env, evaluate
from unisupermimic.tasks.motion_tracking.g1.supermimic.mdp import MAX_STEP_REWARD


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(0)
    started = time.monotonic()
    saved = torch.load(args.checkpoint, map_location="cuda:0", weights_only=True)
    env = make_eval_env(args.checkpoint.parent, backend="mjwarp")
    try:
        motion = env.supermimic_mdp.motion
        assert len(motion.names) == env.num_envs == 40
        result = evaluate(env, saved)
        for index, name in enumerate(motion.names):
            row = result["per_clip"][name]
            full_steps = int(motion.clip_len[index]) - 1
            row["full_clip_steps"] = full_steps
            row["normalization_denominator"] = float(full_steps * MAX_STEP_REWARD)
            row["reward_sum"] = row["norm_return"] * row["normalization_denominator"]
            assert 0 < row["steps"] <= full_steps
            if not row["failed"]:
                assert row["steps"] == full_steps, (name, row)
        macro = sum(x["norm_return"] for x in result["per_clip"].values()) / 40
        weighted = sum(x["reward_sum"] for x in result["per_clip"].values()) / sum(
            x["normalization_denominator"] for x in result["per_clip"].values())
        assert abs(macro - result["norm_return"]) < 1e-5
        assert abs(weighted - result["norm_return_framew"]) < 1e-5
        result.update(checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                      wall_seconds=time.monotonic()-started, backend="mjwarp", seed=0,
                      first_frame=True, deterministic_policy=True, observation_noise=False,
                      action_noise=False, domain_randomization=False, friction=1.0,
                      normalization="full resampled clip length minus one, including early failure",
                      community_baseline=True, strict_paper_reproduction=False)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print("SUPERMIMIC_EVALUATION " + json.dumps(result), flush=True)
    finally:
        env.close()
        env.backend.close()


if __name__ == "__main__":
    main()
