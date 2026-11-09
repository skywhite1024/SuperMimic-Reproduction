"""Controlled 101D F0/F1 interfaces; native reward, termination and dynamics.

F0 appends two zeros. F1 appends (reference_root - robot_root) XY in the
*reference* yaw frame, in metres, without clipping or added noise. Both use
identical policy shapes. The robot state, world model and critic stay unchanged.
"""
from __future__ import annotations

import copy
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).resolve().parent
UNI = Path(os.environ.get("SUPERMIMIC_UNI", str(ROOT / "implementation/uni"))).resolve()
sys.path.insert(0, str(UNI / "src"))
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MUJOCO_GL", "disable")

import torch
from unisupermimic.rl.algos.supermimic import networks
from unisupermimic.rl.algos.supermimic.geometry import yaw_mat
from unisupermimic.rl.algos.supermimic.networks import Policy
from unisupermimic.tasks.motion_tracking.g1.supermimic.mdp import MDP

PARENT_SHA256 = "0afcca9af97bd9db1f2dd083875898f962acab0b0a70c7313b221e575b70b05b"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_feedback(feedback):
    if feedback not in ("F0", "F1"):
        raise ValueError("feedback must be F0 or F1")
    return feedback


class XYObservationMDP(MDP):
    obs_dim = 101

    def __init__(self, *args, feedback="F0", sampler=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.feedback = check_feedback(feedback)
        self.obs_noise = torch.cat((self.obs_noise, self.obs_noise.new_zeros(2)))
        self.obs_scale = torch.cat((self.obs_scale, self.obs_scale.new_ones(2)))
        if sampler is not None:
            self.motion._sports_sampler = sampler

    def obs(self, state, previous_action, context, frames):
        base = super().obs(state, previous_action, context, frames)
        if self.feedback == "F0":
            xy = base.new_zeros((*base.shape[:-1], 2))
        else:
            heading_t = yaw_mat(self.motion.root_R[frames]).transpose(-1, -2)
            delta = self.motion.root_p[frames] - state[0]
            xy = (heading_t @ delta.unsqueeze(-1)).squeeze(-1)[..., :2]
        return torch.cat((base, xy), -1)


def policy_from_checkpoint(checkpoint, obs_dim=None):
    args = checkpoint["args"]
    obs_dim = obs_dim or checkpoint["pi"]["obs_norm.mean"].numel()
    hidden = args.get("test_hidden_dims") or args["policy_hidden"]
    return Policy(obs_dim, 73 * int(args["goal_frames"]), hist=int(args["hist"]),
                  z_dim=int(args["z_dim"]), hidden=tuple(hidden))


def extra_columns(hist, z_dim, *, device=None):
    return (torch.arange(hist, device=device)[:, None] * 101 + z_dim
            + torch.tensor([99, 100], device=device)).flatten()


def expand_decoder(value, hist, z_dim):
    if value.ndim != 2 or value.shape[1] != z_dim + hist * 99:
        raise ValueError("Expected the parent's 99D history decoder matrix")
    out = value.new_zeros(value.shape[0], z_dim + hist * 101)
    out[:, :z_dim] = value[:, :z_dim]
    out[:, z_dim:].reshape(value.shape[0], hist, 101)[:, :, :99] = (
        value[:, z_dim:].reshape(value.shape[0], hist, 99))
    return out


def migrate_checkpoint(parent, feedback):
    """Copy a full 99D parent, preserving optimizer order/counters and WM/critic.

    The return value is in memory; no source or destination file is overwritten.
    New decoder weights and their Adam moments are zero for both F conditions.
    """
    check_feedback(feedback)
    required = {"wm", "pi", "critic", "opt_wm", "opt_pi", "opt_v", "args", "iteration"}
    if not required.issubset(parent):
        raise ValueError(f"A full training parent is required; missing {sorted(required - parent.keys())}")
    if parent["pi"]["obs_norm.mean"].numel() != 99:
        raise ValueError("Migration requires a 99D common parent, not a previous 101D candidate")
    if (parent["wm"]["in_norm.mean"].numel(), parent["wm"]["out_norm.mean"].numel()) != (133, 70):
        raise ValueError("Expected unchanged G1 world-model contract (133 input, 70 increment)")
    args = parent["args"]
    hist, z_dim = int(args["hist"]), int(args["z_dim"])
    original = policy_from_checkpoint(parent)
    original.load_state_dict(parent["pi"])
    result = copy.deepcopy(parent)
    result["pi"]["dec.0.weight"] = expand_decoder(parent["pi"]["dec.0.weight"], hist, z_dim)
    for name, fill in (("mean", 0), ("var", 1)):
        value = parent["pi"][f"obs_norm.{name}"]
        result["pi"][f"obs_norm.{name}"] = torch.cat((value, value.new_full((2,), fill)))
    names = [name for name, _ in original.named_parameters()]
    ids = [i for group in parent["opt_pi"]["param_groups"] for i in group["params"]]
    if len(ids) != len(names) or len(set(ids)) != len(ids):
        raise ValueError("Cannot map parent policy optimizer parameter IDs unambiguously")
    decoder_id = ids[names.index("dec.0.weight")]
    for name, value in result["opt_pi"]["state"].get(decoder_id, {}).items():
        if torch.is_tensor(value) and value.ndim > 0:
            if value.shape != parent["pi"]["dec.0.weight"].shape:
                raise ValueError(f"Unexpected Adam decoder slot shape: {name}")
            result["opt_pi"]["state"][decoder_id][name] = expand_decoder(value, hist, z_dim)
    result["variant"] = {
        "study": "sports-video-xy", "feedback": feedback, "observation_dim": 101,
        "parent_observation_dim": 99, "xy_frame": "reference_yaw", "xy_units": "metres",
        "xy_clip": None, "xy_noise": 0, "new_decoder_columns_zero_initialized": True,
        "new_decoder_columns_frozen": feedback == "F0", "normalizers_fixed": True,
        "reward_changed": False, "termination_changed": False,
        "world_model_dimensions": [133, 70], "critic_changed": False,
        "exact_simulator_rng_replay_restore": False,
    }
    # Shape/load validation is independent of the initial-output numerical test.
    policy_from_checkpoint(result).load_state_dict(result["pi"])
    return result


def initial_policy_identity(parent, migrated, seed=90210, samples=8):
    """Check both zero and nonzero new observations against the original policy."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        old, new = policy_from_checkpoint(parent), policy_from_checkpoint(migrated)
        old.load_state_dict(parent["pi"]); new.load_state_dict(migrated["pi"])
        old.eval(); new.eval()
        hist, goal_dim = old.hist, old.goal_norm.mean.numel()
        obs = torch.randn(samples, hist, 99) * .1
        goal = torch.randn(samples, goal_dim) * .1
        previous_amp = networks.AMP_DTYPE
        try:
            networks.AMP_DTYPE = None
            with torch.inference_mode():
                expected = old(obs, goal)
                errors = []
                for xy in (torch.zeros(samples, hist, 2), torch.randn(samples, hist, 2)):
                    errors.append(float((new(torch.cat((obs, xy), -1), goal) - expected).abs().max()))
        finally:
            networks.AMP_DTYPE = previous_amp
    maximum = max(errors)
    if maximum > 2e-5:
        raise AssertionError(f"Initial policy identity failed: max absolute error {maximum}")
    return {"max_abs_error": maximum, "absolute_tolerance": 2e-5,
            "nonzero_xy_tested": True, "exact_bitwise_equality_claimed": False}


def freeze_normalizers(learner):
    for norm in (learner.pi.obs_norm, learner.pi.goal_norm, learner.wm.in_norm,
                 learner.wm.out_norm, learner.critic.norm):
        norm.max_count = 0


def install_f0_column_freeze(policy, optimizer):
    """Mask new-column gradients and clear their weights/moments after each Adam step."""
    parameter = policy.dec[0].weight
    cols = extra_columns(policy.hist, policy.enc[-1].out_features, device=parameter.device)
    mask = torch.ones_like(parameter)
    mask[:, cols] = 0
    gradient_hook = parameter.register_hook(lambda grad: grad * mask)

    @torch.no_grad()
    def enforce(*_):
        parameter[:, cols] = 0
        for value in optimizer.state.get(parameter, {}).values():
            if torch.is_tensor(value) and value.shape == parameter.shape:
                value[:, cols] = 0

    enforce()
    optimizer_hook = optimizer.register_step_post_hook(enforce)
    return gradient_hook, optimizer_hook


def load_manifest(path):
    """Read a frozen, uniform-FPS manifest; paths are relative to its directory."""
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    clips = data.get("clips", [])
    if not clips:
        raise ValueError("Manifest needs nonempty clips with path, fps, group and sha256")
    rows, names, rates = [], set(), set()
    for source in clips:
        file = (path.parent / source["path"]).resolve()
        fps = float(source.get("fps", data.get("fps", 0)))
        if not math.isfinite(fps) or fps <= 0 or source["group"] not in ("old", "new"):
            raise ValueError("Each clip needs finite positive fps and group old/new")
        if file.stem in names:
            raise ValueError(f"MotionLib requires unambiguous clip names: {file.stem}")
        if sha(file) != source["sha256"]:
            raise ValueError(f"Motion hash mismatch: {file}")
        rates.add(fps); names.add(file.stem)
        rows.append({**source, "path": file, "fps": fps})
    if len(rates) != 1:
        raise ValueError("Mixed source FPS is unsupported: explicitly resample every clip to a common rate")
    return data, rows, rates.pop()


class FixedCollection:
    """Fixed group mass, uniform clips and frames; no performance-adaptive mixing."""
    def __init__(self, new_names, old_fraction=.5):
        if not 0 < old_fraction < 1:
            raise ValueError("Both retained and new groups require positive mass")
        self.new_names, self.old_fraction = set(new_names), old_fraction
        self.counts = None
        self.calls = 0

    def __call__(self, motion, bin_G, bin_w_avg, gamma, num_samples, rng):
        new = torch.tensor([name in self.new_names for name in motion.names], device=motion.device)
        if not bool(new.any()) or not bool((~new).any()):
            raise ValueError("Both retained and new groups are required")
        mass = torch.where(new, (1-self.old_fraction)/new.sum(), self.old_fraction/(~new).sum()).float()
        clips = rng.choice(len(motion.names), (num_samples,), p=mass)
        # Cover every clip once in the first reset when the batch can accommodate it.
        coverage = len(motion.names) if self.calls == 0 and num_samples >= len(motion.names) else 0
        if coverage:
            clips[:coverage] = torch.arange(coverage, device=motion.device)[rng.random((coverage,)).argsort()]
        lengths = motion.clip_end[clips] - motion.clip_start[clips]
        frames = motion.clip_start[clips] + (rng.random((num_samples,)) * lengths.float()).long()
        if coverage:
            frames[:coverage] = motion.clip_start[clips[:coverage]]
        if self.counts is None:
            self.counts = torch.zeros(len(motion.names), dtype=torch.long, device=motion.device)
            self.names = list(motion.names)
        self.counts += torch.bincount(clips, minlength=len(motion.names))
        self.calls += 1
        return frames, motion.frame_bin[frames]

    def receipt(self):
        return {"old_group_mass": self.old_fraction, "calls": self.calls,
                "first_reset_coverage": "one start per clip if batch permits; reported separately from random group mass",
                "clip_reset_samples": dict(zip(self.names, self.counts.cpu().tolist())) if self.counts is not None else {}}


def make_env(files, device="cpu", count=1, seed=0, feedback="F0", src_fps=30.,
             evaluation=False, sampler=None, full_trace=False):
    """Create a native environment, changing only its observation implementation.

    full_trace only changes evaluation auto-termination, never training rules.
    A full-trace evaluator must still score the native first-failure gate.
    """
    from hydra import compose, initialize_config_dir
    from unilab.training import ensure_registries
    from unilab.base.config_adapter import create_env
    from unisupermimic.scripts.train_supermimic import build_env_override
    from unisupermimic.tasks.motion_tracking.g1.supermimic import manager_terms
    check_feedback(feedback)
    files = [str(Path(file).resolve()) for file in files]
    if not files or not math.isfinite(src_fps) or src_fps <= 0:
        raise ValueError("Valid motion files and source FPS are required")
    if full_trace and not evaluation:
        raise ValueError("full_trace is evaluation-only")
    # Dispatch by the particular motion library so another environment does not
    # inherit this environment's sampler on later resets.
    if not hasattr(manager_terms, "_sports_native_sampler"):
        manager_terms._sports_native_sampler = manager_terms.sample_start_frames
        def dispatch(motion, *args):
            selected = getattr(motion, "_sports_sampler", manager_terms._sports_native_sampler)
            return selected(motion, *args)
        manager_terms.sample_start_frames = dispatch
    previous_mdp, previous_cwd = manager_terms.MDP, Path.cwd()
    try:
        manager_terms.MDP = partial(XYObservationMDP, feedback=feedback, sampler=sampler)
        os.chdir(UNI)
        backend = "mujoco" if str(device) == "cpu" else "mjwarp"
        with initialize_config_dir(version_base="1.3", config_dir=str(UNI / "src/unisupermimic/conf/supermimic")):
            cfg = compose(config_name="config", overrides=[f"task=g1_super_mimic/{backend}"])
        cfg.env.commands.motion.motion_files = files
        cfg.env.commands.motion.src_fps = src_fps
        cfg.env.seed = seed
        if evaluation:
            cfg.env.commands.motion.max_episode_steps = 1000000
            cfg.env.max_episode_seconds = 1000000.
            if full_trace:
                for name in ("hard", "timeout", "clip_end"):
                    cfg.env.terminations[name].func = "sports_runtime.never_terminate"
        ensure_registries()
        override = build_env_override(cfg, eval_mode=evaluation, learner_device=device)
        if evaluation:
            override["auto_reset"] = False
        return create_env(cfg, num_envs=count, env_cfg_override=override)
    finally:
        manager_terms.MDP = previous_mdp
        os.chdir(previous_cwd)


def never_terminate(env, command_name="motion"):
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
