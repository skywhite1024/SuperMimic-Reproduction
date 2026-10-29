"""Isolated local CPU runtime for the preserved Uni-SuperMimic checkpoint."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
UNI = Path(os.environ.get("SUPERMIMIC_UNI", str(ROOT.parent / "implementation/uni")))
CHECKPOINT = Path(os.environ.get("SUPERMIMIC_POLICY", str(ROOT.parent / "implementation/continuation-20261025-a/downloaded/policy/ckpt_016000.pt")))
sys.path.insert(0, str(UNI / "src"))
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def never_terminate(env, command_name="motion"):
    """Keep physical sampling after a failure; the runner separately records the frozen hard gate."""
    import torch
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def make_env(files, fps=30.0, continuous=False):
    import torch
    from hydra import compose, initialize_config_dir
    from unilab.base.config_adapter import create_env
    from unilab.training import ensure_registries
    from unisupermimic.scripts.train_supermimic import build_env_override
    os.chdir(UNI)
    torch.set_num_threads(2)
    with initialize_config_dir(version_base="1.3", config_dir=str(UNI / "src/unisupermimic/conf/supermimic")):
        cfg = compose(config_name="config", overrides=["training.play_only=true", "task=g1_super_mimic/mujoco"])
    cfg.env.commands.motion.motion_files = [str(Path(f).resolve()) for f in files]
    cfg.env.commands.motion.src_fps = fps
    cfg.env.commands.motion.max_episode_steps = 1000000
    cfg.env.max_episode_seconds = 1000000.0
    cfg.env.seed = 0
    if continuous:
        for name in ["hard", "timeout", "clip_end"]:
            cfg.env.terminations[name].func = "runtime.never_terminate"
    ensure_registries()
    override = build_env_override(cfg, eval_mode=True, learner_device="cpu")
    override["auto_reset"] = False
    env = create_env(cfg, num_envs=len(files), env_cfg_override=override)
    return env


def load_policy(env, checkpoint=None):
    import torch
    from unisupermimic.tasks.motion_tracking.g1.supermimic.evaluate import load_policy as upstream_load
    ckpt = torch.load(CHECKPOINT if checkpoint is None else checkpoint, map_location="cpu", weights_only=True)
    return upstream_load(env, ckpt, device="cpu"), ckpt


if __name__ == "__main__":
    import json
    import torch
    from unisupermimic.rl.algos.supermimic.geometry import unflatten_state
    from unisupermimic.rl.contracts.supermimic import SUPERMIMIC_STATE_KEY
    files = [UNI / "src/unisupermimic/assets/motions/g1/lafan1_supermimic/walk1_subject1.csv"]
    env = make_env(files)
    try:
        pi, ckpt = load_policy(env)
        _, info = env.reset_to_frames(env.supermimic_mdp.motion.clip_start.clone())
        cur = info[SUPERMIMIC_STATE_KEY]
        history = None
        with torch.inference_mode():
            for i in range(20):
                mdp = env.supermimic_mdp
                obs = mdp.obs(unflatten_state(cur["s"]),cur["a_prev"],cur["xi"],cur["f"])
                history = obs[:,None].repeat(1,ckpt["args"]["hist"],1) if history is None else torch.cat((history[:,1:],obs[:,None]),1)
                state = env.step(pi(history,mdp.goal(cur["f"],ckpt["args"]["goal_frames"])))
                cur = state.info[SUPERMIMIC_STATE_KEY]
        print('LOCAL_PHYSICAL_POLICY_PROBE_PASS '+json.dumps({'steps':20,'root':cur['s'][0,:3].tolist(),'backend':'mujoco','cuda':torch.cuda.device_count(),'obs_dim':mdp.obs_dim,'goal_frame_dim':mdp.goal_frame_dim}))
    finally:
        env.close();env.backend.close()
