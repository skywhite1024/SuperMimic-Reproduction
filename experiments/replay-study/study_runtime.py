"""Native 99D G1 runtime; no XY observation, reward, teacher or controller change."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).resolve().parent
UNI = ROOT / 'implementation/uni'
LEGACY = ROOT / 'acro-library-20261027-b/scripts'
sys.path.insert(0, str(UNI / 'src'))
sys.path.insert(0, str(LEGACY))
os.environ.setdefault('WANDB_MODE', 'offline')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('MUJOCO_GL', 'disable')
os.environ.setdefault('OMP_NUM_THREADS', '2')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def references():
    from runtime import references as raw_references
    files = raw_references('all')
    new = raw_references('new')
    assert len(files) == 55 and len(new) == 15
    return files, {path.stem for path in new}


def verify_inputs(files):
    hashes = {path.relative_to(ROOT).as_posix(): sha(path) for path in files}
    frozen = json.loads((STUDY/'inputs.lock.json').read_text(encoding='utf-8'))
    assert hashes == frozen['motion_sha256'], 'Input motions differ from the shared study snapshot'
    return hashes


def make_env(files, device, count, seed, sampler):
    from hydra import compose, initialize_config_dir
    from unilab.training import ensure_registries
    from unilab.base.config_adapter import create_env
    from unisupermimic.scripts.train_supermimic import build_env_override
    from unisupermimic.tasks.motion_tracking.g1.supermimic import manager_terms
    from unisupermimic.tasks.motion_tracking.g1.supermimic.mdp import MDP
    manager_terms.MDP = MDP
    manager_terms.sample_start_frames = sampler
    backend = 'mujoco' if device == 'cpu' else 'mjwarp'
    previous = Path.cwd()
    try:
        os.chdir(UNI)
        with initialize_config_dir(version_base='1.3', config_dir=str(UNI/'src/unisupermimic/conf/supermimic')):
            cfg = compose(config_name='config', overrides=[f'task=g1_super_mimic/{backend}'])
        cfg.env.commands.motion.motion_files = [str(path.resolve()) for path in files]
        cfg.env.commands.motion.src_fps = 30.
        cfg.env.seed = seed
        ensure_registries()
        return create_env(cfg, num_envs=count, env_cfg_override=build_env_override(cfg, learner_device=device))
    finally:
        os.chdir(previous)


def plan():
    return json.loads((STUDY/'plan.json').read_text(encoding='utf-8'))
