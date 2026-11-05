"""Isolated runtime for unified tracking over the frozen new library and old LAFAN1 clips."""
import os,sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
UNI=Path(os.environ.get('SUPERMIMIC_UNI',str(ROOT.parent/'implementation/uni')))
CHECKPOINT=Path(os.environ.get('SUPERMIMIC_POLICY',str(ROOT.parent/'interactive-g1-20261027-a/policy/acro-h100-2000.pt')))
sys.path.insert(0,str(UNI/'src'))
os.environ.setdefault('WANDB_MODE','offline');os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')


def references(group='all'):
    manifest=json.loads((ROOT/'library-manifest.json').read_text());new=[ROOT/x['path'] for x in manifest['files']]
    for path,row in zip(new,manifest['files']):assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
    old=sorted((UNI/'src/unisupermimic/assets/motions/g1/lafan1_supermimic').glob('*.csv'));assert len(old)==40
    return new if group=='new' else old if group=='lafan' else new+old


def never_terminate(env,command_name='motion'):
    import torch
    return torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)


def make_env(files,device='cpu',observation_dim=101,evaluation=True,num_envs=None,full_trace=True):
    from hydra import compose,initialize_config_dir
    from unilab.training import ensure_registries
    from unilab.base.config_adapter import create_env
    from unisupermimic.scripts.train_supermimic import build_env_override
    from unisupermimic.tasks.motion_tracking.g1.supermimic import manager_terms
    from unisupermimic.tasks.motion_tracking.g1.supermimic.mdp import MDP,MAX_STEP_REWARD
    if observation_dim==101:
        from global_model import GlobalPositionMDP
        if evaluation:
            class EvaluationMDP(GlobalPositionMDP):
                max_step_reward=MAX_STEP_REWARD
                def reward(self,*args,**kwargs):return MDP.reward(self,*args,**kwargs)
            manager_terms.MDP=EvaluationMDP
        else:manager_terms.MDP=GlobalPositionMDP
    else:assert observation_dim==99;manager_terms.MDP=MDP
    os.chdir(UNI)
    backend='mujoco' if device=='cpu' else 'mjwarp'
    with initialize_config_dir(version_base='1.3',config_dir=str(UNI/'src/unisupermimic/conf/supermimic')):
        cfg=compose(config_name='config',overrides=[f'task=g1_super_mimic/{backend}'])
    cfg.env.commands.motion.motion_files=[str(Path(p).resolve()) for p in files]
    cfg.env.commands.motion.src_fps=30.;cfg.env.seed=27 if not evaluation else 0
    if evaluation:
        cfg.env.commands.motion.max_episode_steps=1000000;cfg.env.max_episode_seconds=1000000.
        if full_trace:
            for name in ['hard','timeout','clip_end']:cfg.env.terminations[name].func='runtime.never_terminate'
    ensure_registries();override=build_env_override(cfg,eval_mode=evaluation,learner_device=device)
    if evaluation:override['auto_reset']=False
    return create_env(cfg,num_envs=len(files) if num_envs is None else num_envs,env_cfg_override=override)
