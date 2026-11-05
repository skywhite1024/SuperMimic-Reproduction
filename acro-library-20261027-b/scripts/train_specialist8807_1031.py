"""Finite SuperMimic acrobatic specialist; preserved global tracker is never overwritten."""
import argparse
from dataclasses import fields
import hashlib
import json
import signal
import time
from pathlib import Path
import torch
from runtime import ROOT, UNI
def digest(module):
    h=hashlib.sha256()
    for k,v in sorted(module.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
from unisupermimic.rl.algos.supermimic.learner import SuperMimicConfig, SuperMimicLearner


def environment(count, device):
    import os
    from hydra import compose, initialize_config_dir
    from unilab.training import ensure_registries
    from unilab.base.config_adapter import create_env
    from unisupermimic.scripts.train_supermimic import build_env_override
    from unisupermimic.tasks.motion_tracking.g1.supermimic import manager_terms
    from unisupermimic.tasks.motion_tracking.g1.supermimic.mdp import MDP
    manager_terms.MDP=MDP
    os.chdir(UNI)
    with initialize_config_dir(version_base='1.3',config_dir=str(UNI/'src/unisupermimic/conf/supermimic')):
        cfg=compose(config_name='config',overrides=[f'task=g1_super_mimic/{"mujoco" if device=="cpu" else "mjwarp"}'])
    cfg.env.commands.motion.motion_files=[str(ROOT/'data/88_07_stageii_unitree_g1.csv')]
    cfg.env.commands.motion.src_fps=30.;cfg.env.seed=27
    ensure_registries()
    return create_env(cfg,num_envs=count,env_cfg_override=build_env_override(cfg,learner_device=device))


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--label',required=True)
    p.add_argument('--steps',type=int,default=1000);p.add_argument('--world-model-warmup',type=int,default=400)
    p.add_argument('--num-envs',type=int,default=512);p.add_argument('--engineering',action='store_true');p.add_argument('--initial',type=Path)
    a=p.parse_args();torch.set_num_threads(2 if a.device=='cpu' else 4)
    out=ROOT/'runs'/a.label;assert out.resolve().is_relative_to((ROOT/'runs').resolve())
    assert not (out/'args.json').exists(),'Use a new run label to preserve results'
    parent=(a.initial or Path(__import__('os').environ.get('SUPERMIMIC_FULL',str(ROOT.parent/'implementation/continuation-20261025-a/downloaded/policy/full_016000.pt')))).resolve()
    source=ROOT/'data/88_07_stageii_unitree_g1.csv'
    assert hashlib.sha256(source.read_bytes()).hexdigest()=='b8cfbb0470486e625bcc3b840fa33437c333ebe3a9fef8ef83655f07dbc7cc7d'
    assert a.device=='cuda' and a.steps==1000 and a.world_model_warmup==400 and a.num_envs==512 and not a.engineering
    assert hashlib.sha256(parent.read_bytes()).hexdigest()=='0afcca9af97bd9db1f2dd083875898f962acab0b0a70c7313b221e575b70b05b'
    saved=torch.load(parent,map_location='cpu',weights_only=True);assert saved['iteration']==16000 and saved['pi']['obs_norm.mean'].numel()==99
    cfg={f.name:saved['args'][f.name] for f in fields(SuperMimicConfig) if f.name in saved['args']}
    for key in ['wm_hidden','policy_hidden','critic_hidden','test_hidden_dims']:
        if cfg.get(key):cfg[key]=tuple(cfg[key])
    cfg.update(iterations=saved['iteration']+a.steps,wm_only_iters=saved['iteration']+a.world_model_warmup,
               buffer_steps=300,wm_batches=4,wm_batch=512,pi_batch=512,critic_epochs=2,critic_minibatches=2,
               lr=5e-5,ckpt_every=500,log_every=50,compile=False,amp='bf16' if a.device=='cuda' else 'none')
    if a.engineering:cfg.update(buffer_steps=80,wm_batches=1,wm_batch=16,pi_batch=16,critic_epochs=1,critic_minibatches=1)
    if a.device=='cuda':assert torch.cuda.device_count()==1 and 'H100' in torch.cuda.get_device_name(0)
    env=environment(a.num_envs,a.device);learner=SuperMimicLearner(env,SuperMimicConfig(**cfg),out,device=a.device,backend='mujoco' if a.device=='cpu' else 'mjwarp')
    learner.load(parent)
    # Old coordinate normalizers remain exactly fixed while new acrobatic dynamics are learned.
    for norm in [learner.pi.obs_norm,learner.pi.goal_norm,learner.wm.in_norm,learner.wm.out_norm,learner.critic.norm]:norm.max_count=0
    before={k:digest(getattr(learner,k)) for k in ['wm','pi','critic']}
    spec={'variant':'specialist-8807-99d-1000','specialist_plan_sha256':hashlib.sha256((ROOT/'specialist8807-1031d-plan.json').read_bytes()).hexdigest(),'target_clip':'88_07_stageii_unitree_g1','parent_sha256':hashlib.sha256(parent.read_bytes()).hexdigest(),
          'reference_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'reference_source':'https://github.com/lez666/chunwan-flip',
          'original_policy_modified':False,'BONES_downloaded':False,'normalizers_frozen':True,'reference_fps':30,
          'training_resets_allowed':True,'evaluation_requires_zero_resets_after_t0':True}
    checkpoint=learner._checkpoint
    learner._checkpoint=lambda include_optimizers:{**checkpoint(include_optimizers),'variant':spec}
    def stop(signum,frame):learner.save(out/'interrupted-full.pt');raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop);start=time.perf_counter()
    try:
        result=learner.train();learner.save(out/'final-full.pt')
        after={k:digest(getattr(learner,k)) for k in before}
        assert before['wm']!=after['wm']
        if a.steps>a.world_model_warmup:assert before['pi']!=after['pi'] and before['critic']!=after['critic']
        readback=torch.load(out/'final-full.pt',map_location='cpu',weights_only=True)
        for name in before:
            assert all(torch.equal(v.detach().cpu(),readback[name][k]) for k,v in getattr(learner,name).state_dict().items())
        proof={'status':'updates_and_save_verified','device':a.device,'new_iterations':a.steps,'world_model_warmup_updates':a.world_model_warmup,
               'before':before,'after':after,'wall_seconds':time.perf_counter()-start,'training_summary':result,
               'normalizer_tensors_unchanged':all(torch.equal(saved[n][k],readback[n][k]) for n in before for k in saved[n] if '_norm.' in k or k.startswith('norm.')),
               'physical_success_not_inferred_from_training':True,'spec':spec}
        assert proof['normalizer_tensors_unchanged']
        export_path=ROOT/'policy/specialist8807-h100-1000-1031d.pt'
        assert not export_path.exists()
        exported={'pi':readback['pi'],'args':readback['args'],'iteration':readback['iteration'],'variant':spec}
        torch.save(exported,export_path)
        exported_readback=torch.load(export_path,map_location='cpu',weights_only=True)
        assert exported_readback['iteration']==17000
        assert all(torch.equal(v,exported_readback['pi'][k]) for k,v in readback['pi'].items())
        from unisupermimic.tasks.motion_tracking.g1.supermimic.evaluate import load_policy
        restored=load_policy(env,exported_readback,device=a.device)
        with torch.inference_mode():
            history=learner.obs_hist[:4]
            goal=learner.mdp.goal(learner._state_info()['f'][:4],learner.cfg.goal_frames)
            actual=learner.pi(history,goal);reloaded=restored(history,goal)
            assert torch.equal(actual,reloaded)
        proof.update(final_iteration=17000,export_path=export_path.relative_to(ROOT).as_posix(),
                     export_sha256=hashlib.sha256(export_path.read_bytes()).hexdigest(),
                     full_checkpoint_sha256=hashlib.sha256((out/'final-full.pt').read_bytes()).hexdigest(),
                     exported_PI_tensors_equal=True,functional_readback={'real_training_history_rows':4,'exact_output_equality':True},
                     final_unified_55_goal_complete=False,physical_acceptance_inferred=False)
        (out/'proof.json').write_text(json.dumps(proof,indent=2)+'\n');print('ACRO_TRAINING_VERIFIED '+json.dumps(proof),flush=True)
    finally:env.close();env.backend.close()


if __name__=='__main__':main()
