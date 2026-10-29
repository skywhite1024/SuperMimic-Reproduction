"""Finite independent global-position adaptation; no mutation of the accepted baseline."""
import argparse
from dataclasses import fields
import hashlib
import json
import signal
import time
import torch
from runtime import ROOT, UNI
from global_model import GlobalPositionMDP
from unisupermimic.rl.algos.supermimic.learner import SuperMimicConfig,SuperMimicLearner


def digest(module):
    h=hashlib.sha256()
    for k,v in sorted(module.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def make_training_env(num_envs,backend='mjwarp'):
    import os
    from hydra import compose,initialize_config_dir
    from unilab.training import ensure_registries
    from unilab.base.config_adapter import create_env
    from unisupermimic.scripts.train_supermimic import build_env_override
    from unisupermimic.tasks.motion_tracking.g1.supermimic import manager_terms
    manager_terms.MDP=GlobalPositionMDP
    os.chdir(UNI)
    with initialize_config_dir(version_base='1.3',config_dir=str(UNI/'src/unisupermimic/conf/supermimic')):
        cfg=compose(config_name='config',overrides=[f'task=g1_super_mimic/{backend}'])
    cfg.env.commands.motion.motion_files=[str(ROOT/'references'/f'seed-{s}/sequence.csv') for s in [0,1,2]]
    cfg.env.commands.motion.src_fps=50.0
    cfg.env.seed=4
    ensure_registries()
    override=build_env_override(cfg,learner_device='cpu' if backend=='mujoco' else 'cuda:0')
    return create_env(cfg,num_envs=num_envs,env_cfg_override=override)


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--steps',type=int,default=2000);p.add_argument('--num-envs',type=int,default=4096);p.add_argument('--engineering',action='store_true');p.add_argument('--label');p.add_argument('--initial',type=__import__('pathlib').Path);p.add_argument('--adapter-only',action='store_true')
    args=p.parse_args();torch.set_num_threads(2 if args.device=='cpu' else 4)
    root=ROOT/'runs'/(args.label or ('global-cpu-engineering' if args.engineering else 'global-position-training'))
    assert root.resolve().is_relative_to((ROOT/'runs').resolve())
    assert not (root/'last.pt').exists(),'Existing run needs explicit resume selection, not overwrite'
    initial=(args.initial or ROOT/'runs/global-position/initial-full.pt').resolve()
    saved=torch.load(initial,map_location='cpu',weights_only=True)
    data={f.name:saved['args'][f.name] for f in fields(SuperMimicConfig) if f.name in saved['args']}
    for name in ['wm_hidden','policy_hidden','critic_hidden','test_hidden_dims']:
        if data.get(name):data[name]=tuple(data[name])
    data.update(iterations=saved['iteration']+args.steps,ckpt_every=500,log_every=50,compile=False)
    if args.engineering:data.update(wm_batches=1,wm_batch=32,pi_batch=32,critic_epochs=1,critic_minibatches=1,amp='none',buffer_steps=100)
    if args.adapter_only:data.update(wm_batches=0,no_critic=True)
    if args.device=='cuda':
        assert torch.cuda.device_count()==1 and 'H100' in torch.cuda.get_device_name(0)
        torch.cuda.set_device(0)
    env=make_training_env(args.num_envs,'mujoco' if args.device=='cpu' else 'mjwarp')
    learner=SuperMimicLearner(env,SuperMimicConfig(**data),root,device=args.device,backend='mujoco' if args.device=='cpu' else 'mjwarp')
    def configure_adapter():
        for parameter in learner.pi.parameters():parameter.requires_grad_(False)
        weight=learner.pi.dec[0].weight;weight.requires_grad_(True)
        mask=torch.zeros_like(weight)
        for i in range(data['hist']):mask[:,data['z_dim']+i*101+99:data['z_dim']+(i+1)*101]=1
        weight.register_hook(lambda grad:grad*mask)
        learner.opt_pi=torch.optim.Adam([weight],lr=data['lr'],betas=(.9,.95))
        learner.pi.obs_norm.max_count=0;learner.pi.goal_norm.max_count=0
        learner.wm.in_norm.max_count=0;learner.wm.out_norm.max_count=0
        return weight.detach().clone(),mask.bool()
    if args.adapter_only and len(saved['opt_pi']['param_groups'][0]['params'])==1:
        adapter_before,adapter_mask=configure_adapter()
    learner.load(initial)
    if args.adapter_only:
        if len(saved['opt_pi']['param_groups'][0]['params'])!=1:adapter_before,adapter_mask=configure_adapter()
        adapter_before=learner.pi.dec[0].weight.detach().clone()
    variant=saved.get('variant') or json.loads((ROOT/'runs/global-position/initial-full.json').read_text())
    if args.adapter_only:
        variant={**variant,'old_optimizer_counters_preserved':False,'policy_optimizer':'Fresh Adam on dec.0.weight with gradients masked to the 20 new XY-history columns; legacy columns fixed. WM/critic optimizer states retained without updates.','wm_normalizers_frozen':True,'legacy_policy_normalizers_frozen':True}
    original_checkpoint=learner._checkpoint
    learner._checkpoint=lambda include_optimizers:{**original_checkpoint(include_optimizers),'variant':variant}
    before={k:digest(getattr(learner,k)) for k in ['wm','pi','critic']}
    def stop(signum,frame):
        learner.save(root/'interrupted-full.pt');raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop)
    start=time.perf_counter()
    try:
        summary=learner.train()
        learner.save(root/'final-full.pt')
        after={k:digest(getattr(learner,k)) for k in ['wm','pi','critic']}
        if args.adapter_only:
            assert before['pi']!=after['pi'] and before['wm']==after['wm'] and before['critic']==after['critic']
            assert torch.equal(adapter_before[~adapter_mask],learner.pi.dec[0].weight.detach()[~adapter_mask])
        else:assert all(before[k]!=after[k] for k in before)
        loaded=torch.load(root/'final-full.pt',map_location='cpu',weights_only=True)
        for name in ['wm','pi','critic']:
            assert all(torch.equal(v.detach().cpu(),loaded[name][k]) for k,v in getattr(learner,name).state_dict().items())
        proof={'status':'passed','start_iteration':saved['iteration'],'end_iteration':learner.iteration,'new_iterations':args.steps,'num_envs':args.num_envs,'device':args.device,'before':before,'after':after,'all_three_models_changed':not args.adapter_only,'adapter_only':args.adapter_only,'base_policy_decoder_columns_frozen_exactly':args.adapter_only,'saved_model_readback_exact':True,'wall_seconds':time.perf_counter()-start,'variant':variant,'reward_addition':'1.5*exp(-world_xy_error_squared/.75**2)*.02; original reward retained','training_references':'same frozen seeds0,1,2 continuous references; independent extra seeds must be evaluated after training','summary':summary,'full_gpu_acceptance':args.device=='cuda' and not args.engineering}
        (root/'proof.json').write_text(json.dumps(proof,indent=2)+'\n');print('GLOBAL_POSITION_TRAIN_PASS '+json.dumps(proof),flush=True)
    finally:env.close();env.backend.close()


if __name__=='__main__':main()
