"""Finite unified 101D continuation with balanced 15-clip learning and 40-clip rehearsal."""
import argparse,hashlib,json,signal,time
from dataclasses import fields
from pathlib import Path
import torch
from runtime import ROOT,make_env
from stage2_references import references
from balanced_sampling import BalancedSampling
from balanced_replay import BalancedReplay
from unisupermimic.tasks.motion_tracking.g1.supermimic import manager_terms
from unisupermimic.rl.algos.supermimic.learner import SuperMimicConfig,SuperMimicLearner


def digest(module):
    h=hashlib.sha256()
    for key,value in sorted(module.state_dict().items()):h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--initial',type=Path,required=True);p.add_argument('--label',required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--steps',type=int,default=4000)
    p.add_argument('--wm-warmup',type=int,default=400);p.add_argument('--num-envs',type=int,default=1024);p.add_argument('--engineering',action='store_true')
    a=p.parse_args();torch.manual_seed(27);initial=a.initial.resolve();out=ROOT/'runs'/a.label
    plan=json.loads((ROOT/'stage3-plan.json').read_text());assert plan['initial_full_sha256']==hashlib.sha256(initial.read_bytes()).hexdigest();assert out.resolve().is_relative_to((ROOT/'runs').resolve())
    assert not (out/'args.json').exists(),'Choose a fresh label; preserve every prior run'
    torch.set_num_threads(2 if a.device=='cpu' else 4)
    parent=torch.load(initial,map_location='cpu',weights_only=True);assert parent['pi']['obs_norm.mean'].numel()==101
    cfg={f.name:parent['args'][f.name] for f in fields(SuperMimicConfig) if f.name in parent['args']}
    for name in ['wm_hidden','policy_hidden','critic_hidden','test_hidden_dims']:
        if cfg.get(name):cfg[name]=tuple(cfg[name])
    cfg.update(iterations=parent['iteration']+a.steps,wm_only_iters=parent['iteration']+a.wm_warmup,buffer_steps=300,
               wm_batches=4,wm_batch=512,pi_batch=512,critic_epochs=2,critic_minibatches=2,lr=5e-5,
               ckpt_every=500,log_every=50,compile=False,amp='none' if a.device=='cpu' else 'bf16')
    if a.engineering:cfg.update(buffer_steps=80,wm_batches=1,wm_batch=16,pi_batch=16,critic_epochs=1,critic_minibatches=1)
    if a.device=='cuda':assert torch.cuda.device_count()==1 and 'H100' in torch.cuda.get_device_name(0)
    files=references('all');assert len(files)==55
    sampler=BalancedSampling([path.stem for path in references('new')]);manager_terms.sample_start_frames=sampler
    env=make_env(files,device=a.device,evaluation=False,num_envs=a.num_envs)
    learner=SuperMimicLearner(env,SuperMimicConfig(**cfg),out,device=a.device,backend='mujoco' if a.device=='cpu' else 'mjwarp')
    learner.load(initial)
    replay=BalancedReplay(learner.buffer,learner.mdp,learner.cfg.horizon,[path.stem for path in references('new')],plan['old_failure_replay_priorities'])
    replay.install(learner)
    for norm in [learner.pi.obs_norm,learner.pi.goal_norm,learner.wm.in_norm,learner.wm.out_norm,learner.critic.norm]:norm.max_count=0
    before={name:digest(getattr(learner,name)) for name in ['wm','pi','critic']}
    variant={**parent.get('variant',{}),'name':'unified-55-reference-tracker-101d-balanced-replay-stage3','reference_variant':'ground-calibration-v1','raw_reference_assessment_retained':True,'one_policy':True,'new_library_count':15,'old_lafan_rehearsal_count':40,
             'normalizers_fixed':True,'replay_sampling':'group_clip_balanced_stage3','stage3_plan_sha256':hashlib.sha256((ROOT/'stage3-plan.json').read_bytes()).hexdigest(),'group_sampling_mass':{'new':.6,'lafan':.4},'BONES_downloaded':False,'training_resets_allowed':True,'evaluation_requires_zero_resets_after_t0':True}
    checkpoint=learner._checkpoint;learner._checkpoint=lambda include_optimizers:{**checkpoint(include_optimizers),'variant':variant}
    def stop(signum,frame):learner.save(out/'interrupted-full.pt');raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop);start=time.perf_counter()
    try:
        summary=learner.train();learner.save(out/'final-full.pt');saved=torch.load(out/'final-full.pt',map_location='cpu',weights_only=True)
        after={name:digest(getattr(learner,name)) for name in before};assert before['wm']!=after['wm']
        if a.steps>a.wm_warmup:assert all(before[name]!=after[name] for name in before)
        for name in before:assert all(torch.equal(v.detach().cpu(),saved[name][k]) for k,v in getattr(learner,name).state_dict().items())
        norms=all(torch.equal(parent[name][key],saved[name][key]) for name in before for key in parent[name] if '_norm.' in key or key.startswith('norm.'));assert norms
        sampling=sampler.receipt();replay_sampling=replay.receipt()
        if not a.engineering:
            for name in ['wm_starts','pi_consumed']:
                observed=replay_sampling['observed_sampling'][name]
                assert all(count>0 for count in observed['by_clip'].values())
                assert abs(observed['new_fraction']-.6)<.015
        if not a.engineering:assert len(sampling['clip_reset_samples'])==55 and min(sampling['clip_reset_samples'].values())>0
        result={'status':'updates_and_save_readback_verified','device':a.device,'new_updates':a.steps,'world_model_warmup':a.wm_warmup,
                'parent_sha256':hashlib.sha256(initial.read_bytes()).hexdigest(),'before':before,'after':after,'normalizer_tensors_unchanged':norms,
                'wall_seconds':time.perf_counter()-start,'sampling':sampling,'replay_sampling':replay_sampling,'training_summary':summary,'variant':variant,'physical_acceptance_inferred':False}
        (out/'proof.json').write_text(json.dumps(result,indent=2)+'\n');print('UNIFIED_TRAINING_VERIFIED '+json.dumps(result),flush=True)
    finally:env.close();env.backend.close()


if __name__=='__main__':main()
