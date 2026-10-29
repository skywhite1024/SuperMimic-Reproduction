"""Separate finite CPU collection diagnostic; not a measurement of the running GPU buffer."""
import hashlib,json,time
from dataclasses import fields
from pathlib import Path
import torch
from runtime import ROOT,make_env
from stage2_references import references
from balanced_sampling import BalancedSampling
from train_unified import digest
from unisupermimic.tasks.motion_tracking.g1.supermimic import manager_terms
from unisupermimic.rl.algos.supermimic.learner import SuperMimicConfig,SuperMimicLearner

def main():
    target=ROOT/'evidence/replay-occupancy-cpu-diagnostic.json';assert not target.exists()
    initial=ROOT/'runs/unified-h100-4000/final-full.pt';parent=torch.load(initial,map_location='cpu',weights_only=True)
    torch.set_num_threads(2);torch.manual_seed(27)
    cfg={field.name:parent['args'][field.name] for field in fields(SuperMimicConfig) if field.name in parent['args']}
    for name in ['wm_hidden','policy_hidden','critic_hidden','test_hidden_dims']:
        if cfg.get(name):cfg[name]=tuple(cfg[name])
    cfg.update(buffer_steps=300,compile=False,amp='none');names={path.stem for path in references('new')}
    sampler=BalancedSampling(names);manager_terms.sample_start_frames=sampler
    env=make_env(references('all'),device='cpu',evaluation=False,num_envs=128);started=time.monotonic()
    try:
        learner=SuperMimicLearner(env,SuperMimicConfig(**cfg),ROOT/'runs/replay-occupancy-cpu-diagnostic',device='cpu',backend='mujoco')
        learner.load(initial)
        for norm in [learner.pi.obs_norm,learner.pi.goal_norm,learner.wm.in_norm,learner.wm.out_norm,learner.critic.norm]:norm.max_count=0
        before={key:digest(getattr(learner,key)) for key in ['wm','pi','critic']}
        for _ in range(5):
            assert time.monotonic()-started<600
            learner.collect(100);print('REPLAY_DIAGNOSTIC_COLLECTED '+str(learner.sim_steps-parent['sim_steps']),flush=True)
        buffer=learner.buffer;motion=learner.mdp.motion;rows=buffer._row(torch.arange(buffer.size));frames=buffer.data['f'][rows]
        is_new=torch.tensor([name in names for name in motion.names]);ids=motion.clip_id[frames]
        valid=learner.mdp.valid_imagination_start(frames,learner.cfg.horizon)
        window_ids=ids[:buffer.size-learner.cfg.wm_window+1]
        reset=sampler.receipt()['clip_reset_samples'];reset_new=sum(value for key,value in reset.items() if key in names)/sum(reset.values())
        per_clip=[]
        for index,name in enumerate(motion.names):
            per_clip.append({'clip':name,'group':'new' if name in names else 'lafan','reset_start_samples':reset[name],
                             'buffer_frames':int((ids==index).sum()),'valid_policy_start_frames':int(((ids==index)&valid).sum())})
        after={key:digest(getattr(learner,key)) for key in before};assert before==after
        result={'status':'separate_CPU_collection_diagnostic_complete','measures_formal_GPU_training_buffer':False,
                'checkpoint_sha256':hashlib.sha256(initial.read_bytes()).hexdigest(),'reference_variant':'ground-calibration-v1','device':'cpu','num_envs':128,
                'collection_steps':500,'buffer_time_steps':buffer.size,'buffer_entries':frames.numel(),'new_reset_prior':.6,'observed_new_reset_fraction':reset_new,
                'new_replay_frame_fraction':float(is_new[ids].float().mean()),'new_world_model_window_start_fraction':float(is_new[window_ids].float().mean()),
                'new_valid_policy_start_fraction':float(is_new[ids][valid].float().mean()),'replay_and_eligible_sampling_rule':'native uniform age and environment, then reject starts without a complete imagination horizon',
                'all_model_and_normalizer_tensors_unchanged':True,'world_model_policy_critic_updates_executed':0,'wall_seconds':time.monotonic()-started,
                'formal_training_gradient_fraction_inferred':False,'clips':per_clip}
        target.write_text(json.dumps(result,indent=2)+'\n');print('REPLAY_DIAGNOSTIC_COMPLETE '+json.dumps({key:value for key,value in result.items() if key!='clips'}),flush=True)
    finally:env.close();env.backend.close()

if __name__=='__main__':main()
