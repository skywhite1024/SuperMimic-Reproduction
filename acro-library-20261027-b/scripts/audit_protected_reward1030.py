"""Read-only native reward decomposition on existing complete protected trajectories."""
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import psutil
import torch
from runtime import ROOT, make_env, references
from unisupermimic.rl.algos.supermimic.geometry import unflatten_state
from unisupermimic.tasks.motion_tracking.g1.supermimic.mdp import MDP

def read(name):
    return json.loads((ROOT/name).read_text(encoding='utf8'))

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    plan_path=ROOT/'evidence/protected-reward-audit-1030-plan-v2.json'
    plan=read(str(plan_path.relative_to(ROOT)))
    assert sha(Path(__file__).resolve())==plan['script_sha256']
    for name,digest in plan['input_and_source_sha256'].items():
        assert sha(ROOT/name)==digest,name
    qualification=read('evidence/teacher-qualification-1030-execution.json')
    assert qualification['status']=='complete_all30_full_tail_videos_decoded_hashed'
    assert not qualification['protected_teacher_qualified']
    output=ROOT/'evidence/protected-reward-audit-1030.json'
    assert not output.exists(),'Preserve the first complete audit'
    process=psutil.Process();process.cpu_affinity(process.cpu_affinity()[:2]);torch.set_num_threads(2)
    start=time.monotonic();rows=[];sources={}
    env=make_env(references('new'),device='cpu',observation_dim=101,evaluation=True)
    mdp=env.supermimic_mdp
    try:
        # Native reset is used only to read the defined initial previous-action value.
        # No retained trajectory state is written to this environment and env.step is never called.
        _,info=env.reset_to_frames(mdp.motion.clip_start.clone())
        previous_at_t0=info['supermimic_state']['a_prev'] if 'supermimic_state' in info else None
        if previous_at_t0 is None:
            from unisupermimic.rl.contracts.supermimic import SUPERMIMIC_STATE_KEY
            previous_at_t0=info[SUPERMIMIC_STATE_KEY]['a_prev']
        for mode in plan['modes']:
            assert time.monotonic()-start<plan['maximum_program_seconds']
            summary=read(f'runs/{mode}/summary.json')
            contact=read(f'runs/{mode}/contact-samples.json')
            samples={item['clip']:item['samples'] for item in contact}
            for filename in ['summary.json','trajectory.npz','combined-assessment.json','contact-samples.json']:
                relative=f'runs/{mode}/{filename}';sources[relative]=sha(ROOT/relative)
            with np.load(ROOT/f'runs/{mode}/trajectory.npz') as archive:
                for name in plan['protected_new']:
                    index=next(i for i,item in enumerate(summary['clips']) if item['clip']==name)
                    row=summary['clips'][index];length=row['full_clip_steps']
                    states=torch.from_numpy(archive['state'][1:length+1,index].copy())
                    actions=torch.from_numpy(archive['action'][:length,index].copy())
                    assert torch.isfinite(states).all() and torch.isfinite(actions).all()
                    prev=torch.cat([previous_at_t0[index:index+1],actions[:-1]])
                    frames=torch.arange(length,dtype=torch.long)+mdp.motion.clip_start[index]+1
                    assert mdp.motion.names[index]==name
                    expected=torch.cat([mdp.motion.root_p[frames],mdp.motion.root_R[frames].flatten(1),mdp.motion.q[frames]],dim=-1)
                    assert np.allclose(expected.numpy(),archive['reference_qpos'][1:length+1,index],rtol=0,atol=1e-6)
                    with torch.inference_mode():
                        rewards,terms=MDP.reward(mdp,unflatten_state(states),frames,actions,prev,terms=True)
                    scored=row['scored_steps'];score=float(rewards[:scored].sum())
                    difference=abs(score-row['reward_sum_original_until_first_failure'])
                    assert difference<5e-4,(mode,name,difference)
                    scale=row['normalization_denominator']
                    action_penalty=terms['action_rate']*mdp.cfg.w_action_rate*.02
                    joint_penalty=terms['joint_limit']*mdp.cfg.w_joint_limit*.02
                    components={'native_tracking':float(terms['track'][:scored].sum()/scale),
                                'action_rate_penalty':float(action_penalty[:scored].sum()/scale),
                                'joint_limit_penalty':float(joint_penalty[:scored].sum()/scale)}
                    assert abs(sum(components.values())-row['norm_return_original'])<5e-5
                    geometric=samples[name][5:]
                    bound=next(item for item in summary['action_bound_observations'] if item['clip']==name)
                    rows.append({'mode':mode,'clip':name,'denominator_steps':length,'scored_steps':scored,
                                 'original_normalized_return':row['norm_return_original'],'normalized_native_components':components,
                                 'reward_reconstruction_abs_difference':difference,'return_reconstruction_abs_difference':abs(sum(components.values())-row['norm_return_original']),
                                 'actual_ground_distance_min_after_100ms_m':min(item['actual_min_contact_distance_m'] for item in geometric),
                                 'reference_ground_distance_min_after_100ms_m':min(item['reference_min_contact_distance_m'] for item in geometric),
                                 'actual_penetration_over_5cm_samples_after_100ms':sum(item['actual_min_contact_distance_m']<-.05 for item in geometric),
                                 'reference_penetration_over_5cm_samples_after_100ms':sum(item['reference_min_contact_distance_m']<-.05 for item in geometric),
                                 'clipped_action_fraction':bound['clipped_elements']/max(1,bound['active_action_elements']),
                                 'peak_unbounded_action_output':bound['peak_raw_output'],
                                 'full_trajectory_reused':True,'post_t0_resets_in_original_trial':row['post_initialization_resets']})
            print('PROTECTED_REWARD_AUDIT_MODE '+mode,flush=True)
        assert len(rows)==len(plan['modes'])*3
        for name,digest in plan['input_and_source_sha256'].items():
            assert sha(ROOT/name)==digest,name
        result={'status':'complete_original_reward_components_reconstructed_on_all18_protected_records',
                'checked_at':datetime.now(timezone.utc).isoformat(),'plan_sha256':sha(plan_path),'CPU_affinity':process.cpu_affinity(),
                'wall_seconds':time.monotonic()-start,'GPU':0,'training_updates':0,'new_physical_rollouts':0,'env_step_calls':0,
                'all_full_trajectories_preserved':True,'source_hashes':sources,'rows':rows,
                'causal_claim':False,'full_goal_complete':False,'next_GPU_training_automatically_authorized':False}
        assert result['wall_seconds']<plan['maximum_program_seconds']
        output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
        print('PROTECTED_REWARD_AUDIT_COMPLETE '+json.dumps({'records':len(rows),'maximum_reward_error':max(row['reward_reconstruction_abs_difference'] for row in rows),'wall_seconds':result['wall_seconds']}),flush=True)
    finally:
        env.close();env.backend.close()

if __name__=='__main__':
    main()
