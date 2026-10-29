"""Three frozen 120 s physical trials, with no controlled-state writes after t=0."""
import argparse,hashlib,json,time,os
from pathlib import Path
import numpy as np
import torch
from runtime import ROOT,UNI
from unisupermimic.rl.algos.supermimic.geometry import unflatten_state,yaw_mat
from unisupermimic.rl.contracts.supermimic import SUPERMIMIC_STATE_KEY,SUPERMIMIC_TRANSITION_KEY
from unisupermimic.tasks.motion_tracking.g1.supermimic.evaluate import load_policy


def sequence_env(files,dimension,device):
    from hydra import compose,initialize_config_dir
    from unilab.training import ensure_registries
    from unilab.base.config_adapter import create_env
    from unisupermimic.scripts.train_supermimic import build_env_override
    from unisupermimic.tasks.motion_tracking.g1.supermimic import manager_terms
    from unisupermimic.tasks.motion_tracking.g1.supermimic.mdp import MDP
    if dimension==101:
        from global_model import GlobalPositionMDP
        class EvaluationMDP(GlobalPositionMDP):
            max_step_reward=.1
            def reward(self,*args,**kwargs):return MDP.reward(self,*args,**kwargs)
        manager_terms.MDP=EvaluationMDP
    else:assert dimension==99;manager_terms.MDP=MDP
    os.chdir(UNI)
    with initialize_config_dir(version_base='1.3',config_dir=str(UNI/'src/unisupermimic/conf/supermimic')):
        cfg=compose(config_name='config',overrides=['task=g1_super_mimic/'+('mujoco' if device=='cpu' else 'mjwarp')])
    cfg.env.commands.motion.motion_files=[str(x) for x in files];cfg.env.commands.motion.src_fps=50.
    cfg.env.commands.motion.max_episode_steps=1000000;cfg.env.max_episode_seconds=1000000.;cfg.env.seed=0
    for name in ['hard','timeout','clip_end']:cfg.env.terminations[name].func='runtime.never_terminate'
    ensure_registries();override=build_env_override(cfg,eval_mode=True,learner_device=device);override['auto_reset']=False
    return create_env(cfg,num_envs=len(files),env_cfg_override=override)


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--mode',required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu');a=p.parse_args();checkpoint=a.checkpoint.resolve()
    folder=ROOT/'runs'/a.mode;assert folder.resolve().is_relative_to((ROOT/'runs').resolve()) and not (folder/'summary.json').exists();folder.mkdir(parents=True,exist_ok=True)
    index=ROOT/'continuous-corrected-v1/index.json';frozen=json.loads(index.read_text());assert frozen['seeds']==[0,1,2]
    files=[ROOT/f'continuous-corrected-v1/seed-{seed}/sequence.csv' for seed in frozen['seeds']]
    for file,row in zip(files,frozen['sequences']):assert hashlib.sha256(file.read_bytes()).hexdigest()==row['reference_sha256']
    torch.set_num_threads(2 if a.device=='cpu' else 4);torch.manual_seed(0);saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    env=sequence_env(files,saved['pi']['obs_norm.mean'].numel(),a.device);records={key:[] for key in ['state','reference_qpos','xy_error','yaw_error','body_global','body_local','hard_failure','action']}
    resets=0;started=time.monotonic()
    try:
        policy=load_policy(env,saved,device=a.device);mdp=env.supermimic_mdp;motion=mdp.motion
        _,info=env.reset_to_frames(motion.clip_start.clone());current=info[SUPERMIMIC_STATE_KEY];history=None
        def forbidden_reset(*args,**kwargs):
            nonlocal resets
            resets+=1;raise RuntimeError('No resets after t0')
        env.reset=forbidden_reset
        score=torch.zeros(3,device=a.device);failed=torch.zeros(3,dtype=torch.bool,device=a.device)
        with torch.inference_mode():
            for step in range(6001):
                assert time.monotonic()-started<900,'Finite sequence evaluation exceeded wall budget'
                frame=current['f'];s=unflatten_state(current['s']);assert torch.isfinite(current['s']).all()
                refp,refR=motion.root_p[frame],motion.root_R[frame];body=mdp.kin.forward(*s)[0]
                rotation=yaw_mat(s[1]@refR.transpose(-1,-2));anchor=torch.cat((s[0][:,:2],refp[:,2:3]),-1)
                local=anchor[:,None]+(rotation[:,None]@(motion.body_p[frame]-refp[:,None])[...,None]).squeeze(-1)
                yaw=torch.atan2(s[1][:,1,0],s[1][:,0,0])-torch.atan2(refR[:,1,0],refR[:,0,0])
                values={'state':current['s'],'reference_qpos':torch.cat((refp,refR.flatten(1),motion.q[frame]),-1),
                        'xy_error':(s[0][:,:2]-refp[:,:2]).norm(dim=-1),'yaw_error':torch.atan2(torch.sin(yaw),torch.cos(yaw)).abs(),
                        'body_global':(body-motion.body_p[frame]).norm(dim=-1).mean(-1),'body_local':(body-local).norm(dim=-1).mean(-1),'hard_failure':mdp.termination(s,frame)}
                for key,value in values.items():records[key].append(value.cpu().numpy().copy())
                if step==6000:break
                obs=mdp.obs(s,current['a_prev'],current['xi'],frame)
                history=obs[:,None].repeat(1,saved['args']['hist'],1) if history is None else torch.cat((history[:,1:],obs[:,None]),1)
                action=policy(history,mdp.goal(frame,saved['args']['goal_frames']));assert torch.isfinite(action).all();records['action'].append(action.cpu().numpy().copy())
                state=env.step(action);current=state.info[SUPERMIMIC_STATE_KEY]
                gate=mdp.termination(unflatten_state(current['s']),frame+1);score+=torch.where(~failed,state.info[SUPERMIMIC_TRANSITION_KEY]['r'],0);failed|=gate
                assert not bool(state.terminated.any() or state.truncated.any())
                if step%1000==0:print('FROZEN_SEQUENCE_PROGRESS '+json.dumps({'step':step,'seconds':step*.02,'resets':resets}),flush=True)
        packed={key:np.stack(value) for key,value in records.items()};np.savez_compressed(folder/'trajectory.npz',**packed);rows=[]
        for i,seed in enumerate(frozen['seeds']):
            hard=packed['hard_failure'][:,i];states=packed['state'][:,i];end=states[-50:];rzz=1-2*(states[:,4]**2+states[:,5]**2)
            windows=[]
            for segment in frozen['sequences'][i]['segments']:
                start=segment['transition_start_frame'];stop=segment['source_start_in_sequence_frame']+50
                windows.append({'source':segment['source'],'group':segment['group'],'time_s':start/50,
                                'hard_failure':bool(hard[start:stop].any()),'xy_peak_m':float(packed['xy_error'][start:stop,i].max())})
            rows.append({'seed':seed,'elapsed_seconds':120,'post_initialization_resets':resets,'hard_failure_frames':int(hard.sum()),
                         'first_hard_failure_s':float(np.flatnonzero(hard)[0]/50) if hard.any() else None,'original_return_until_first_failure':float(score[i]/600),
                         'root_xy_rmse_m':float(np.sqrt(np.mean(packed['xy_error'][:,i]**2))),'root_xy_peak_m':float(packed['xy_error'][:,i].max()),
                         'yaw_rmse_deg':float(np.rad2deg(np.sqrt(np.mean(packed['yaw_error'][:,i]**2)))),'body_global_mean_m':float(packed['body_global'][:,i].mean()),
                         'body_local_mean_m':float(packed['body_local'][:,i].mean()),'passed_original_hard_gate':not bool(hard.any()),
                         'terminal_last_second_root_speed_peak_m_s':float(np.linalg.norm(end[:,7:10],axis=1).max()),
                         'terminal_last_second_joint_speed_rms_rad_s':float(np.sqrt(np.mean(end[:,42:]**2))),
                         'terminal_last_second_pelvis_min_m':float(end[:,2].min()),'terminal_last_second_Rzz_min':float(rzz[-50:].min()),'transition_windows':windows})
        result={'mode':a.mode,'checkpoint':str(checkpoint),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'device':a.device,
                'frozen_reference_index_sha256':hashlib.sha256(index.read_bytes()).hexdigest(),'reference_version':'continuous-corrected-v1','denominator':3,
                'new_full_clip_union_denominator':15,'wall_seconds':time.monotonic()-started,'no_controlled_state_writes_after_t0':True,'post_initialization_resets':resets,
                'passed_original_hard_gate':sum(x['passed_original_hard_gate'] for x in rows),'trials':rows}
        (folder/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print('FROZEN_SEQUENCES_COMPLETE '+json.dumps({key:value for key,value in result.items() if key!='trials'}),flush=True)
    finally:env.close();env.backend.close()


if __name__=='__main__':main()
