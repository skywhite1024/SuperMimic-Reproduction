"""Full physical traces plus the frozen original first-failure reward denominator."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from runtime import ROOT,CHECKPOINT,references,make_env
from unisupermimic.rl.algos.supermimic.geometry import unflatten_state,yaw_mat
from unisupermimic.rl.contracts.supermimic import SUPERMIMIC_STATE_KEY,SUPERMIMIC_TRANSITION_KEY
from unisupermimic.tasks.motion_tracking.g1.supermimic.evaluate import load_policy
from unisupermimic.tasks.motion_tracking.g1.supermimic.mdp import MAX_STEP_REWARD


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,default=CHECKPOINT);p.add_argument('--mode',required=True)
    p.add_argument('--group',choices=['new','lafan','all'],default='all');p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--clips',nargs='+');a=p.parse_args();checkpoint=a.checkpoint.resolve()
    out=ROOT/'runs'/a.mode;assert out.resolve().is_relative_to((ROOT/'runs').resolve())
    assert not (out/'summary.json').exists(),'Preserve previous physical evidence';out.mkdir(parents=True,exist_ok=True)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True);dimension=saved['pi']['obs_norm.mean'].numel()
    files=references(a.group)
    if a.clips:
        selected=set(a.clips);files=[path for path in files if path.stem in selected];assert len(files)==len(selected)
    torch.set_num_threads(2 if a.device=='cpu' else 4);torch.manual_seed(0);start=time.perf_counter()
    env=make_env(files,device=a.device,observation_dim=dimension,evaluation=True)
    packed={};reset_calls=0
    try:
        policy=load_policy(env,saved,device=a.device);mdp=env.supermimic_mdp;motion=mdp.motion
        _,info=env.reset_to_frames(motion.clip_start.clone());current=info[SUPERMIMIC_STATE_KEY]
        def forbidden_reset(*args,**kwargs):
            nonlocal reset_calls
            reset_calls+=1;raise RuntimeError('Physical evaluation forbids resets after t0')
        env.reset=forbidden_reset
        n=len(files);lengths=(motion.clip_len-1).cpu().numpy().astype(int);maximum=int(lengths.max())
        ends=motion.clip_end.clone();command=env.command_manager.get_term('motion')
        shapes={'state':71,'reference_qpos':41,'action':29}
        for name,size in shapes.items():packed[name]=np.full((maximum+1,n,size),np.nan,dtype=np.float32)
        for name in ['xy_error','yaw_error','body_global','body_local','root_error','hard_failure_aligned','hard_failure_original']:
            packed[name]=np.full((maximum+1,n),np.nan,dtype=np.float32)
        valid=np.zeros((maximum+1,n),dtype=bool);score=torch.zeros(n,device=a.device);failed=torch.zeros(n,dtype=torch.bool,device=a.device)
        score_steps=torch.zeros(n,dtype=torch.long,device=a.device);first_failure=np.full(n,np.nan);history=None
        with torch.inference_mode():
            for step in range(maximum+1):
                frame=current['f'];s=unflatten_state(current['s']);refp,refR=motion.root_p[frame],motion.root_R[frame]
                mask=step<=lengths;valid[step]=mask
                assert bool(torch.isfinite(current['s'][torch.tensor(mask,device=a.device)]).all()),'Nonfinite active physical trajectory'
                body=mdp.kin.forward(*s)[0];local_rotation=yaw_mat(s[1]@refR.transpose(-1,-2))
                anchor=torch.cat((s[0][:,:2],refp[:,2:3]),-1)
                local_ref=anchor[:,None]+(local_rotation[:,None]@(motion.body_p[frame]-refp[:,None])[...,None]).squeeze(-1)
                yaw=torch.atan2(s[1][:,1,0],s[1][:,0,0])-torch.atan2(refR[:,1,0],refR[:,0,0])
                values={'state':current['s'],'reference_qpos':torch.cat((refp,refR.flatten(1),motion.q[frame]),-1),
                        'xy_error':(s[0][:,:2]-refp[:,:2]).norm(dim=-1),'root_error':(s[0]-refp).norm(dim=-1),
                        'yaw_error':torch.atan2(torch.sin(yaw),torch.cos(yaw)).abs(),
                        'body_global':(body-motion.body_p[frame]).norm(dim=-1).mean(-1),'body_local':(body-local_ref).norm(dim=-1).mean(-1),
                        'hard_failure_aligned':mdp.termination(s,frame)}
                for name,value in values.items():packed[name][step,mask]=value.cpu().numpy()[mask]
                if step==maximum:break
                obs=mdp.obs(s,current['a_prev'],current['xi'],frame)
                history=obs[:,None].repeat(1,saved['args']['hist'],1) if history is None else torch.cat((history[:,1:],obs[:,None]),1)
                action=policy(history,mdp.goal(frame,saved['args']['goal_frames']));assert torch.isfinite(action).all()
                packed['action'][step,step<lengths]=action.cpu().numpy()[step<lengths]
                previous_frames=frame.clone()
                # Completed rows still undergo batch physics, but the native reward reads f+1.
                # Keep their command f at end-1 so that read remains inside their own clip.
                command.f.copy_(torch.minimum(command.f,ends-1))
                state=env.step(action);next_info=state.info[SUPERMIMIC_STATE_KEY]
                # Native manager evaluates the post-physics state at pre-action f + 1.
                gate=mdp.termination(unflatten_state(next_info['s']),torch.minimum(previous_frames+1,ends))
                active=torch.tensor(step<lengths,device=a.device);scoring=active&~failed
                score+=torch.where(scoring,state.info[SUPERMIMIC_TRANSITION_KEY]['r'],0);score_steps+=scoring
                newly_failed=(scoring&gate).cpu().numpy();first_failure[newly_failed]=(step+1)*.02;failed|=gate&active
                packed['hard_failure_original'][step,step<lengths]=gate.cpu().numpy()[step<lengths]
                # Only the reference clock of already-completed clips is held for vectorization.
                # No root/joint state is written; active clips retain their original frame timing.
                command.f.copy_(torch.minimum(command.f,ends));next_info['f'].copy_(torch.minimum(next_info['f'],ends));current=next_info
                assert not bool(state.terminated.any() or state.truncated.any())
                if step%500==0:print('LIBRARY_PHYSICAL_PROGRESS '+json.dumps({'step':step,'maximum':maximum,'resets':reset_calls}),flush=True)
        assert np.array_equal(valid.sum(0),lengths+1)
        packed['valid']=valid;packed['planned_steps']=lengths;np.savez_compressed(out/'trajectory.npz',**packed)
        normalized=(score/((motion.clip_len-1).float()*MAX_STEP_REWARD)).cpu().numpy()
        reward=score.cpu().numpy();steps=score_steps.cpu().numpy();rows=[];new_names={path.stem for path in references('new')}
        for i,name in enumerate(motion.names):
            mask=valid[:,i];last=int(lengths[i]);states=packed['state'][:last+1,i];ref=packed['reference_qpos'][:last+1,i]
            rzz=1-2*(states[:,4]**2+states[:,5]**2);inversion=ref[:,11]<-.5
            row={'clip':name,'group':'new' if name in new_names else 'lafan',
                 'full_clip_steps':int(lengths[i]),'elapsed_seconds':last*.02,'post_initialization_resets':reset_calls,
                 'norm_return_original':float(normalized[i]),'reward_sum_original_until_first_failure':float(reward[i]),
                 'normalization_denominator':float(lengths[i]*MAX_STEP_REWARD),'scored_steps':int(steps[i]),
                 'first_original_hard_failure_s':None if np.isnan(first_failure[i]) else float(first_failure[i]),
                 'hard_failure_original_frames':int(np.nansum(packed['hard_failure_original'][:,i])),
                 'hard_failure_aligned_frames':int(np.nansum(packed['hard_failure_aligned'][:,i])),
                 'passed_original_hard_gate':not bool(failed[i]),'root_xy_rmse_m':float(np.sqrt(np.mean(packed['xy_error'][mask,i]**2))),
                 'root_xy_peak_m':float(packed['xy_error'][mask,i].max()),'yaw_rmse_deg':float(np.rad2deg(np.sqrt(np.mean(packed['yaw_error'][mask,i]**2)))),
                 'body_global_mean_m':float(packed['body_global'][mask,i].mean()),'body_local_mean_m':float(packed['body_local'][mask,i].mean()),
                 'reference_inversion_frames':int(inversion.sum()),'inversion_observed_when_expected':bool((rzz[inversion]<-.5).any()) if inversion.any() else None,
                 'end_pelvis_m':float(states[-1,2]),'end_Rzz':float(rzz[-1]),'end_root_speed_m_s':float(np.linalg.norm(states[-1,7:10]))}
            rows.append(row);print('LIBRARY_PHYSICAL_CLIP '+json.dumps({k:row[k] for k in ['clip','group','passed_original_hard_gate','norm_return_original','first_original_hard_failure_s']}),flush=True)
        aggregate={}
        for group in ['new','lafan']:
            subset=[x for x in rows if x['group']==group]
            if subset:aggregate[group]={'denominator':len(subset),'passed':sum(x['passed_original_hard_gate'] for x in subset),'macro_return_original':float(np.mean([x['norm_return_original'] for x in subset])),'frame_weighted_return_original':sum(x['reward_sum_original_until_first_failure'] for x in subset)/sum(x['normalization_denominator'] for x in subset)}
        result={'mode':a.mode,'checkpoint':str(checkpoint),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'device':a.device,
                'observation_dim':dimension,'one_policy':True,'backend':'mujoco' if a.device=='cpu' else 'mjwarp','seed':0,
                'evaluation_contract_version':2,'normalization':'original float32 0.1 maximum reward times complete resampled clip length minus one; post-physics gate at pre-action f+1; stop reward at first original failure, retain full physical failed tail',
                'original_reward_only':True,'auto_reset':False,'post_initialization_resets':reset_calls,
                'completed_reference_clocks_held_only_after_their_metric_window':True,'no_controlled_robot_state_writes_after_t0':True,
                'wall_seconds':time.perf_counter()-start,'groups':aggregate,'clips':rows}
        (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print('LIBRARY_PHYSICAL_COMPLETE '+json.dumps(aggregate),flush=True)
    finally:env.close();env.backend.close()


if __name__=='__main__':main()
