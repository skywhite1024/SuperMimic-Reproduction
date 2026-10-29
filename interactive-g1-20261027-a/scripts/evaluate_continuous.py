"""Actual physical closed-loop trials. Only the initial state may be reset."""
import argparse
import hashlib
import json
import time
import numpy as np
import torch
from runtime import ROOT, CHECKPOINT, make_env, load_policy
from unisupermimic.rl.algos.supermimic.geometry import unflatten_state, yaw_mat
from unisupermimic.rl.contracts.supermimic import SUPERMIMIC_STATE_KEY


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="baseline")
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument("--checkpoint",type=__import__('pathlib').Path)
    parser.add_argument("--seeds",type=int,nargs='+',default=[0,1,2])
    parser.add_argument("--reference-set",default="references")
    args = parser.parse_args()
    out = ROOT / "runs" / args.mode
    if (out / "summary.json").exists():
        raise FileExistsError("Preserve existing results; choose a new experiment label")
    out.mkdir(parents=True, exist_ok=True)
    seeds = args.seeds
    files = [ROOT / args.reference_set / f"seed-{seed}/sequence.csv" for seed in seeds]
    manifests = [json.loads(p.with_name("manifest.json").read_text()) for p in files]
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == m['reference_csv_sha256'] for p,m in zip(files,manifests))
    start = time.perf_counter()
    checkpoint = (args.checkpoint or CHECKPOINT).resolve()
    header = torch.load(checkpoint,map_location='cpu',weights_only=True)
    if header['pi']['obs_norm.mean'].numel()==101:
        from global_model import GlobalPositionMDP
        from unisupermimic.tasks.motion_tracking.g1.supermimic import manager_terms
        manager_terms.MDP=GlobalPositionMDP
    del header
    env = make_env(files, fps=50.0, continuous=True)
    pi, ckpt = load_policy(env,checkpoint)
    mdp, motion = env.supermimic_mdp, env.supermimic_mdp.motion
    _, info = env.reset_to_frames(motion.clip_start.clone())
    cur = info[SUPERMIMIC_STATE_KEY]
    init_state = cur['s'].clone()
    reset_calls = 0
    def forbidden_reset(*a, **kw):
        nonlocal reset_calls
        reset_calls += 1
        raise RuntimeError("Robot-state reset is forbidden after t=0 in this experiment")
    env.reset = forbidden_reset
    history = None
    records = {key: [] for key in ['state','reference_qpos','xy_error','yaw_error','body_global','body_local','hard_failure','floor_fall','action']}
    boundary_probes = []
    # A diagnostic trial continues physics after the original hard gate. Its failures
    # remain measured with the unchanged original thresholds and are never cleared.
    try:
        with torch.inference_mode():
            for step in range(round(args.seconds * 50) + 1):
                s = unflatten_state(cur['s']); frames = cur['f']
                refp, refR = motion.root_p[frames], motion.root_R[frames]
                body = mdp.kin.forward(*s)[0]
                local_delta = yaw_mat(s[1] @ refR.transpose(-1,-2))
                anchor = torch.cat((s[0][:,:2],refp[:,2:3]),-1)
                local_ref = anchor[:,None] + (local_delta[:,None] @ (motion.body_p[frames]-refp[:,None])[...,None]).squeeze(-1)
                xy_error = (s[0][:,:2]-refp[:,:2]).norm(dim=-1)
                yaw = torch.atan2(s[1][:,1,0],s[1][:,0,0])-torch.atan2(refR[:,1,0],refR[:,0,0])
                yaw = torch.atan2(torch.sin(yaw),torch.cos(yaw)).abs()
                values = {'state':cur['s'],'reference_qpos':torch.cat((refp,motion.root_R[frames].reshape(len(seeds),9),motion.q[frames]),-1),
                          'xy_error':xy_error,'yaw_error':yaw,'body_global':(body-motion.body_p[frames]).norm(dim=-1).mean(-1),
                          'body_local':(body-local_ref).norm(dim=-1).mean(-1),'hard_failure':mdp.termination(s,frames),'floor_fall':s[0][:,2]<.35}
                for key,value in values.items():
                    records[key].append(value.cpu().numpy().copy())
                if not torch.isfinite(cur['s']).all():
                    raise FloatingPointError(f"Nonfinite state at {step/50}s")
                if step == round(args.seconds*50):
                    break
                obs = mdp.obs(s,cur['a_prev'],cur['xi'],frames)
                history = obs[:,None].repeat(1,ckpt['args']['hist'],1) if history is None else torch.cat((history[:,1:],obs[:,None]),1)
                goals = mdp.goal(frames,ckpt['args']['goal_frames'])
                if args.mode == 'goal-feedback':
                    # Feed fixed-world root position error into the future displacement
                    # goal. The reference poses themselves are never moved or realigned.
                    error = (yaw_mat(refR).transpose(-1,-2) @ (refp-s[0])[...,None]).squeeze(-1)
                    error[:,2] = 0
                    error = error.clamp(-.25,.25)
                    goals = goals.reshape(len(seeds),ckpt['args']['goal_frames'],73).clone()
                    goals[:,:,:3] += error[:,None]
                    goals = goals.reshape(len(seeds),-1)
                action = pi(history,goals)
                assert torch.isfinite(action).all()
                records['action'].append(action.cpu().numpy().copy())
                state = env.step(action)
                assert not bool(state.terminated.any() or state.truncated.any()),'Manager termination must not cause physical resets'
                cur = state.info[SUPERMIMIC_STATE_KEY]
                if step % 250 == 0:
                    print('CONTINUOUS_PROGRESS '+json.dumps({'mode':args.mode,'seconds':step/50,'xy_m':xy_error.tolist(),'hard_failure':values['hard_failure'].tolist(),'reset_count':reset_calls}),flush=True)
        packed = {key:np.stack(value) for key,value in records.items()}
        np.savez_compressed(out/'trajectory.npz',**packed)
        summaries = []
        for i,seed in enumerate(seeds):
            fail = packed['hard_failure'][:,i]
            floor = packed['floor_fall'][:,i]
            switch_rows=[]
            for seg in manifests[i]['segments'][1:]:
                boundary=seg['start_frame'];lo=max(0,boundary-30);hi=min(len(fail),boundary+60)
                switch_rows.append({'segment':seg['segment'],'time':boundary/50,'xy_peak_m':float(packed['xy_error'][lo:hi,i].max()),'body_global_peak_m':float(packed['body_global'][lo:hi,i].max()),'hard_failure_in_window':bool(fail[lo:hi].any())})
            summaries.append({'seed':seed,'planned_seconds':args.seconds,'elapsed_seconds':(len(fail)-1)/50,'segments':29,'post_initialization_resets':reset_calls,
                              'first_hard_failure_s':float(np.flatnonzero(fail)[0]/50) if fail.any() else None,
                              'hard_failure_frames':int(fail.sum()),'hard_failure_entries':int(np.sum(fail & ~np.r_[False,fail[:-1]])),
                              'floor_fall_entries':int(np.sum(floor & ~np.r_[False,floor[:-1]])),
                              'root_xy_rmse_m':float(np.sqrt(np.mean(packed['xy_error'][:,i]**2))),
                              'root_xy_peak_m':float(packed['xy_error'][:,i].max()),'yaw_rmse_deg':float(np.rad2deg(np.sqrt(np.mean(packed['yaw_error'][:,i]**2)))),
                              'body_global_mean_m':float(packed['body_global'][:,i].mean()),'body_local_mean_m':float(packed['body_local'][:,i].mean()),
                              'passed_continuous_no_hard_failure':not bool(fail.any()),'switch_windows':switch_rows})
        result={'mode':args.mode,'checkpoint':str(checkpoint),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                'reference_set':args.reference_set,
                'backend':'MuJoCo 3.11 CPU / mjbatch-uni, adapted from the existing mjwarp collector; this is a new backend test',
                'observation_dim':mdp.obs_dim,'goal_dim':mdp.goal_frame_dim*ckpt['args']['goal_frames'],
                'simulation_dt':.005,'control_dt':.02,'wall_seconds':time.perf_counter()-start,'trials':summaries,
                'reset_policy':'one initialization at t=0 only; env.reset guarded thereafter; zero reference-to-robot state writes in the loop',
                'hard_gate':'unchanged abs(height-reference)>0.25 or abs(Rzz-reference)>0.8; independently recorded, manager terminal disabled only to retain the full post-failure physical trajectory',
                'goal_feedback':{'error_clamp_m':.25,'reference_moved':False,'policy_weights_changed':False,'semantics_changed':True} if args.mode=='goal-feedback' else None,
                'physics_not_reference_playback':True,'flip_coverage':False}
        (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
        print('CONTINUOUS_TRIALS_COMPLETE '+json.dumps({'mode':args.mode,'trials':[{k:v for k,v in row.items() if k!='switch_windows'} for row in summaries]}),flush=True)
    finally:
        env.close();env.backend.close()


if __name__=='__main__':
    main()
