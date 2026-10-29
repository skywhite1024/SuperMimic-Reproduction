"""Measure the preserved policy on the new small public acrobatic reference."""
import argparse
import hashlib
import json
import numpy as np
import torch
from runtime import ROOT,make_env,load_policy
from unisupermimic.rl.algos.supermimic.geometry import unflatten_state
from unisupermimic.rl.contracts.supermimic import SUPERMIMIC_STATE_KEY


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',default='acro-baseline');parser.add_argument('--checkpoint',type=__import__('pathlib').Path)
    args=parser.parse_args();checkpoint=args.checkpoint.resolve() if args.checkpoint else None
    out=ROOT/'runs'/args.mode;out.mkdir(exist_ok=True)
    assert not (out/'summary.json').exists(),'Preserve previous physical acceptance results'
    source=ROOT/'assets/acro-candidates/88_08_stageii_unitree_g1.csv'
    env=make_env([source],fps=30,continuous=True)
    try:
        pi,ckpt=load_policy(env,checkpoint);mdp=env.supermimic_mdp;motion=mdp.motion
        _,info=env.reset_to_frames(motion.clip_start.clone());cur=info[SUPERMIMIC_STATE_KEY]
        resets=0
        def forbidden_reset(*a,**kw):
            nonlocal resets
            resets+=1;raise RuntimeError('Acro evaluation forbids resets after t=0')
        env.reset=forbidden_reset
        history=None;states=[];failures=[];errors=[];reference=[];body_errors=[]
        with torch.inference_mode():
            for i in range(int(motion.clip_len[0])-1):
                s=unflatten_state(cur['s']);frames=cur['f']
                states.append(cur['s'][0].numpy().copy());failures.append(bool(mdp.termination(s,frames)[0]));errors.append(float((s[0]-motion.root_p[frames]).norm()))
                reference.append(torch.cat((motion.root_p[frames],motion.root_R[frames].flatten(1),motion.q[frames]),-1)[0].numpy().copy())
                body_errors.append(float((mdp.kin.forward(*s)[0]-motion.body_p[frames]).norm(dim=-1).mean()))
                obs=mdp.obs(s,cur['a_prev'],cur['xi'],frames)
                history=obs[:,None].repeat(1,ckpt['args']['hist'],1) if history is None else torch.cat((history[:,1:],obs[:,None]),1)
                action=pi(history,mdp.goal(frames,ckpt['args']['goal_frames']));assert torch.isfinite(action).all()
                state=env.step(action);assert not bool(state.terminated.any() or state.truncated.any());cur=state.info[SUPERMIMIC_STATE_KEY];assert torch.isfinite(cur['s']).all()
        packed=np.stack(states);ref=np.stack(reference)
        # The retained simulator quaternion is wxyz; this matrix entry needs no SciPy dependency.
        rzz=1-2*(packed[:,4]**2+packed[:,5]**2)
        ref_inverted=ref[:,11]<-.5
        inversion=bool((rzz[ref_inverted]<-.5).any()) if ref_inverted.any() else False
        landing=bool(np.all(rzz[-25:]>.8) and np.all(packed[-25:,2]>.55))
        np.savez_compressed(out/'trajectory.npz',state=packed,reference_qpos=ref,hard_failure=failures,root_error=errors,body_global=body_errors)
        result={'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'checkpoint':str(checkpoint) if checkpoint else 'preserved 16000','elapsed_seconds':len(states)*.02,'post_initialization_resets':resets,'hard_failure_frames':sum(failures),'first_hard_failure_s':next((i*.02 for i,x in enumerate(failures) if x),None),'max_root_error_m':max(errors),'body_global_mean_m':float(np.mean(body_errors)),'inversion_observed':inversion,'landing_last_half_second_upright':landing,'passed':not any(failures) and inversion and landing,'reference_not_BONES':True,'scope':'one new public acrobatic reference; reference-source claims and numerical audit in evidence/acro-candidate-audit.json','success_gate':'zero original hard-gate failures; robot Rzz<-0.5 during reference inversion; final 0.5s Rzz>0.8 and pelvis>0.55m; no resets'}
        (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print('ACRO_BASELINE_RESULT '+json.dumps(result),flush=True)
    finally:env.close();env.backend.close()


if __name__=='__main__':main()
