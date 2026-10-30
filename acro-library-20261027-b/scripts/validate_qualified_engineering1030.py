"""Check the saved6-update model through independent training/deployment policy loaders."""
import hashlib,json,time
from datetime import datetime,timezone
import numpy as np
import psutil,torch
from runtime import ROOT,make_env,references
from retention_components import load_teacher
from evaluate_action_bound import BoundedPolicy
from unisupermimic.rl.algos.supermimic.geometry import unflatten_state
from unisupermimic.rl.contracts.supermimic import SUPERMIMIC_STATE_KEY
from unisupermimic.tasks.motion_tracking.g1.supermimic.evaluate import load_policy

def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    process=psutil.Process();process.cpu_affinity(process.cpu_affinity()[:2]);torch.set_num_threads(2);start=time.monotonic()
    run=ROOT/'runs/engineering-qualified8808-cpu-6-1030';proof=json.loads((run/'proof.json').read_text());assert proof['new_updates']==6
    checkpoint=run/'final-full.pt';saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    training_load=load_teacher(checkpoint,'cpu');file=next(p for p in references('new') if p.stem=='88_08_stageii_unitree_g1')
    env=make_env([file],device='cpu',observation_dim=101,evaluation=True)
    try:
        native=load_policy(env,saved,device='cpu');deployment=BoundedPolicy(native,(env.supermimic_mdp.motion.clip_len-1).repeat(4)).eval()
        mdp=env.supermimic_mdp;_,info=env.reset_to_frames(mdp.motion.clip_start.clone());initial=info[SUPERMIMIC_STATE_KEY]
        with np.load(ROOT/'runs/specialist8808-bound20-raw-cpu-1030/trajectory.npz') as data:
            states=torch.from_numpy(data['state'][:,0].copy());actions=torch.from_numpy(data['action'][:-1,0].copy())
        selected=set(np.linspace(0,len(actions)-1,4).round().astype(int).tolist());histories=[];goals=[];history=None
        for i in range(len(actions)):
            frame=mdp.motion.clip_start+i;prev=initial['a_prev'] if i==0 else actions[i-1:i]
            obs=mdp.obs(unflatten_state(states[i:i+1]),prev,initial['xi'],frame)
            history=obs[:,None].repeat(1,saved['args']['hist'],1) if history is None else torch.cat([history[:,1:],obs[:,None]],1)
            if i in selected:histories.append(history.clone());goals.append(mdp.goal(frame,saved['args']['goal_frames']).clone())
        with torch.no_grad():
            h=torch.cat(histories);g=torch.cat(goals);expected=training_load(h,g).clamp(-20,20);actual=deployment(h,g)
        assert torch.equal(expected,actual) and torch.isfinite(actual).all() and actual.abs().max()<=20
        result={'status':'independent_saved_policy_loaders_exact_bounded_output_match','checked_at':datetime.now(timezone.utc).isoformat(),
                'checkpoint_sha256':sha(checkpoint),'proof_sha256':sha(run/'proof.json'),'real_histories':4,'source_physical_steps':sorted(selected),
                'output_max_absolute_difference':float((actual-expected).abs().max()),'exact_output_equality':True,'GPU':0,'new_training_updates':0,
                'CPU_affinity':process.cpu_affinity(),'wall_seconds':time.monotonic()-start,'physical_acceptance_inferred':False,'candidate_adopted':False}
        out=ROOT/'evidence/qualified-expert-engineering-1030-readback.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    finally:env.close();env.backend.close()

if __name__=='__main__':main()
