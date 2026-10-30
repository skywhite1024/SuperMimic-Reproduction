"""Zero-update gradient/adapter probe on the qualified specialist's real retained histories."""
import hashlib,json,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import psutil,torch
from runtime import ROOT,make_env,references
from retention_components import load_teacher,module_digest
from unisupermimic.rl.algos.supermimic.geometry import unflatten_state
from unisupermimic.rl.contracts.supermimic import SUPERMIMIC_STATE_KEY

def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    plan_path=ROOT/'evidence/qualified-teacher-adapter-1030-plan.json';plan=json.loads(plan_path.read_text())
    assert sha(Path(__file__))==plan['script_sha256']
    for name,digest in plan['input_sha256'].items():assert sha(ROOT/name)==digest,name
    out=ROOT/'evidence/qualified-teacher-adapter-1030.json';assert not out.exists()
    start=time.monotonic();process=psutil.Process();process.cpu_affinity(process.cpu_affinity()[:2]);torch.set_num_threads(2)
    specialist=json.loads((ROOT/'evidence/specialist-teacher-8808-1030.json').read_text())
    assert specialist['qualified_for_8808_under_both_references']
    teacher=load_teacher((ROOT/plan['teacher']).resolve(),'cpu');student=load_teacher(ROOT/plan['student'],'cpu')
    for parameter in student.parameters():parameter.requires_grad_(True)
    before={'teacher':module_digest(teacher),'student':module_digest(student)}
    saved=torch.load((ROOT/plan['teacher']).resolve(),map_location='cpu',weights_only=True)
    student_saved=torch.load(ROOT/plan['student'],map_location='cpu',weights_only=True)
    assert saved['args']['hist']==student_saved['args']['hist'] and saved['args']['goal_frames']==student_saved['args']['goal_frames']
    file=next(path for path in references('new') if path.stem==plan['clip'])
    env=make_env([file],device='cpu',observation_dim=101,evaluation=True)
    try:
        mdp=env.supermimic_mdp;_,info=env.reset_to_frames(mdp.motion.clip_start.clone());initial=info[SUPERMIMIC_STATE_KEY]
        with np.load(ROOT/f'runs/{plan["source_run"]}/trajectory.npz') as archive:
            states=torch.from_numpy(archive['state'][:,0].copy());actions=torch.from_numpy(archive['action'][:-1,0].copy())
        history=None;histories=[];goals=[];maximum=0.;selected=set(np.linspace(0,len(actions)-1,32).round().astype(int).tolist())
        for i in range(len(actions)):
            assert time.monotonic()-start<120
            state=unflatten_state(states[i:i+1]);frame=mdp.motion.clip_start+i;previous=initial['a_prev'] if i==0 else actions[i-1:i]
            obs=mdp.obs(state,previous,initial['xi'],frame)
            history=obs[:,None].repeat(1,saved['args']['hist'],1) if history is None else torch.cat([history[:,1:],obs[:,None]],1)
            goal=mdp.goal(frame,saved['args']['goal_frames'])
            with torch.no_grad():prediction=teacher(history[:,:,:99],goal).clamp(-20,20)
            maximum=max(maximum,float((prediction-actions[i:i+1]).abs().max()))
            if i in selected:histories.append(history.clone());goals.append(goal.clone())
        assert maximum<1e-4,maximum
        history=torch.cat(histories);goal=torch.cat(goals)
        with torch.no_grad():target=teacher(history[:,:,:99],goal).clamp(-20,20)
        raw=student(history,goal);loss=(raw-target).square().mean();assert torch.isfinite(loss)
        loss.backward();gradients=[parameter.grad for parameter in student.parameters() if parameter.grad is not None]
        assert gradients and all(torch.isfinite(value).all() for value in gradients) and any(value.abs().max()>0 for value in gradients)
        assert all(parameter.grad is None for parameter in teacher.parameters())
        after={'teacher':module_digest(teacher),'student':module_digest(student)};assert before==after
        for name,digest in plan['input_sha256'].items():assert sha(ROOT/name)==digest,name
        result={'status':'complete_real_history_99D_teacher_adapter_and_finite_student_gradient_verified','checked_at':datetime.now(timezone.utc).isoformat(),
                'plan_sha256':sha(plan_path),'CPU_affinity':process.cpu_affinity(),'wall_seconds':time.monotonic()-start,'GPU':0,'optimizer_steps':0,'training_updates':0,
                'retained_physical_steps_checked':len(actions),'real_histories_in_gradient_probe':len(histories),'maximum_teacher_action_reconstruction_abs_error':maximum,
                'unweighted_teacher_MSE_at_parent_states':float(loss.detach()),'teacher_gradients_absent':True,'student_gradient_tensors':len(gradients),
                'all_weights_and_norms_unchanged':True,'module_digests_before':before,'module_digests_after':after,'selected_protected_clip':plan['clip'],
                'unqualified_8807_and_8809_not_imitation_targets':True,'deployed_student_policies':1,'deployment_uses_teacher':False,
                'physical_improvement_not_inferred':True,'full_goal_complete':False,'new_GPU_job_automatically_authorized':False}
        out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    finally:env.close();env.backend.close()

if __name__=='__main__':main()
