"""One finite1000-update candidate, matched bound20 controls, and fixed assessments."""
import hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def operational_preflight(plan):
    import importlib.metadata
    operation=json.loads((ROOT/'retention-resume-1030b-plan.json').read_text())
    for name,digest in operation['files'].items():assert sha(ROOT/name)==digest,name
    dependency=json.loads((ROOT/'evidence/retention-1030-dependency-preflight.json').read_text())
    assert dependency['status']=='all_dependencies_inputs_sequence_constructors_and_parent_contact_audit_passed'
    assert dependency['training_updates_performed']==0 and dependency['six_exact_fixed_sequences_verified']
    assert dependency['packages_before']==dependency['packages_after']
    for name,version in dependency['packages_after'].items():assert importlib.metadata.version(name)==version,name
    assert importlib.metadata.version('scipy')=='1.17.0'
    from scipy.spatial.transform import Rotation
    assert Rotation.from_quat([0,0,0,1]).as_matrix().shape==(3,3)
    recovery=json.loads((ROOT/'retention-recovery-1030-plan.json').read_text())
    assert sha(ROOT/'retention-recovery-1030-plan.json')==dependency['recovery_plan_sha256']
    for name,digest in recovery['files'].items():assert sha(ROOT/name)==digest,name
    previous=json.loads((ROOT/'evidence/gpu-retention-v1-1000.json').read_text())
    assert previous['status']=='failed_or_interrupted' and 'audit_physical_contacts.py' in previous['error']
    assert previous['wall_seconds']==operation['previous_program_wall_seconds']
    assert operation['maximum_new_updates']==1000 and operation['new_program_seconds_max']==4200
    assert operation['previous_program_wall_seconds']+operation['new_program_seconds_max']<=5400
    parent=json.loads((ROOT/'runs/retention-v1-parent-bound20-corrected-cuda-all55/summary.json').read_text())
    assert parent['checkpoint_sha256']==plan['initial_full_sha256']
    assert parent['device']=='cuda' and parent['observation_dim']==101 and parent['one_policy']
    assert len(parent['clips'])==55 and parent['groups']['new']['denominator']==15 and parent['groups']['lafan']['denominator']==40
    assert parent['controller_variant']=='action-output-bound-20-v1' and parent['reference_variant']=='ground-calibration-v1'
    assert not parent['auto_reset'] and parent['post_initialization_resets']==0 and parent['no_controlled_robot_state_writes_after_t0']
    combined=json.loads((ROOT/'runs/retention-v1-parent-bound20-corrected-cuda-all55/combined-assessment.json').read_text())
    assert combined['new_denominator']==15 and combined['combined_engineering_passed']==10
    assert sha(ROOT/'runs/retention-v1-parent-bound20-corrected-cuda-all55/summary.json')==dependency['parent55_summary_sha256']
    assert sha(ROOT/'runs/retention-v1-parent-bound20-corrected-cuda-all55/trajectory.npz')==dependency['parent55_trace_sha256']
    assert not (ROOT/'runs/unified-retention-v1-h100-1000').exists(),'Never duplicate any candidate updates'
    for mode in ['retention-v1-parent-bound20-raw-cuda-new15','retention-v1-parent-bound20-raw-continuous-cuda',
                 'retention-v1-parent-bound20-corrected-continuous-cuda','retention-v1-corrected-cuda-all55',
                 'retention-v1-raw-cuda-new15','retention-v1-raw-continuous-cuda','retention-v1-corrected-continuous-cuda']:
        assert not (ROOT/'runs'/mode).exists(),f'Inspect any unknown outcome before retry: {mode}'
    return operation

def main():
    started=time.monotonic();plan=json.loads((ROOT/'retention-trial-v1-plan.json').read_text())
    assert plan['status']=='frozen_before_v2_engineering_and_cuda_candidate_assessment'
    operation=operational_preflight(plan)
    for name,expected in plan['files'].items():assert sha(ROOT/name)==expected,name
    engineering=json.loads((ROOT/'runs/engineering-retention-v2-cpu-6/proof.json').read_text())
    assert engineering['new_updates']==6 and engineering['functional_bounded_policy_readback']['exact_output_equality']
    assert engineering['retention']['teacher_parameters_and_norms_unchanged'] and engineering['normalizer_tensors_unchanged']
    initial=ROOT/plan['initial_full'];assert sha(initial)==plan['initial_full_sha256']==engineering['parent_sha256']
    import torch
    assert torch.cuda.device_count()==1 and 'H100' in torch.cuda.get_device_name(0)
    out=ROOT/'runs/unified-retention-v1-h100-1000';assert not (out/'args.json').exists(),'Inspect existing candidate before any retry'
    receipt_path=ROOT/'evidence/gpu-retention-v1-1000-resume1030b.json'
    assert not receipt_path.exists(),'Inspect unknown outcome; never duplicate a submission'
    receipt={'status':'running','plan_sha256':sha(ROOT/'retention-trial-v1-plan.json'),
             'config':{'gpu_count':1,'gpu_name':torch.cuda.get_device_name(0),'cpu_max':Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
                       'memory_max':Path('/sys/fs/cgroup/memory.max').read_text().strip(),'maximum_seconds':4200,'new_updates':1000,'wm_only_updates':200,
                       'num_envs':1024,'new_denominator':15,'old_denominator':40,'continuous_denominator':3,'seconds_per_trial':120},
             'controller_variant':'consistent-output-bound-20-retention-v1','full_goal_complete':False,'operational_resume_plan_sha256':sha(ROOT/'retention-resume-1030b-plan.json'),'previous_program_wall_seconds':operation['previous_program_wall_seconds'],'completed_parent55_reused_without_resimulation':True}
    def save():receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    def run(script,args,limit):
        remaining=min(4200,5400-operation['previous_program_wall_seconds'])-60-(time.monotonic()-started);assert remaining>0,'Finite program deadline reached'
        print('RETENTION_TRIAL_COMMAND '+json.dumps({'script':script,'arguments':args}),flush=True)
        subprocess.run([sys.executable,'-u',str(ROOT/'scripts'/script),*args],check=True,timeout=min(limit,remaining))
    def assess_library(mode):
        for script in ['assess_tracking.py','audit_physical_contacts.py','combine_acceptance.py']:run(script,['--run',mode],240)
    def evaluate(checkpoint,mode,variant,suite='library',parent=False):
        argv=['--suite',suite,'--reference-variant',variant,'--checkpoint',str(checkpoint),'--mode',mode,'--device','cuda']
        if suite=='library':argv+=['--group','all' if variant=='corrected' else 'new']
        if parent:argv.append('--parent-control')
        run('evaluate_declared_bound.py',argv,900)
        if suite=='library':assess_library(mode)
        else:
            run('assess_sequences.py',['--run',mode],120)
            run('audit_sequence_contacts.py',['--run',mode],240)
    save();print('RETENTION_TRIAL_START '+json.dumps(receipt),flush=True)
    try:
        print('RETENTION_COMPLETED_PARENT55_REUSED '+json.dumps({'files':operation['reused_parent_artifacts'],'source_failed_attempt':'d6a6cfc61c094fac88d66b88fabf554e'}),flush=True)
        evaluate(initial,'retention-v1-parent-bound20-raw-cuda-new15','raw',parent=True)
        for variant in ['raw','corrected']:
            evaluate(initial,f'retention-v1-parent-bound20-{variant}-continuous-cuda',variant,'sequences',True)
        receipt['status']='matched_parent_controls_complete';save()
        run('train_retention_trial.py',['--initial',str(initial),'--label','unified-retention-v1-h100-1000','--device','cuda','--steps','1000','--wm-warmup','200','--num-envs','1024'],3000)
        proof=json.loads((out/'proof.json').read_text());assert proof['new_updates']==1000 and proof['functional_bounded_policy_readback']['exact_output_equality']
        receipt['status']='1000_updates_and_readback_complete';receipt['full_checkpoint_sha256']=sha(out/'final-full.pt');save()
        full=out/'final-full.pt'
        evaluate(full,'retention-v1-corrected-cuda-all55','corrected')
        evaluate(full,'retention-v1-raw-cuda-new15','raw')
        for variant in ['raw','corrected']:
            evaluate(full,f'retention-v1-{variant}-continuous-cuda',variant,'sequences')
        parent=json.loads((ROOT/'runs/retention-v1-parent-bound20-corrected-cuda-all55/summary.json').read_text())
        candidate=json.loads((ROOT/'runs/retention-v1-corrected-cuda-all55/summary.json').read_text())
        before=json.loads((ROOT/'runs/retention-v1-parent-bound20-corrected-cuda-all55/combined-assessment.json').read_text())
        after=json.loads((ROOT/'runs/retention-v1-corrected-cuda-all55/combined-assessment.json').read_text())
        parent_old,candidate_old=parent['groups']['lafan']['passed'],candidate['groups']['lafan']['passed']
        improvement=after['combined_engineering_passed']>=before['combined_engineering_passed'] and candidate_old>=parent_old and (after['combined_engineering_passed']>before['combined_engineering_passed'] or candidate_old>parent_old)
        saved=torch.load(full,map_location='cpu',weights_only=True)
        policy=ROOT/'policy/unified-retention-v1-h100-1000.pt';assert not policy.exists()
        torch.save({key:saved[key] for key in ['args','pi','variant']},policy)
        receipt.update(status='finite_candidate_and_all_cuda_assessments_complete',wall_seconds=time.monotonic()-started,
                       policy_sha256=sha(policy),combined_attempt_program_wall_seconds=operation['previous_program_wall_seconds']+time.monotonic()-started,matched_parent_groups=parent['groups'],candidate_groups=candidate['groups'],
                       matched_parent_corrected_strict=before['combined_engineering_passed'],candidate_corrected_strict=after['combined_engineering_passed'],
                       predeclared_improvement_criterion_passed=improvement,
                       classification='limited_candidate_improved_requires_matched_CPU_and_full_goal_review' if improvement else 'candidate_unsuccessful_no_automatic_extension',
                       further_training_automatically_authorized=False,all_failed_tails_retained=True,
                       matched_cpu_quality_full_video_and_recovery='pending_local_recovery_assessment')
    except BaseException as error:
        receipt.update(status='failed_or_interrupted',error=repr(error),traceback=traceback.format_exc(),wall_seconds=time.monotonic()-started)
        save();raise
    save();print('RETENTION_TRIAL_COMPLETE '+json.dumps(receipt),flush=True)

if __name__=='__main__':main()
