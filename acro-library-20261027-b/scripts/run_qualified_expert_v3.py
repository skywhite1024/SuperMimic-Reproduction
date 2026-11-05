"""Prepare/verify locally; execute one finite CUDA hypothesis only with its explicit flag."""
import argparse,hashlib,json,os,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def read(name):return json.loads((ROOT/name).read_text())

def one_clip_gate(mode):
    tracking=read(f'runs/{mode}/tracking-assessment.json');contact=read(f'runs/{mode}/contact-review.json');samples=read(f'runs/{mode}/contact-samples.json')
    contract=read('contact-acceptance-contract.json');assert tracking['new_denominator']==contact['new_denominator']==len(samples)==1
    records=samples[0]['samples'];window=round(contract['reference_expected_contact_time_tolerance_seconds']/.02);unexpected=0
    for i,record in enumerate(records):
        expected={body for near in records[max(0,i-window):i+window+1] for body in near['reference_ground_bodies']}
        actual={body for body in record['actual_ground_bodies'] if any(word in body.lower() for word in ['head','torso','pelvis'])}
        unexpected+=bool(actual-expected)
    return bool(tracking['clips'][0]['tracking_quality_passed'] and unexpected<=contract['unexpected_head_torso_or_pelvis_ground_contact_samples_max'] and contact['clips'][0]['actual_min_after_initial_100ms_m']>=-contract['actual_ground_penetration_after_initial_100ms_max_m'])

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--verify-only',action='store_true');parser.add_argument('--local-teacher-control',action='store_true');parser.add_argument('--execute-reviewed-plan',action='store_true');args=parser.parse_args()
    assert sum([args.verify_only,args.local_teacher_control,args.execute_reviewed_plan])==1
    plan_path=ROOT/'qualified-expert-v3-plan.json';plan=read(plan_path.name)
    cloud=args.execute_reviewed_plan
    for name,digest in plan['files'].items():assert sha(ROOT/name)==digest,name
    for key in ['initial_full','old_teacher','new_teacher']:
        path=Path(plan[key+'_cloud']) if cloud else (ROOT/plan[key+'_local']).resolve()
        assert sha(path)==plan[key+'_sha256'],key
    for name,digest in plan['matched_parent_result_sha256'].items():assert sha(ROOT/name)==digest,name
    proof=read(plan['six_update_engineering'])
    assert proof['new_updates']==6 and proof['normalizer_tensors_unchanged'] and proof['retention']['teacher_parameters_and_norms_unchanged']
    assert proof['old_teacher_weight']==.2 and proof['new_teacher_weight']==.05 and proof['functional_bounded_policy_readback']['exact_output_equality']
    assert read(plan['independent_readback'])['exact_output_equality']
    if args.verify_only:
        print(json.dumps({'status':'new_hypothesis_source_models_parent_controls_CPU6proof_and_readback_verified','files':len(plan['files']),'GPU_requested':0,'plan_sha256':sha(plan_path)},indent=2));return
    device='cuda' if cloud else 'cpu'
    if cloud:
        assert ROOT.resolve().is_relative_to(Path('/mnt/afs/USER_ID'))
        import torch
        assert torch.cuda.device_count()==1 and 'H100' in torch.cuda.get_device_name(0)
    else:
        import psutil
        process=psutil.Process();process.cpu_affinity(process.cpu_affinity()[:2])
    start=time.monotonic();budget=5400 if cloud else 120;commands=[]
    receipt_path=ROOT/('evidence/qualified-expert-v3-CUDA-execution.json' if cloud else 'evidence/qualified-expert-v3-local-control.json');assert not receipt_path.exists()
    result={'status':'running','started_at':datetime.now(timezone.utc).isoformat(),'plan_sha256':sha(plan_path),'device':device,'maximum_seconds':budget,'new_training_updates':0,'commands':commands,'full_goal_complete':False,'candidate_adopted':False}
    def save(status,**extra):
        result.update(status=status,wall_seconds=time.monotonic()-start,updated_at=datetime.now(timezone.utc).isoformat(),**extra);receipt_path.write_text(json.dumps(result,indent=2)+'\n')
    def run(script,argv):
        remaining=budget-(time.monotonic()-start);assert remaining>0
        env=os.environ.copy();env.update(MUJOCO_GL='disable',PYTHONUTF8='1',OMP_NUM_THREADS='4' if cloud else '2',OPENBLAS_NUM_THREADS='4' if cloud else '2',WANDB_MODE='offline',HF_HUB_OFFLINE='1')
        log=ROOT/f'evidence/qualified-expert-v3-{device}-{len(commands):02d}.log'
        with log.open('x',encoding='utf8') as stream:subprocess.run([sys.executable,'-X','utf8','-u',str(ROOT/'scripts'/script),*argv],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=remaining)
        commands.append({'script':script,'arguments':argv,'log':log.relative_to(ROOT).as_posix()});save('running')
    save('running')
    try:
        teacher=Path(plan['new_teacher_cloud']) if cloud else (ROOT/plan['new_teacher_local']).resolve();controls={}
        for variant in ['raw','corrected']:
            mode=f'qualified-expert-v3-teacher-{variant}-{device}-8808'
            run('evaluate_qualified_expert_v3.py',['--role','teacher','--suite','library','--reference-variant',variant,'--checkpoint',str(teacher),'--device',device,'--mode',mode,'--group','new','--clips','88_08_stageii_unitree_g1'])
            for script in ['assess_tracking.py','audit_physical_contacts.py']:run(script,['--run',mode])
            controls[variant]=one_clip_gate(mode)
        if not all(controls.values()):save('teacher_pretraining_control_failed_no_training',teacher_controls=controls);return
        if not cloud:save('complete_local_full_trial_teacher_gate_passed_no_training',teacher_controls=controls);return
        gate=ROOT/'evidence/qualified-expert-v3-CUDA-pretraining-qualification.json';assert not gate.exists()
        gate.write_text(json.dumps({'status':'pass','plan_sha256':sha(plan_path),'teacher_sha256':sha(teacher),'controls':controls,'training_updates_at_gate':0},indent=2)+'\n')
        label='unified-qualified-expert-v3-h100-1000'
        run('train_qualified_expert_v3.py',['--initial',plan['initial_full_cloud'],'--label',label,'--device','cuda','--steps','1000','--wm-warmup','200','--num-envs','1024'])
        training=read(f'runs/{label}/proof.json');assert training['new_updates']==1000 and training['functional_bounded_policy_readback']['exact_output_equality'];result['new_training_updates']=1000
        checkpoint=ROOT/f'runs/{label}/final-full.pt';counts={};sequence_tracking={};sequence_contact={}
        for variant,suite,group in [('raw','library','new'),('corrected','library','all'),('raw','sequences',None),('corrected','sequences',None)]:
            mode=f'qualified-expert-v3-{variant}-{suite}-cuda';argv=['--role','candidate','--suite',suite,'--reference-variant',variant,'--checkpoint',str(checkpoint),'--device','cuda','--mode',mode]
            if group:argv+=['--group',group]
            run('evaluate_qualified_expert_v3.py',argv)
            scripts=['assess_tracking.py','audit_physical_contacts.py','combine_acceptance.py'] if suite=='library' else ['assess_sequences.py','audit_sequence_contacts.py']
            for script in scripts:run(script,['--run',mode])
            q=read(f'runs/{mode}/'+('combined-assessment.json' if suite=='library' else 'continuous-assessment.json'))
            if suite=='library':counts[variant+'_'+suite]=q['combined_engineering_passed']
            else:
                contacts=read(f'runs/{mode}/continuous-contact-review.json');by_seed={x['seed']:x for x in contacts['trials']}
                sequence_tracking[variant]=q['continuous_engineering_passed'];sequence_contact[variant]=contacts['contact_review_passed']
                counts[variant+'_'+suite]=sum(x['continuous_engineering_passed'] and by_seed[x['seed']]['contact_review_passed'] for x in q['trials'])
        old=read('runs/qualified-expert-v3-corrected-library-cuda/summary.json')['groups']['lafan']['passed']
        adoption=counts['corrected_library']>=10 and old>=39 and (counts['corrected_library']>10 or old>39) and counts['raw_library']>=9 and counts['raw_sequences']>=1 and counts['corrected_sequences']>=1
        save('finite_qualified_expert_v3_candidate_and_all_CUDA_assessments_complete',counts=counts,sequence_tracking_counts=sequence_tracking,sequence_contact_counts=sequence_contact,sequence_counts_require_tracking_and_contact=True,old_original_gate_passed=old,predeclared_improvement_condition_passed=adoption,
             candidate_adoption_pending_all_full_acceptance=True,candidate_adopted=False,checkpoint_sha256=sha(checkpoint),automatic_extension_allowed=False)
    except BaseException as error:
        save('failed_or_interrupted_all_partial_evidence_retained',error=repr(error));raise

if __name__=='__main__':main()
