"""Finite2-clip control, with identical per-motion contracts and explicit denominator1."""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime,timezone
import psutil

ROOT=Path(__file__).resolve().parents[1]
def read(name):return json.loads((ROOT/name).read_text())
def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    plan=read('evidence/specialist-teacher-8808-1030-plan.json');out=ROOT/'evidence/specialist-teacher-8808-1030.json'
    assert not out.exists()
    process=psutil.Process();process.cpu_affinity(process.cpu_affinity()[:2]);start=time.monotonic();completed=[]
    receipt={'status':'running','started_at':datetime.now(timezone.utc).isoformat(),'plan_sha256':sha(ROOT/'evidence/specialist-teacher-8808-1030-plan.json'),
             'GPU':0,'training_updates':0,'maximum_seconds':300,'CPU_affinity':process.cpu_affinity(),'commands':completed,'groups':{},
             'student_model_replaced':False,'unified_goal_complete':False,'architecture_confound':'Historical99D single-motion specialist versus101D stage2 teacher; descriptive only.'}
    def save():
        receipt.update(updated_at=datetime.now(timezone.utc).isoformat(),wall_seconds=time.monotonic()-start)
        out.write_text(json.dumps(receipt,indent=2)+'\n')
    def run(script,args,render=False):
        remaining=300-(time.monotonic()-start);assert remaining>0
        env=os.environ.copy();env.update(MUJOCO_GL='glfw' if render else 'disable',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',HF_HUB_OFFLINE='1',WANDB_MODE='offline')
        log=ROOT/f'evidence/specialist-teacher-8808-1030-{len(completed):02d}.log'
        with log.open('x',encoding='utf8') as stream:
            subprocess.run([sys.executable,'-X','utf8','-u',str(ROOT/'scripts'/script),*args],env=env,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=remaining)
        completed.append({'script':script,'arguments':args,'log':log.relative_to(ROOT).as_posix()});save()
    save()
    try:
        for variant,mode in plan['modes'].items():
            assert not (ROOT/f'runs/{mode}').exists()
            run('evaluate_frozen_specialist1030.py',[variant])
            for script in ['assess_tracking.py','audit_physical_contacts.py']:run(script,['--run',mode])
            tracking=read(f'runs/{mode}/tracking-assessment.json');contact=read(f'runs/{mode}/contact-review.json')
            samples=read(f'runs/{mode}/contact-samples.json');contract=read('contact-acceptance-contract.json')
            assert tracking['new_denominator']==contact['new_denominator']==len(samples)==1
            row=tracking['clips'][0];records=samples[0]['samples'];window=round(contract['reference_expected_contact_time_tolerance_seconds']/.02);unexpected=[]
            for i,record in enumerate(records):
                expected={body for near in records[max(0,i-window):i+window+1] for body in near['reference_ground_bodies']}
                actual={body for body in record['actual_ground_bodies'] if any(word in body.lower() for word in ['head','torso','pelvis'])}
                if actual-expected:unexpected.append({'time_s':record['time_s'],'bodies':sorted(actual-expected)})
            checks={'no_unexpected_trunk_contact':len(unexpected)<=contract['unexpected_head_torso_or_pelvis_ground_contact_samples_max'],
                    'penetration_within_engineering_limit':contact['clips'][0]['actual_min_after_initial_100ms_m']>=-contract['actual_ground_penetration_after_initial_100ms_max_m']}
            combined={'clip':row['clip'],'denominator':1,'tracking_quality_passed':row['tracking_quality_passed'],'contact_review_passed':all(checks.values()),
                      'combined_engineering_passed':row['tracking_quality_passed'] and all(checks.values()),'tracking_failed_checks':row['failed_checks'],
                      'contact_checks':checks,'unexpected_trunk_contacts':unexpected,'actual_min_after_initial_100ms_m':contact['clips'][0]['actual_min_after_initial_100ms_m'],
                      'norm_return_original':row['norm_return_original'],'tracking_contract_sha256':tracking['contract_sha256'],
                      'contact_contract_sha256':sha(ROOT/'contact-acceptance-contract.json'),'one_motion_diagnostic_only':True,'not_a15_or55_clip_acceptance':True}
            (ROOT/f'runs/{mode}/single-clip-combined-assessment.json').write_text(json.dumps(combined,indent=2)+'\n')
            run('render_teacher_qualification.py',['--run',mode],render=True)
            video=read(f'videos/{mode}/{plan["clip"]}.json');assert video['complete_decode_passed'] and sha(ROOT/video['file'])==video['sha256']
            receipt['groups'][variant]={'assessment':combined,'video':video};save()
        assert sha((ROOT/plan['checkpoint']).resolve())==plan['checkpoint_sha256']
        for name,digest in plan['source_sha256'].items():assert sha(ROOT/name)==digest,name
        receipt.update(status='complete_two_full_clip_specialist_controls',checkpoint_sha256=plan['checkpoint_sha256'],
                       qualified_for_8808_under_both_references=all(group['assessment']['combined_engineering_passed'] for group in receipt['groups'].values()),
                       all_failed_tails_retained=True,next_GPU_training_automatically_authorized=False);save();print(json.dumps(receipt,indent=2))
    except BaseException as error:
        receipt.update(status='failed_or_interrupted_partial_evidence_retained',error=repr(error));save();raise

if __name__=='__main__':main()
