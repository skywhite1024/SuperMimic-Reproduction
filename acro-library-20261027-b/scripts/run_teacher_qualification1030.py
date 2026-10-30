"""Finite zero-training CPU qualification, including all30 decoded full-tail videos."""
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import psutil

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT/'evidence/teacher-qualification-1030-execution.json'
MODES = {variant:f'teacher-stage2-bound20-{variant}-cpu-new15-1030' for variant in ['raw','corrected']}

def read(name):
    return json.loads((ROOT/name).read_text(encoding='utf8'))

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    assert not RECEIPT.exists(), 'Preserve each execution attempt; do not repeat blindly'
    plan_path = ROOT/'evidence/retention-1030-next-diagnostic-plan.json'
    plan = read(str(plan_path.relative_to(ROOT)))
    assert read('evidence/retention-1030-Git-publication.json')['publication_complete']
    assert plan['GPU'] == 0 and plan['new_training_updates'] == 0
    checkpoint = ROOT/plan['checkpoint']
    assert sha(checkpoint) == plan['checkpoint_sha256']
    frozen = read('retention-trial-v1-plan.json')['files']
    for name,digest in frozen.items():
        assert sha(ROOT/name) == digest, name
    for mode in MODES.values():
        assert not (ROOT/'runs'/mode).exists() and not (ROOT/'videos'/mode).exists(), mode
    process = psutil.Process()
    affinity = process.cpu_affinity()[:plan['maximum_local_CPU_threads']]
    process.cpu_affinity(affinity)
    assert len(process.cpu_affinity()) <= 2
    start = time.monotonic()
    commands = []
    result = {'status':'running','started_at':datetime.now(timezone.utc).isoformat(),
              'plan_sha256':sha(plan_path),'checkpoint_sha256':sha(checkpoint),'GPU':0,'new_training_updates':0,
              'maximum_program_seconds':plan['maximum_program_seconds'],'CPU_affinity':affinity,
              'maximum_library_threads':2,'modes':MODES,'commands_completed':commands,
              'candidate_adopted':False,'full_goal_complete':False}
    def record(status, **extra):
        result.update(status=status,updated_at=datetime.now(timezone.utc).isoformat(),wall_seconds=time.monotonic()-start,**extra)
        RECEIPT.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    def run(script,argv,render=False):
        remaining=plan['maximum_program_seconds']-(time.monotonic()-start)
        assert remaining>0, 'Finite program deadline reached'
        env=os.environ.copy()
        env.update(PYTHONUTF8='1',MUJOCO_GL='glfw' if render else 'disable',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
                   MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',HF_HUB_OFFLINE='1',WANDB_MODE='offline')
        log=ROOT/f'evidence/teacher-qualification-1030-{len(commands):02d}-{script[:-3]}.log'
        command=[sys.executable,'-X','utf8','-u',str(ROOT/'scripts'/script),*argv]
        print('TEACHER_QUALIFICATION_START '+json.dumps({'script':script,'arguments':argv}),flush=True)
        with log.open('x',encoding='utf8') as stream:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            try:
                assert len(psutil.Process(child.pid).cpu_affinity())<=2
                code=child.wait(timeout=remaining)
                if code:
                    raise subprocess.CalledProcessError(code,command)
            except BaseException:
                try:
                    parent=psutil.Process(child.pid)
                    descendants=parent.children(recursive=True)
                    for item in descendants:
                        try:item.kill()
                        except psutil.NoSuchProcess:pass
                    parent.kill()
                    psutil.wait_procs([parent,*descendants],timeout=5)
                except psutil.NoSuchProcess:pass
                raise
        commands.append({'script':script,'arguments':argv,'log':str(log.relative_to(ROOT))})
        record('running')
        print('TEACHER_QUALIFICATION_END '+json.dumps(commands[-1]),flush=True)
    record('running')
    try:
        for variant,mode in MODES.items():
            run('evaluate_teacher_bound20.py',['--reference-variant',variant,'--checkpoint',str(checkpoint),'--group','new','--device','cpu','--mode',mode])
            summary=read(f'runs/{mode}/summary.json')
            assert summary['checkpoint_sha256']==plan['checkpoint_sha256'] and len(summary['clips'])==15
            assert summary['post_initialization_resets']==0 and all(row['post_initialization_resets']==0 for row in summary['clips'])
            for script in ['assess_tracking.py','audit_physical_contacts.py','combine_acceptance.py']:
                run(script,['--run',mode])
        groups={}
        for variant,mode in MODES.items():
            quality=read(f'runs/{mode}/combined-assessment.json')
            protected=[row for row in quality['clips'] if row['clip'] in plan['protected_new']]
            assert len(protected)==3
            groups[variant]={'denominator':15,'original_hard_gate_passed':quality['original_hard_gate_passed'],
                             'tracking_quality_passed':quality['tracking_quality_passed'],'contact_review_passed':quality['contact_review_passed'],
                             'combined_engineering_passed':quality['combined_engineering_passed'],'protected_denominator':3,
                             'protected_strict_passed':sum(row['combined_engineering_passed'] for row in protected),'protected':protected}
        record('all30_physical_trajectories_and_quality_complete_rendering',groups=groups)
        for mode in MODES.values():
            run('render_teacher_qualification.py',['--run',mode],render=True)
        videos=[]
        for mode in MODES.values():
            receipts=sorted((ROOT/'videos'/mode).glob('*.json'))
            assert len(receipts)==15
            for path in receipts:
                value=json.loads(path.read_text(encoding='utf8'))
                assert value['complete_decode_passed'] and sha(ROOT/value['file'])==value['sha256']
                assert (ROOT/value['file']).stat().st_size==value['bytes']
                videos.append(value)
        assert sha(checkpoint)==plan['checkpoint_sha256'] and sha(plan_path)==result['plan_sha256']
        for name,digest in frozen.items():
            assert sha(ROOT/name)==digest,name
        qualification=all(group['protected_strict_passed']==3 for group in groups.values())
        record('complete_all30_full_tail_videos_decoded_hashed',groups=groups,videos=videos,video_count=len(videos),
               protected_teacher_qualified=qualification,weights_and_all_frozen_contract_source_hashes_unchanged=True,
               next_GPU_training_automatically_authorized=False)
        print('TEACHER_QUALIFICATION_COMPLETE '+json.dumps({'groups':groups,'protected_teacher_qualified':qualification,'video_count':30}),flush=True)
    except BaseException as error:
        record('failed_or_interrupted_all_partial_evidence_retained',error=repr(error))
        raise

if __name__ == '__main__':
    main()
