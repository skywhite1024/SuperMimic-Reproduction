"""Explicit allocated-H100 initialization plus the unchanged qualified V2 command.

The CPU preflight hides CUDA while one GPU remains allocated. This is not a
zero-GPU job. All initialization and scientific execution share one deadline.
"""
import argparse,hashlib,io,json,os,signal,subprocess,sys,tarfile,time
from datetime import datetime,timezone
from pathlib import Path
import qualified_expert_v2_cpu_stage as cpu

MODE='allocated-h1001-cpu4-ram64-init'
BASE_SHA='414eb6a2d1b01c6ba382d53c67ee746e88507e18802130b5a7e058e372d63ea4'
BASE_BOOT_SHA='f8d7d0406370af540b1adf5f8d4c863b4eced70903350347daa42a9b0755b620'
CPU_HELPER_SHA='077dffa11924df78db34e72f58da6d541755485290b23e24ee749d42d7860f50'
COMMAND_SHA='029cb2640baa1a7b0a42480b7cd160e3e54f225080684cbdd56d4d74988f9d33'
LABEL='qualified-expert-v2-inline-init-v1'
_CLEANUP_ABORT_HOOK=None

class Budget:
    def __init__(self,started,clock=time.monotonic):
        self.clock=clock;self.started=float(started)
        assert 0<=clock()-self.started<600,'Invalid/future/expired initialization origin'
        self.init_deadline=self.started+585;self.init_hard_deadline=self.started+600;self.work_deadline=self.started+5355
    def remaining(self,initialization=False):
        value=(self.init_deadline if initialization else self.work_deadline)-self.clock()
        if value<=0:raise TimeoutError('One shared finite job deadline reached')
        return value

def arm_initialization_timer(budget):
    """Linux work interrupt at585s; a default fatal alarm at600s bounds cleanup."""
    assert sys.platform=='linux' and hasattr(signal,'setitimer')
    previous=signal.getsignal(signal.SIGALRM)
    def expired(signum,frame):
        signal.signal(signal.SIGALRM,signal.SIG_DFL)
        signal.setitimer(signal.ITIMER_REAL,max(.001,budget.init_hard_deadline-budget.clock()))
        if _CLEANUP_ABORT_HOOK is not None:_CLEANUP_ABORT_HOOK();return
        raise TimeoutError('Initialization absolute work deadline reached; hard cutoff remains armed')
    signal.signal(signal.SIGALRM,expired)
    signal.setitimer(signal.ITIMER_REAL,budget.remaining(True))
    def cancel():signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
    return cancel

def allocated_limits(cpu_text,memory_text,mode):
    assert mode==MODE,'Explicit allocated-GPU initialization mode required'
    quota,period=cpu_text.split();assert quota!='max' and int(period)>0 and int(quota)/int(period)==4
    assert memory_text.strip()!='max' and int(memory_text)==64*1024**3
    return {'CPU':4,'memory_bytes':64*1024**3,'GPU_requested':1,'resource_mode':MODE}

def verify_probe(value):
    assert value['GPU_count']==1 and 'H100' in value['GPU_name'] and value['GPU_memory_bytes']>=79*1024**3
    return value

def verify_envelope(blob,expected,*,afs_user_root,local_workspace):
    assert cpu.digest(blob)==expected,'Inline envelope SHA mismatch'
    files=cpu.transport_files(blob)
    assert set(files)=={'base-source-only-transport.tar.gz','inline-wrapper.py','inline-preflight-helper.py.template','envelope-manifest.json'}
    meta=json.loads(files['envelope-manifest.json'])
    assert meta['schema']=='qualified-expert-v2-inline-init-v1' and meta['resource_mode']==MODE
    assert meta['base_sha256']==BASE_SHA and cpu.digest(files['base-source-only-transport.tar.gz'])==BASE_SHA
    assert meta['original_CPU_helper_sha256']==CPU_HELPER_SHA and meta['scientific_plan_sha256']==cpu.PLAN_SHA
    assert cpu.digest(files['inline-wrapper.py'])==meta['wrapper_sha256']
    helper=files['inline-preflight-helper.py.template'].replace(cpu.AFS_SLOT,afs_user_root.encode())
    assert cpu.digest(helper)==meta['restored_inline_helper_sha256']
    restored=cpu.restore_transport(files['base-source-only-transport.tar.gz'],BASE_SHA,afs_user_root=afs_user_root,local_workspace=local_workspace)
    return restored,helper,meta

def prepare_directory(own,phase,restored):
    assert phase in ('incoming','work');own=cpu.owner_directory(own)
    root=cpu.checked_path(own,own/'SuperMimic/acro-library-20261027-b');assert root.is_dir()
    paths={'own':own,'root':root,'deployment':root/'deployment','evidence':root/'evidence',
           'incoming':root/'deployment'/f'{LABEL}-incoming','work':root/'deployment'/f'{LABEL}-work',
           'receipt':root/'evidence'/f'{LABEL}-execution.json'}
    for key in ['deployment','evidence','incoming','work']:
        path=cpu.checked_path(own,paths[key],reject_leaf_link=key in ('incoming','work'))
        if os.path.lexists(path):assert path.is_dir()
    for path in [paths['receipt'],root/'evidence/qualified-expert-v2-cloud-CPU-preflight.json',
                 root/'evidence/qualified-expert-v2-CUDA-execution.json',root/'evidence/qualified-expert-v2-CUDA-pretraining-qualification.json',
                 root/'evidence/qualified-expert-v2-GPU-outer-guard.json',root/'runs/unified-qualified-expert-v2-h100-1000']:
        cpu.checked_path(own,path,reject_leaf_link=True);assert not os.path.lexists(path),'Preserve previous V2/partial evidence; no retry'
    cpu.validate_restore_targets(own,restored)
    assert not os.path.lexists(paths[phase]),'Do not reuse an initialization directory'
    if phase=='incoming':assert not os.path.lexists(paths['work'])
    for key in ['deployment','evidence']:cpu.ensure_owned_directory(own,paths[key])
    paths[phase].mkdir(mode=0o700);cpu.checked_path(own,paths[phase],reject_leaf_link=True)
    return paths

def live_group(pgid):
    """Only inspect members of our own Linux process group; ignore dead zombies."""
    assert sys.platform=='linux'
    found=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        try:
            if os.getpgid(int(entry.name))!=pgid:continue
            state=(entry/'stat').read_text().rsplit(')',1)[1].split()[0]
            if state!='Z':found.append(int(entry.name))
        except (OSError,ProcessLookupError,PermissionError):continue
    return found

def stop_group(child,grace=5):
    global _CLEANUP_ABORT_HOOK
    pgid=child.pid;signals=[];interrupted=[False]
    previous=signal.getsignal(signal.SIGTERM)
    def defer_term(signum,frame):interrupted[0]=True
    signal.signal(signal.SIGTERM,defer_term)
    previous_hook=_CLEANUP_ABORT_HOOK;_CLEANUP_ABORT_HOOK=lambda:defer_term(None,None)
    try:
        before=live_group(pgid)
        if before:
            try:os.killpg(pgid,signal.SIGTERM);signals.append('SIGTERM')
            except ProcessLookupError:pass
        until=time.monotonic()+grace
        while live_group(pgid) and time.monotonic()<until:time.sleep(.02)
        if live_group(pgid):
            try:os.killpg(pgid,signal.SIGKILL);signals.append('SIGKILL')
            except ProcessLookupError:pass
        until=time.monotonic()+2
        while live_group(pgid) and time.monotonic()<until:time.sleep(.02)
        if child.poll() is None:child.wait(timeout=1)
        after=live_group(pgid)
        assert not after,'Own child group is not empty: abort and require platform stop'
        return {'group':pgid,'members_before':before,'signals':signals,'live_members_after':after,
                'leader_exitcode':child.returncode,'interrupted_during_cleanup':interrupted[0]}
    finally:_CLEANUP_ABORT_HOOK=previous_hook;signal.signal(signal.SIGTERM,previous)

class PhaseRunner:
    def __init__(self,own,work,budget):self.own=own;self.work=work;self.budget=budget;self.events=[]
    def run(self,name,command,env,*,initialization,cap=None):
        remaining=self.budget.remaining(initialization)
        if cap is not None:remaining=min(remaining,cap)
        log=self.work/f'{name}.log';event={'phase':name,'started_at':datetime.now(timezone.utc).isoformat(),'maximum_wait_seconds':remaining,'log':str(log)}
        self.events.append(event)
        with cpu.owned_file(self.own,log,exclusive=True) as stream:
            child=subprocess.Popen(command,cwd=self.work,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                code=child.wait(timeout=remaining);event['exitcode']=code
                if code!=0:raise RuntimeError(f'Finite phase {name} exited {code}')
                self.budget.remaining(initialization)
            finally:
                event['termination']=stop_group(child);event['ended_at']=datetime.now(timezone.utc).isoformat()
                if event['termination'].get('interrupted_during_cleanup'):raise TimeoutError('Job interrupted during completed group cleanup')
        return log

def group_selftest(runner,python,env):
    # Two finite CPU hash workloads exercise a leader and descendant. No GPU work.
    inner="import hashlib,time,signal; signal.signal(signal.SIGTERM,signal.SIG_IGN); end=time.monotonic()+15\nwhile time.monotonic()<end: hashlib.sha256(b'guard-selftest'*512).digest()"
    outer="import subprocess,sys,hashlib,time,signal\nsignal.signal(signal.SIGTERM,signal.SIG_IGN)\np=subprocess.Popen([sys.executable,'-u','-c',"+repr(inner)+"])\nprint(p.pid,flush=True)\nend=time.monotonic()+15\nwhile time.monotonic()<end: hashlib.sha256(b'guard-leader'*512).digest()"
    try:runner.run('mandatory-process-group-selftest',[str(python),'-u','-c',outer],env,initialization=True,cap=.5)
    except subprocess.TimeoutExpired:
        event=runner.events[-1];assert not event['termination']['live_members_after']
        assert 'SIGKILL' in event['termination']['signals'],'KILL escalation was not exercised'
        assert len(event['termination']['members_before'])>=2,'Descendant group test was not exercised'
        return event
    raise AssertionError('Mandatory process-group deadline selftest did not exercise timeout')

def interpret_science(value):
    if value.get('status')=='teacher_pretraining_control_failed_no_training':
        assert value['new_training_updates']==0
        return {'status':'strict_CUDA_teacher_gate_failed_zero_updates','confirmed_new_updates':0}
    if value.get('status')=='finite_qualified_expert_v2_candidate_and_all_CUDA_assessments_complete':
        assert value['new_training_updates']==1000
        return {'status':'finite_scientific_driver_finished_acceptance_still_required','confirmed_new_updates':1000}
    return {'status':'scientific_driver_incomplete_partial_evidence_preserved','confirmed_new_updates':None}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute-reviewed-inline-job',action='store_true')
    parser.add_argument('--resource-mode',choices=[MODE],required=True);parser.add_argument('--envelope',type=Path,required=True)
    parser.add_argument('--expected-envelope-sha256',required=True);parser.add_argument('--afs-user-root',required=True)
    parser.add_argument('--restore-local-workspace',required=True);args=parser.parse_args()
    assert args.execute_reviewed_inline_job and sys.platform=='linux','Explicit reviewed Linux job required'
    budget=Budget(os.environ['SUPERMIMIC_JOB_STARTED_MONOTONIC'])
    cancel_initialization_timer=arm_initialization_timer(budget)
    def interrupted(signum,frame):raise TimeoutError('One finite allocated-GPU job interrupted')
    signal.signal(signal.SIGTERM,interrupted)
    own=cpu.owner_directory(Path(args.afs_user_root));cpu.checked_path(own,args.envelope,reject_leaf_link=True)
    allocated=allocated_limits(Path('/sys/fs/cgroup/cpu.max').read_text(),Path('/sys/fs/cgroup/memory.max').read_text(),args.resource_mode)
    restored,helper,meta=verify_envelope(args.envelope.read_bytes(),args.expected_envelope_sha256,afs_user_root=args.afs_user_root,local_workspace=args.restore_local_workspace)
    paths=prepare_directory(own,'work',restored);root=paths['root'];work=paths['work'];os.umask(0o077)
    result={'status':'allocated_GPU_initialization_running','maximum_total_program_seconds':5400,'initialization_maximum_seconds':600,
            'shared_budget_origin':budget.started,'resource':allocated,'GPU_already_allocated_during_CPU_preflight':True,
            'CPU_preflight_is_not_a_zero_GPU_job':True,'CPU_preflight_CUDA_VISIBLE_DEVICES':'','confirmed_new_updates':0,'scientific_driver_started':False,
            'new_GPU_job_submitted_by_this_program':False,'full_goal_complete':False,'candidate_adopted':False}
    first=True;runner=PhaseRunner(own,work,budget)
    def save():
        nonlocal first
        result.update(events=runner.events,updated_at=datetime.now(timezone.utc).isoformat(),total_wall_seconds=time.monotonic()-budget.started)
        cpu.write_owned_bytes(own,paths['receipt'],(json.dumps(result,indent=2)+'\n').encode(),exclusive=first);first=False
        print('INLINE_JOB_RECEIPT '+json.dumps(result),flush=True)
    save();base_env=os.environ.copy();cpu_env=base_env|{'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'}
    python=own/'SuperMimic/lafan1-20261024/uni/.venv/bin/python';assert python.is_file()
    try:
        result['native_group_selftest']=group_selftest(runner,python,cpu_env);save()
        probe="import torch,json\nassert torch.cuda.device_count()==1\np=torch.cuda.get_device_properties(0)\nprint('INLINE_ALLOCATION_JSON '+json.dumps({'GPU_count':1,'GPU_name':p.name,'GPU_memory_bytes':p.total_memory}),flush=True)"
        log=runner.run('actual-H100-allocation-probe',[str(python),'-u','-c',probe],base_env,initialization=True,cap=60)
        evidence=verify_probe(json.loads(next(x.split('INLINE_ALLOCATION_JSON ',1)[1] for x in log.read_text().splitlines() if x.startswith('INLINE_ALLOCATION_JSON '))))
        evidence.update(resource_mode=MODE,CPU=4,memory_bytes=64*1024**3,probe_CUDA_visible=True)
        attestation=work/'allocation-evidence.json';cpu.write_owned_bytes(own,attestation,(json.dumps(evidence,indent=2)+'\n').encode());result['allocation_evidence']=evidence;save()
        teacher=cpu.checked_path(own,root/cpu.TEACHER_PATH,reject_leaf_link=True)
        if os.path.lexists(teacher):assert cpu.file_digest(teacher)==cpu.TEACHER_SHA
        else:teacher=work/'public-teacher.pt';cpu.download_teacher(teacher,budget.init_deadline,own=own)
        archives=cpu.repack(restored,teacher,work,own=own)
        helper_path=work/'stage_qualified_expert_v2_inline.py';cpu.write_owned_bytes(own,helper_path,helper)
        cpu.write_owned_bytes(own,work/'qualified_expert_v2_cpu_stage.py',(Path(__file__).parent/'qualified_expert_v2_cpu_stage.py').read_bytes())
        command=[str(python),'-u',str(helper_path),'--archive',archives['main']['path'],'--control-archive',archives['control']['path'],
                 '--allow-manifest-identical-repack','--root',str(root),'--resource-mode',MODE,
                 '--allocation-evidence',str(attestation),'--allocation-evidence-sha256',cpu.file_digest(attestation)]
        runner.run('full-CPU-preflight-with-GPU-allocated',command,cpu_env,initialization=True)
        proof=cpu.checked_path(own,root/'evidence/qualified-expert-v2-cloud-CPU-preflight.json',reject_leaf_link=True)
        value=json.loads(proof.read_text());assert value['GPU_requested']==1 and value['CUDA_visible_devices']==0 and value['new_training_updates']==0
        budget.remaining(True);result.update(status='CPU_initialization_verified_before_frozen_CUDA_driver',CPU_preflight_sha256=cpu.file_digest(proof),actual_repacked_archives=archives);save()
        original=root/'deployment/qualified-expert-v2.command.txt';assert cpu.file_digest(original)==COMMAND_SHA
        assert cpu.file_digest(root/'qualified-expert-v2-plan.json')==cpu.PLAN_SHA
        budget.remaining(True);cancel_initialization_timer()
        result['successful_initialization_wall_seconds']=time.monotonic()-budget.started
        result.update(scientific_driver_started=True,confirmed_new_updates=None);save()
        runner.run('unchanged-reviewed-scientific-driver',['/bin/bash',str(original)],base_env,initialization=False)
        scientific=cpu.checked_path(own,root/'evidence/qualified-expert-v2-CUDA-execution.json',reject_leaf_link=True)
        result.update(interpret_science(json.loads(scientific.read_text())),scientific_receipt_sha256=cpu.file_digest(scientific))
    except BaseException as error:
        result.update(status='finite_inline_job_failed_all_partial_evidence_retained',error=repr(error));raise
    finally:
        result['actual_platform_terminal_and_resource_release_still_require_verification']=True;save()

if __name__=='__main__':main()
