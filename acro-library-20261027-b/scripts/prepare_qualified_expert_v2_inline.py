"""Prepare a disabled inline-GPU initialization envelope; never publish or submit."""
import argparse,json,shlex,textwrap
from datetime import datetime,timezone
from pathlib import Path
import qualified_expert_v2_cpu_stage as cpu
import qualified_expert_v2_inline_init as inline
from prepare_qualified_expert_v2_cpu_acp import scan_public

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'deployment/qualified-expert-v2-inline-init-v1'

RESOURCE_GUARD='''
def _require_allocated_init(args,root,quota,period,memory):
    from qualified_expert_v2_cpu_stage import checked_path
    assert args.resource_mode=='allocated-h1001-cpu4-ram64-init'
    assert quota!='max' and int(period)>0 and int(quota)/int(period)==4
    assert memory!='max' and int(memory)==64*1024**3
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU preflight must hide CUDA despite the allocated GPU'
    evidence=checked_path(root.parent.parent,args.allocation_evidence,reject_leaf_link=True)
    raw=evidence.read_bytes();assert hashlib.sha256(raw).hexdigest()==args.allocation_evidence_sha256
    value=json.loads(raw)
    assert value['resource_mode']==args.resource_mode and value['CPU']==4 and value['memory_bytes']==64*1024**3
    assert value['GPU_count']==1 and 'H100' in value['GPU_name'] and value['GPU_memory_bytes']>=79*1024**3
    assert value['probe_CUDA_visible'] is True

'''

def inline_helper(source):
    assert cpu.digest(source)==inline.CPU_HELPER_SHA
    text=source.decode()
    anchor="    assert quota!='max' and int(quota)/int(period)<=2 and memory!='max' and int(memory)<=8*1024**3"
    assert text.count(anchor)==1
    text=text.replace(anchor,'    _require_allocated_init(args,root,quota,period,memory)')
    anchor='    args=parser.parse_args();started='
    assert text.count(anchor)==1
    text=text.replace(anchor,"    parser.add_argument('--resource-mode',choices=['allocated-h1001-cpu4-ram64-init'],required=True)\n    parser.add_argument('--allocation-evidence',type=Path,required=True);parser.add_argument('--allocation-evidence-sha256',required=True)\n"+anchor)
    assert text.count('def main():')==1;text=text.replace('def main():',RESOURCE_GUARD+'def main():')
    assert text.count("'GPU_requested':0")==1
    text=text.replace("'GPU_requested':0","'GPU_requested':1,'resource_mode':args.resource_mode,'GPU_allocated_while_CUDA_hidden':True,'allocation_evidence_sha256':args.allocation_evidence_sha256")
    text=text.replace('CPU-only complete preflight','explicit allocated-H100 CPU preflight')
    compile(text,'explicit-inline-resource-helper','exec')
    return text.encode()

def capsule(envelope_sha,envelope_bytes,wrapper_sha):
    body=f'''own=pathlib.Path(os.environ['SUPERMIMIC_AFS_USER_ROOT'])
assert re.fullmatch(r'/mnt/afs/[0-9]{{9}}',str(own)) and own.resolve()==own and own.is_dir()
quota,period=pathlib.Path('/sys/fs/cgroup/cpu.max').read_text().split();memory=pathlib.Path('/sys/fs/cgroup/memory.max').read_text().strip()
print('INLINE_ALLOCATED_JOB_LIMITS '+json.dumps({{'cpu_max':quota+' '+period,'memory_max':memory,'GPU_requested':1,'GPU_actual_probe_pending':True}}),flush=True)
assert quota!='max' and int(quota)/int(period)==4 and memory!='max' and int(memory)==64*1024**3
source=os.environ['SUPERMIMIC_SOURCE_URL'];assert source_url_allowed(source)
blob=fetch_source(source,'{envelope_sha}',{envelope_bytes},started+585)
assert len(blob)=={envelope_bytes} and hashlib.sha256(blob).hexdigest()=='{envelope_sha}'
with tarfile.open(fileobj=io.BytesIO(blob),mode='r:gz') as bundle:
 members=bundle.getmembers();assert len(members)==4 and all(m.isfile() for m in members)
 base=bundle.extractfile('base-source-only-transport.tar.gz').read();wrapper=bundle.extractfile('inline-wrapper.py').read()
assert hashlib.sha256(base).hexdigest()=='{inline.BASE_SHA}' and hashlib.sha256(wrapper).hexdigest()=='{wrapper_sha}'
with tarfile.open(fileobj=io.BytesIO(base),mode='r:gz') as bundle:common=bundle.extractfile('bootstrap/qualified_expert_v2_cpu_stage.py').read()
assert hashlib.sha256(common).hexdigest()=='{inline.BASE_BOOT_SHA}'
module=types.ModuleType('qualified_expert_v2_cpu_stage');sys.modules[module.__name__]=module;exec(compile(common,'<verified-common>','exec'),module.__dict__)
worker=types.ModuleType('_verified_inline');exec(compile(wrapper,'<verified-inline>','exec'),worker.__dict__)
restored,helper,meta=worker.verify_envelope(blob,'{envelope_sha}',afs_user_root=str(own),local_workspace=os.environ['SUPERMIMIC_RESTORE_LOCAL_WORKSPACE'])
os.umask(0o077);paths=worker.prepare_directory(own,'incoming',restored);incoming=paths['incoming']
archive=incoming/'inline-envelope.tar.gz';module.write_owned_bytes(own,archive,blob)
program=incoming/'qualified_expert_v2_inline_init.py';module.write_owned_bytes(own,program,wrapper)
module.write_owned_bytes(own,incoming/'qualified_expert_v2_cpu_stage.py',common)
os.environ['SUPERMIMIC_JOB_STARTED_MONOTONIC']=str(started)
os.execv(sys.executable,[sys.executable,'-u',str(program),'--execute-reviewed-inline-job','--resource-mode','{inline.MODE}','--envelope',str(archive),'--expected-envelope-sha256','{envelope_sha}','--afs-user-root',str(own),'--restore-local-workspace',os.environ['SUPERMIMIC_RESTORE_LOCAL_WORKSPACE']])
'''
    header=r'''import time
started=time.monotonic()
import hashlib,io,json,os,pathlib,re,signal,sys,tarfile,types
assert sys.platform=='linux' and hasattr(signal,'setitimer')
def source_url_allowed(source):
 return '@@' not in source and (source.startswith('https://github.com/skywhite1024/SuperMimic-Reproduction/releases/download/') or re.fullmatch(r'https://raw\.githubusercontent\.com/skywhite1024/SuperMimic-Reproduction/[0-9a-f]{40}/artifacts/qualified-expert-v2-inline-init/qualified-expert-v2-inline-init-envelope\.tar\.gz',source) is not None)
def initialization_expired(signum,frame):
 signal.signal(signal.SIGALRM,signal.SIG_DFL)
 signal.setitimer(signal.ITIMER_REAL,max(.001,started+600-time.monotonic()))
 raise TimeoutError('Initialization interrupted at585s; hard600s alarm remains armed')
def initialization_terminated(signum,frame):
 raise TimeoutError('Initialization terminated; supervised source child cleanup required')
signal.signal(signal.SIGTERM,initialization_terminated)
signal.signal(signal.SIGALRM,initialization_expired)
signal.setitimer(signal.ITIMER_REAL,max(.001,started+585-time.monotonic()))
'''
    guard=(ROOT/'scripts/bounded_inline_source.py').read_text(encoding='utf8')
    failure="except BaseException as error:\n print('INLINE_CAPSULE_FAILURE '+json.dumps({'status':'initialization_failed_before_handoff','elapsed_seconds':time.monotonic()-started,'error_type':type(error).__name__,'error':str(error),'hard_initialization_deadline_seconds':600}),flush=True)\n raise\n"
    return header+guard+'\ntry:\n'+textwrap.indent(body,' ')+failure

def startup(code,afs,local):
    values={'SUPERMIMIC_AFS_USER_ROOT':afs,'SUPERMIMIC_RESTORE_LOCAL_WORKSPACE':local,'SUPERMIMIC_SOURCE_URL':'@@PUBLIC_INLINE_ENVELOPE_HTTPS_URL@@',
            'PYTHONUTF8':'1','HF_HUB_OFFLINE':'1','WANDB_MODE':'offline','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'}
    return 'set -euo pipefail\numask 077\n'+'\n'.join('export '+k+'='+shlex.quote(v) for k,v in values.items())+"\ntimeout --signal=TERM --kill-after=10s 5380s python3 -u - <<'INLINE_V2_INIT'\n"+code+'INLINE_V2_INIT\n'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--local-publication-window',action='store_true');args=parser.parse_args()
    window_path=ROOT/'evidence'/('qualified-expert-v2-inline-publication-window.json' if args.local_publication_window else 'qualified-expert-v2-inline-local-window.json')
    window=json.loads(window_path.read_text(encoding='utf-8-sig'));assert datetime.now(timezone.utc).timestamp()<window['deadline_epoch']
    if args.local_publication_window:assert window['GPU']==0 and window['training_updates']==0 and window['SenseCore_changes']==0 and window['max_CPU_threads']==2
    base=ROOT/'deployment/qualified-expert-v2-cpu-acp-preparation-v1/public/qualified-expert-v2-source-only-transport.tar.gz'
    assert cpu.file_digest(base)==inline.BASE_SHA
    plan=json.loads((ROOT/'qualified-expert-v2-plan.json').read_text());assert cpu.file_digest(ROOT/'qualified-expert-v2-plan.json')==cpu.PLAN_SHA
    for name,digest in plan['files'].items():assert cpu.file_digest(ROOT/name)==digest,name
    afs=plan['new_teacher_cloud'].split('/SuperMimic/')[0];local=str(ROOT.parent.parent)
    helper=inline_helper((ROOT/'scripts/stage_qualified_expert_v2.py').read_bytes());template=helper.replace(afs.encode(),cpu.AFS_SLOT)
    wrapper=(ROOT/'scripts/qualified_expert_v2_inline_init.py').read_bytes()
    meta={'schema':'qualified-expert-v2-inline-init-v1','resource_mode':inline.MODE,'base_sha256':inline.BASE_SHA,
          'original_CPU_helper_sha256':inline.CPU_HELPER_SHA,'scientific_plan_sha256':cpu.PLAN_SHA,
          'wrapper_sha256':cpu.digest(wrapper),'restored_inline_helper_sha256':cpu.digest(helper)}
    files={'base-source-only-transport.tar.gz':base.read_bytes(),'inline-wrapper.py':wrapper,'inline-preflight-helper.py.template':template,
           'envelope-manifest.json':(json.dumps(meta,indent=2,sort_keys=True)+'\n').encode()}
    scan_public({k:v for k,v in files.items() if not k.endswith('.tar.gz')},afs,local)
    blob=cpu.pack_bytes(files);identity=cpu.digest(blob)
    restored,exact_helper,_=inline.verify_envelope(blob,identity,afs_user_root=afs,local_workspace=local);assert exact_helper==helper
    public=OUT/'public';private=OUT/'private';public.mkdir(parents=True,exist_ok=True);private.mkdir(exist_ok=True)
    (public/'qualified-expert-v2-inline-init-envelope.tar.gz').write_bytes(blob)
    (private/'stage_qualified_expert_v2_inline.py').write_bytes(helper)
    code=capsule(identity,len(blob),cpu.digest(wrapper));compile(code,'inline-capsule','exec')
    public_command=startup(code,'@@AFS_USER_ROOT@@','@@LOCAL_WORKSPACE_RAW@@');private_command=startup(code,afs,local)
    scan_public({'inline-startup.template.sh':public_command.encode()},afs,local)
    (public/'inline-startup.template.sh').write_bytes(public_command.encode());(private/'inline-startup.command.txt').write_bytes(private_command.encode())
    record={'status':'disabled_offline_inline_initialization_plan_for_review','at':datetime.now(timezone.utc).isoformat(),
            'feasibility':'A distinct explicit resource-mode preflight can run inside the same assigned GPU job without altering the original CPU-only entry or scientific files.',
            'resource_mode':inline.MODE,'request':{'GPU':1,'type':'H10080GB','CPU':4,'RAM_GiB':64,'replicas':1,'max_restarts':0,'cluster':'computing_cluster/share-cluster'},
            'GPU_allocated_during_CPU_preflight':True,'CPU_preflight_CUDA_hidden':True,'CPU_preflight_is_zero_GPU_job':False,
            'maximum_total_program_seconds':5400,'initialization_including_source_download_max_seconds':600,'all_work_soft_deadline_seconds':5355,
            'initialization_work_deadline_seconds':585,'initialization_cleanup_reserve_seconds':15,
            'source_download_is_supervised_leaf_process_with_absolute_deadline':True,
            'socket_timeout_is_not_used_as_a_total_download_bound':True,
            'initialization_alarm_spans_decode_validation_and_exec_handoff_until_CPU_preflight_passes':True,
            'external_TERM_seconds':5380,'external_KILL_after_seconds':10,'shared_monotonic_origin_recorded_before_source_download':True,
            'scientific_command_sha256':inline.COMMAND_SHA,'scientific_plan_sha256':cpu.PLAN_SHA,'maximum_new_updates':1000,'WM_only_updates':200,
            'raw_and_calibrated_CUDA_teacher_strict_gate_before_any_update':True,'all55_and_raw15_and_both3x120s_evaluations_retained':True,
            'original_CPU2_RAM8_entry_and_default_archive_rejection_unchanged':True,'original_CPU_helper_sha256':inline.CPU_HELPER_SHA,
            'base_transport_sha256':inline.BASE_SHA,'envelope':{'bytes':len(blob),'sha256':identity,'path':str((public/'qualified-expert-v2-inline-init-envelope.tar.gz').relative_to(ROOT))},
            'inline_helper_sha256':cpu.digest(helper),'wrapper_sha256':cpu.digest(wrapper),'private_command_sha256':cpu.digest(private_command.encode()),
            'public_startup_template_sha256':cpu.digest(public_command.encode()),
            'source_download_supervisor_sha256':cpu.file_digest(ROOT/'scripts/bounded_inline_source.py'),
            'preparation_builder_sha256':cpu.file_digest(Path(__file__)),
            'source_URL_policy':'Existing own GitHub release prefix or exact own raw.githubusercontent.com repository/40hexcommit/canonical-source-asset path only',
            'mandatory_native_Linux_group_timeout_selftest_before_GPU_probe_or_preflight':True,'Linux_native_group_execution_validated_locally':False,
            'native_process_group_failure_blocks_all_scientific_execution':True,'no_automatic_second_block_or_restart':True,
            'automatic_execution_allowed':False,'public_URL_unresolved':True,'published':False,'AFS_uploads':0,'cloud_submissions':0,
            'local_preparation_GPU':0,'local_training_updates':0,'local_CPU_threads_max':2,'local_deadline':window['deadline_at'],
            'publication_authorization':'Existing human GitHub sharing authorization and independent public-payload review' if args.local_publication_window else 'Publication excluded from the original local repair window',
            'future_cloud_prerequisites':['review this concrete wrapper and tests','verify exact immutable public source bytes and distinct resolved private command','verify current Oct31 presence/network, all-account queued/runningGPU count,CCI/DNAT/quota and duplicates','one finite1H100job only; actual final terminal and resource verification required'],
            'limitations':['CPU2/8 standalone spec remains unverified; not substituted or relaxed','Windows has no WSL/Linux runtime; native Linux group tests cannot be claimed passed locally','public networking and actual initialization duration not yet measured','if initialization or cleanup guard fails, abort; preserve partial files and old models; no training retry'],
            'full_goal_complete':False}
    (ROOT/'qualified-expert-v2-inline-init-plan.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'status':record['status'],'envelope':record['envelope'],'allocated_GPU_during_CPU_preflight':True,'new_cloud_jobs':0}))

if __name__=='__main__':main()
