set -euo pipefail
umask 077
export SUPERMIMIC_AFS_USER_ROOT=@@AFS_USER_ROOT@@
export SUPERMIMIC_RESTORE_LOCAL_WORKSPACE=@@LOCAL_WORKSPACE_RAW@@
export SUPERMIMIC_SOURCE_URL=https://raw.githubusercontent.com/skywhite1024/SuperMimic-Reproduction/@@IMMUTABLE_40HEX_COMMIT@@/artifacts/qualified-expert-v2-afs-reuse-activation/qualified-expert-v2-inline-init-envelope.tar.gz
export PYTHONUTF8=1
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
timeout --signal=TERM --kill-after=10s 5380s python3 -u - <<'INLINE_V2_INIT'
import time
started=time.monotonic()
import hashlib,io,json,os,pathlib,re,signal,sys,tarfile,types
assert sys.platform=='linux' and hasattr(signal,'setitimer')
def source_url_allowed(source):
 return '@@' not in source and re.fullmatch(r'https://raw\.githubusercontent\.com/skywhite1024/SuperMimic-Reproduction/[0-9a-f]{40}/artifacts/qualified-expert-v2-afs-reuse-activation/qualified-expert-v2-inline-init-envelope\.tar\.gz',source) is not None

def verify_activation_binding(worker,meta):
 assert worker.LABEL=='qualified-expert-v2-inline-init-v3-afs-reuse' and meta['operational_namespace']=='qualified-expert-v2-inline-init-v3-afs-reuse'
 assert meta['execution_disabled'] is False and meta['execution_state']=='activation_candidate_unexecuted'

def compact_boot_failure(error,elapsed):
 import math
 name=''.join(c if 32<=ord(c)<127 else '?' for c in type(error).__name__)[:40]
 value=round(elapsed,3) if isinstance(elapsed,(int,float)) and math.isfinite(elapsed) else None
 line='INLINE_BOOT '+json.dumps({'ns':'qualified-expert-v2-inline-init-v3-afs-reuse','s':'source-or-handoff-failed','e':name,'t':value},ensure_ascii=True,separators=(',',':'))
 assert len(line.encode('ascii'))<450
 return line


def initialization_expired(signum,frame):
 signal.signal(signal.SIGALRM,signal.SIG_DFL)
 signal.setitimer(signal.ITIMER_REAL,max(.001,started+600-time.monotonic()))
 raise TimeoutError('Initialization interrupted at585s; hard600s alarm remains armed')
def initialization_terminated(signum,frame):
 raise TimeoutError('Initialization terminated; supervised source child cleanup required')
signal.signal(signal.SIGTERM,initialization_terminated)
signal.signal(signal.SIGALRM,initialization_expired)
signal.setitimer(signal.ITIMER_REAL,max(.001,started+585-time.monotonic()))
"""A leaf downloader supervised by an absolute deadline, including continuous I/O."""
import hashlib,json,os,signal,subprocess,sys,time

class SourceDownloadFailure(TimeoutError):
    def __init__(self,evidence):self.evidence=evidence;super().__init__(evidence['status'])

def fetch_source(source,expected_sha,expected_bytes,deadline,*,command=None,cleanup_reserve=3,clock=time.monotonic):
    # The production child only imports urllib and emits a capped body; it never
    # spawns descendants or writes a file. The alternate command is for offline tests.
    if command is None:
        code="import sys,urllib.request\nr=urllib.request.Request(sys.argv[1],headers={'User-Agent':'SuperMimic-bounded-source'})\nwith urllib.request.urlopen(r,timeout=15) as response:\n assert response.url.startswith('https://')\n sys.stdout.buffer.write(response.read(int(sys.argv[2])+1))\n"
        command=[sys.executable,'-u','-c',code,source,str(expected_bytes)]
    remaining=deadline-clock()-cleanup_reserve
    if remaining<=0:raise SourceDownloadFailure({'status':'source_deadline_expired_before_spawn','partial_bytes':0})
    child=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=os.name=='posix')
    data=b'';error=b'';status=None
    try:
        try:data,error=child.communicate(timeout=remaining)
        except subprocess.TimeoutExpired:
            status='source_absolute_deadline_interrupted_download'
            if os.name=='posix':
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
            else:child.kill()
            data,error=child.communicate(timeout=max(.1,deadline-clock()))
        if status is None and child.returncode!=0:status='source_downloader_failed'
        if status is None and (len(data)!=expected_bytes or hashlib.sha256(data).hexdigest()!=expected_sha):status='source_length_or_SHA_mismatch'
        if status is None and clock()>=deadline:status='source_deadline_exceeded_after_read'
        if status is not None:
            proof={'status':status,'child_pid':child.pid,'child_exitcode':child.returncode,'child_reaped':child.poll() is not None,
                   'partial_bytes':len(data),'partial_sha256':hashlib.sha256(data).hexdigest(),'stderr_bytes':len(error),
                   'stderr_sha256':hashlib.sha256(error).hexdigest()}
            print('INLINE_SOURCE_FAILURE '+json.dumps(proof),flush=True);raise SourceDownloadFailure(proof)
        return data
    finally:
        if child.poll() is None:
            if os.name=='posix':
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
            else:child.kill()
            child.wait(timeout=2)

try:
 own=pathlib.Path(os.environ['SUPERMIMIC_AFS_USER_ROOT'])
 assert re.fullmatch(r'/mnt/afs/[0-9]{9}',str(own)) and own.resolve()==own and own.is_dir()
 quota,period=pathlib.Path('/sys/fs/cgroup/cpu.max').read_text().split();memory=pathlib.Path('/sys/fs/cgroup/memory.max').read_text().strip()
 print('INLINE_ALLOCATED_JOB_LIMITS '+json.dumps({'cpu_max':quota+' '+period,'memory_max':memory,'GPU_requested':1,'GPU_actual_probe_pending':True}),flush=True)
 assert quota!='max' and int(quota)/int(period)==4 and memory!='max' and int(memory)==64*1024**3
 source=os.environ['SUPERMIMIC_SOURCE_URL'];assert source_url_allowed(source)
 blob=fetch_source(source,'2108feca16f562401625e1a1b1189d7e1db675bae62c20ab9355bf753ea229b6',78260,started+585)
 assert len(blob)==78260 and hashlib.sha256(blob).hexdigest()=='2108feca16f562401625e1a1b1189d7e1db675bae62c20ab9355bf753ea229b6'
 with tarfile.open(fileobj=io.BytesIO(blob),mode='r:gz') as bundle:
  members=bundle.getmembers();assert len(members)==4 and all(m.isfile() for m in members)
  assert {m.name for m in members}=={'base-source-only-transport.tar.gz','inline-wrapper.py','inline-preflight-helper.py.template','envelope-manifest.json'}
  base=bundle.extractfile('base-source-only-transport.tar.gz').read();wrapper=bundle.extractfile('inline-wrapper.py').read()
 assert hashlib.sha256(base).hexdigest()=='414eb6a2d1b01c6ba382d53c67ee746e88507e18802130b5a7e058e372d63ea4' and hashlib.sha256(wrapper).hexdigest()=='8b33a641d4a9800eace1075554030a1a108b94d360500a4e1b0a18a4d9b16586'
 with tarfile.open(fileobj=io.BytesIO(base),mode='r:gz') as bundle:common=bundle.extractfile('bootstrap/qualified_expert_v2_cpu_stage.py').read()
 assert hashlib.sha256(common).hexdigest()=='f8d7d0406370af540b1adf5f8d4c863b4eced70903350347daa42a9b0755b620'
 module=types.ModuleType('qualified_expert_v2_cpu_stage');sys.modules[module.__name__]=module;exec(compile(common,'<verified-common>','exec'),module.__dict__)
 worker=types.ModuleType('_verified_inline');exec(compile(wrapper,'<verified-inline>','exec'),worker.__dict__)
 restored,helper,meta=worker.verify_envelope(blob,'2108feca16f562401625e1a1b1189d7e1db675bae62c20ab9355bf753ea229b6',afs_user_root=str(own),local_workspace=os.environ['SUPERMIMIC_RESTORE_LOCAL_WORKSPACE'])
 verify_activation_binding(worker,meta)
 os.umask(0o077);paths=worker.prepare_directory(own,'incoming',restored);incoming=paths['incoming']
 archive=incoming/'inline-envelope.tar.gz';module.write_owned_bytes(own,archive,blob)
 program=incoming/'qualified_expert_v2_inline_init.py';module.write_owned_bytes(own,program,wrapper)
 module.write_owned_bytes(own,incoming/'qualified_expert_v2_cpu_stage.py',common)
 os.environ['SUPERMIMIC_JOB_STARTED_MONOTONIC']=str(started)
 os.execv(sys.executable,[sys.executable,'-u',str(program),'--execute-reviewed-inline-job','--resource-mode','allocated-h1001-cpu4-ram64-init','--envelope',str(archive),'--expected-envelope-sha256','2108feca16f562401625e1a1b1189d7e1db675bae62c20ab9355bf753ea229b6','--afs-user-root',str(own),'--restore-local-workspace',os.environ['SUPERMIMIC_RESTORE_LOCAL_WORKSPACE']])
except BaseException as error:
 print(compact_boot_failure(error,time.monotonic()-started),flush=True)
 print('INLINE_CAPSULE_FAILURE '+json.dumps({'status':'initialization_failed_before_handoff','elapsed_seconds':time.monotonic()-started,'error_type':type(error).__name__,'error':str(error),'hard_initialization_deadline_seconds':600}),flush=True)
 raise
INLINE_V2_INIT
