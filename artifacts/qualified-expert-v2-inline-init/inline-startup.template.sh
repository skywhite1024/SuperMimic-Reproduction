set -euo pipefail
umask 077
export SUPERMIMIC_AFS_USER_ROOT=@@AFS_USER_ROOT@@
export SUPERMIMIC_RESTORE_LOCAL_WORKSPACE=@@LOCAL_WORKSPACE_RAW@@
export SUPERMIMIC_SOURCE_URL=@@PUBLIC_INLINE_ENVELOPE_HTTPS_URL@@
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
 blob=fetch_source(source,'f7afc8dbe99bc5711a995b021c62dc862e1b2e17004c4d635fd81d770a3833dd',76418,started+585)
 assert len(blob)==76418 and hashlib.sha256(blob).hexdigest()=='f7afc8dbe99bc5711a995b021c62dc862e1b2e17004c4d635fd81d770a3833dd'
 with tarfile.open(fileobj=io.BytesIO(blob),mode='r:gz') as bundle:
  members=bundle.getmembers();assert len(members)==4 and all(m.isfile() for m in members)
  base=bundle.extractfile('base-source-only-transport.tar.gz').read();wrapper=bundle.extractfile('inline-wrapper.py').read()
 assert hashlib.sha256(base).hexdigest()=='414eb6a2d1b01c6ba382d53c67ee746e88507e18802130b5a7e058e372d63ea4' and hashlib.sha256(wrapper).hexdigest()=='9b39875a250d51447ebfbb381bba50b9939144a85f646ec615f29fa322d872c0'
 with tarfile.open(fileobj=io.BytesIO(base),mode='r:gz') as bundle:common=bundle.extractfile('bootstrap/qualified_expert_v2_cpu_stage.py').read()
 assert hashlib.sha256(common).hexdigest()=='f8d7d0406370af540b1adf5f8d4c863b4eced70903350347daa42a9b0755b620'
 module=types.ModuleType('qualified_expert_v2_cpu_stage');sys.modules[module.__name__]=module;exec(compile(common,'<verified-common>','exec'),module.__dict__)
 worker=types.ModuleType('_verified_inline');exec(compile(wrapper,'<verified-inline>','exec'),worker.__dict__)
 restored,helper,meta=worker.verify_envelope(blob,'f7afc8dbe99bc5711a995b021c62dc862e1b2e17004c4d635fd81d770a3833dd',afs_user_root=str(own),local_workspace=os.environ['SUPERMIMIC_RESTORE_LOCAL_WORKSPACE'])
 os.umask(0o077);paths=worker.prepare_directory(own,'incoming',restored);incoming=paths['incoming']
 archive=incoming/'inline-envelope.tar.gz';module.write_owned_bytes(own,archive,blob)
 program=incoming/'qualified_expert_v2_inline_init.py';module.write_owned_bytes(own,program,wrapper)
 module.write_owned_bytes(own,incoming/'qualified_expert_v2_cpu_stage.py',common)
 os.environ['SUPERMIMIC_JOB_STARTED_MONOTONIC']=str(started)
 os.execv(sys.executable,[sys.executable,'-u',str(program),'--execute-reviewed-inline-job','--resource-mode','allocated-h1001-cpu4-ram64-init','--envelope',str(archive),'--expected-envelope-sha256','f7afc8dbe99bc5711a995b021c62dc862e1b2e17004c4d635fd81d770a3833dd','--afs-user-root',str(own),'--restore-local-workspace',os.environ['SUPERMIMIC_RESTORE_LOCAL_WORKSPACE']])
except BaseException as error:
 print('INLINE_CAPSULE_FAILURE '+json.dumps({'status':'initialization_failed_before_handoff','elapsed_seconds':time.monotonic()-started,'error_type':type(error).__name__,'error':str(error),'hard_initialization_deadline_seconds':600}),flush=True)
 raise
INLINE_V2_INIT
