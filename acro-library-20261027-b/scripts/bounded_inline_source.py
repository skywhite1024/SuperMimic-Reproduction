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
