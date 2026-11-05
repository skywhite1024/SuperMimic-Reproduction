"""Restore an audited public transport and run the existing CPU-only staging check.

This operational entry point never trains, starts a service, or submits a job.
"""
import argparse,gzip,hashlib,io,json,os,re,signal,subprocess,sys,tarfile,time,urllib.request
from datetime import datetime,timezone
from pathlib import Path

MAIN_SHA='3edc27b42a47a16adfb5ab1c8f1e95ec8ec754a345ac07e2ec17b2fe0f2d031f'
CONTROL_SHA='0b809a4011fbc6b66fe197e776d59104be545cfc08d45362e8145d516cc14712'
MAIN_MANIFEST_SHA='1dfb777ef0e80c53e0b6db05db97a3d9197828a5f86893be4e222751b7733d5d'
CONTROL_MANIFEST_SHA='566caae65cfc0436a526a34849e48d1e43af6b49d346c8e2a7e8e9ea948f9666'
PLAN_SHA='bc5e8b397accd4f585732803385516b755538f924afebc9e73fbc205dc763476'
TEACHER_SHA='0f4a7bc67ae9e3440711e401e37f8276102e59946177117d082a6d7aabe108b4'
TEACHER_BYTES=36481141
TEACHER_PATH='policy/qualified-expert-8808.pt'
TEACHER_URL='https://github.com/skywhite1024/SuperMimic-Reproduction/releases/download/v0.1.0/acro88-08-policy.pt'
SCHEMA='qualified-expert-v2-sanitized-transport-v1'
AFS_SLOT=b'@@AFS_USER_ROOT@@'
LOCAL_JSON_SLOT=b'@@LOCAL_WORKSPACE_JSON@@'
LABEL='qualified-expert-v2-CPU-ACP-stage-v1'

def digest(value):return hashlib.sha256(value).hexdigest()

def file_digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def is_link(path):
    path=Path(path)
    return path.is_symlink() or (hasattr(path,'is_junction') and path.is_junction())

def owner_directory(own):
    own=Path(own)
    assert own.is_absolute() and own.resolve(strict=True)==own and own.is_dir(),'Personal AFS anchor must be a real existing directory'
    assert not is_link(own),'Personal AFS anchor cannot itself be a link'
    return own

def checked_path(own,path,*,reject_leaf_link=False):
    """Check lexical and resolved containment, including missing/dangling leaves."""
    own=owner_directory(own);path=Path(path)
    assert path.is_absolute() and path.is_relative_to(own),'Path is outside the personal AFS namespace'
    assert '..' not in path.parts and path.resolve(strict=False).is_relative_to(own),'Resolved path escapes personal AFS'
    if reject_leaf_link:assert not is_link(path),'Leaf symlinks/junctions are not allowed'
    return path

def ensure_owned_directory(own,path):
    path=checked_path(own,path)
    if os.path.lexists(path):
        assert path.is_dir(),'Existing parent is not a directory'
        return checked_path(own,path)
    ensure_owned_directory(own,path.parent)
    checked_path(own,path).mkdir(mode=0o700)
    return checked_path(own,path)

def validate_stage_layout(own,phase):
    own=owner_directory(own);assert phase in ('incoming','work')
    root=checked_path(own,own/'SuperMimic/acro-library-20261027-b');assert root.is_dir()
    layout={'own':own,'root':root,'deployment':root/'deployment','evidence':root/'evidence',
            'teacher':root/TEACHER_PATH,'receipt':root/'evidence'/f'{LABEL}.json',
            'proof':root/'evidence/qualified-expert-v2-cloud-CPU-preflight.json',
            'incoming':root/'deployment/qualified-expert-v2-CPU-ACP-transport-v1','work':root/'deployment'/LABEL}
    # Validate every existing parent/model/receipt before making any parent directory.
    for key in ('deployment','evidence','incoming','work'):
        value=checked_path(own,layout[key],reject_leaf_link=key in ('incoming','work'))
        if os.path.lexists(value):assert value.is_dir(),key
    teacher=checked_path(own,layout['teacher'],reject_leaf_link=True)
    if os.path.lexists(teacher):assert teacher.is_file(),'Preserve invalid existing teacher'
    for key in ('receipt','proof'):
        value=checked_path(own,layout[key],reject_leaf_link=True)
        assert not os.path.lexists(value),'Preserve any previous receipt/proof; do not retry'
    assert not os.path.lexists(layout[phase]),'Stage directory already exists; do not reuse or retry'
    if phase=='incoming':assert not os.path.lexists(layout['work']),'Preserve any previous work directory'
    return layout

def validate_restore_targets(own,restored):
    root=checked_path(own,Path(own)/'SuperMimic/acro-library-20261027-b')
    for name in [*restored['main'],*restored['control'],TEACHER_PATH]:
        assert safe_name(name)
        checked_path(own,root/name,reject_leaf_link=True)
    plan=json.loads(restored['main']['qualified-expert-v2-plan.json'])
    for key in ('initial_full_cloud','old_teacher_cloud','new_teacher_cloud'):
        model=checked_path(own,Path(plan[key]))
        if os.path.lexists(model):assert model.is_file(),'Invalid existing model path'
        elif key!='new_teacher_cloud':raise AssertionError('Required existing model is missing')

def prepare_stage_directory(own,phase,*,restored=None):
    layout=validate_stage_layout(own,phase)
    if restored is not None:validate_restore_targets(own,restored)
    for key in ('deployment','evidence'):ensure_owned_directory(own,layout[key])
    stage=checked_path(own,layout[phase],reject_leaf_link=True)
    assert not os.path.lexists(stage),'Do not overwrite/reuse a prior stage directory'
    stage.mkdir(mode=0o700)  # Exclusive leaf; parents were checked and created separately.
    checked_path(own,stage,reject_leaf_link=True)
    return layout

def owned_file(own,path,*,exclusive):
    path=checked_path(own,path,reject_leaf_link=True)
    assert checked_path(own,path.parent).is_dir()
    if exclusive:
        assert not os.path.lexists(path),'Preserve any existing output'
        flags=os.O_WRONLY|os.O_CREAT|os.O_EXCL
    else:
        assert path.is_file(),'Receipt must remain an existing regular file'
        flags=os.O_WRONLY|os.O_TRUNC
    flags|=getattr(os,'O_NOFOLLOW',0)|getattr(os,'O_BINARY',0)
    descriptor=os.open(path,flags,0o600)
    try:checked_path(own,path,reject_leaf_link=True)
    except BaseException:os.close(descriptor);raise
    return os.fdopen(descriptor,'wb')

def write_owned_bytes(own,path,value,*,exclusive=True):
    with owned_file(own,path,exclusive=exclusive) as stream:stream.write(value)

def safe_name(name):
    return (bool(name) and not name.startswith('/') and '\\' not in name and ':' not in name
            and all(part not in ('','..','.') for part in name.split('/')))

def transport_files(blob):
    assert len(blob)<=512*1024,'Transport exceeds source-only size cap'
    with tarfile.open(fileobj=io.BytesIO(blob),mode='r:gz') as archive:
        members=archive.getmembers();assert 1<=len(members)<=40
        assert all(m.isfile() and safe_name(m.name) and 0<=m.size<=256*1024 for m in members)
        assert sum(m.size for m in members)<=1024*1024
        names=[m.name for m in members];assert len(names)==len(set(names))
        return {m.name:archive.extractfile(m).read() for m in members}

def canonical_rows(raw,expected,count):
    assert digest(raw)==expected,'Canonical manifest anchor mismatch'
    rows=json.loads(raw)['files'];assert len(rows)==count
    assert all(safe_name(r['path']) and isinstance(r['bytes'],int) and r['bytes']>=0
               and re.fullmatch('[a-f0-9]{64}',r['sha256']) for r in rows)
    assert len({r['path'] for r in rows})==count
    return {r['path']:r for r in rows}

def restore_transport(blob,expected_sha,*,afs_user_root,local_workspace):
    """Verify everything in memory before writing any restored private source."""
    assert digest(blob)==expected_sha,'Actual transport SHA mismatch'
    assert re.fullmatch(r'/mnt/afs/[0-9]{9}',afs_user_root)
    assert '\x00' not in local_workspace and '@@' not in local_workspace
    files=transport_files(blob);meta=json.loads(files['transport-manifest.json'])
    assert meta['schema']==SCHEMA
    assert meta['canonical_archives']=={'main':MAIN_SHA,'control':CONTROL_SHA}
    main_raw=files['canonical/bundle-manifest.json'];control_raw=files['canonical/control-manifest.json']
    main=canonical_rows(main_raw,MAIN_MANIFEST_SHA,30);control=canonical_rows(control_raw,CONTROL_MANIFEST_SHA,2)
    assert main[TEACHER_PATH]['bytes']==TEACHER_BYTES and main[TEACHER_PATH]['sha256']==TEACHER_SHA
    assert meta['teacher']=={'url':TEACHER_URL,'bytes':TEACHER_BYTES,'sha256':TEACHER_SHA,'path':TEACHER_PATH}
    expected_sources={('main',p) for p in main if p!=TEACHER_PATH}|{('control',p) for p in control}
    rows=meta['source_members'];assert len(rows)==31
    assert {(r['group'],r['path']) for r in rows}==expected_sources
    required={'transport-manifest.json','canonical/bundle-manifest.json','canonical/control-manifest.json',
              'operations/stage_qualified_expert_v2.py.template','bootstrap/qualified_expert_v2_cpu_stage.py'}
    restored={'main':{},'control':{}}
    def restore(value):
        return value.replace(AFS_SLOT,afs_user_root.encode()).replace(LOCAL_JSON_SLOT,json.dumps(local_workspace,ensure_ascii=False)[1:-1].encode())
    for row in rows:
        name=f'templates/{row["group"]}/{row["path"]}';assert row['template']==name
        required.add(name);template=files[name]
        assert len(template)==row['template_bytes'] and digest(template)==row['template_sha256']
        value=restore(template);original=(main if row['group']=='main' else control)[row['path']]
        assert len(value)==original['bytes'] and digest(value)==original['sha256'],row['path']
        assert AFS_SLOT not in value and LOCAL_JSON_SLOT not in value
        restored[row['group']][row['path']]=value
    assert set(files)==required,'Unexpected/missing transport member'
    assert digest(restored['main']['qualified-expert-v2-plan.json'])==PLAN_SHA
    plan=json.loads(restored['main']['qualified-expert-v2-plan.json'])
    assert len(plan['files'])==25
    for name,sha in plan['files'].items():assert digest(restored['main'][name])==sha,name
    helper_template=files['operations/stage_qualified_expert_v2.py.template']
    assert digest(helper_template)==meta['helper']['template_sha256']
    helper=restore(helper_template);assert digest(helper)==meta['helper']['restored_sha256']
    assert digest(files['bootstrap/qualified_expert_v2_cpu_stage.py'])==meta['bootstrap_sha256']
    restored.update(main_manifest=main_raw,control_manifest=control_raw,helper=helper,
                    identity={'actual_transport_sha256':digest(blob),'canonical_main_archive_sha256':MAIN_SHA,
                              'canonical_control_archive_sha256':CONTROL_SHA,'canonical_main_manifest_sha256':MAIN_MANIFEST_SHA,
                              'canonical_control_manifest_sha256':CONTROL_MANIFEST_SHA,'restored_helper_sha256':digest(helper),
                              'all31_source_or_control_members_restored_byte_identically':True,'all25_scientific_files_unchanged':True})
    return restored

def pack_bytes(files):
    """Deterministic packaging; member bytes stay canonical, compression may differ."""
    output=io.BytesIO()
    with gzip.GzipFile(fileobj=output,mode='wb',filename='',mtime=0,compresslevel=1) as compressed:
        with tarfile.open(fileobj=compressed,mode='w',format=tarfile.USTAR_FORMAT) as archive:
            for name,value in files.items():
                assert safe_name(name)
                info=tarfile.TarInfo(name);info.size=len(value);info.mode=0o600;info.mtime=0
                archive.addfile(info,io.BytesIO(value))
    return output.getvalue()

def repack(restored,teacher_path,output,*,own=None):
    teacher_path=Path(teacher_path)
    if own is not None:checked_path(own,teacher_path,reject_leaf_link=True)
    assert teacher_path.is_file() and teacher_path.stat().st_size==TEACHER_BYTES and file_digest(teacher_path)==TEACHER_SHA
    main=dict(restored['main']);main[TEACHER_PATH]=teacher_path.read_bytes();main['bundle-manifest.json']=restored['main_manifest']
    control=dict(restored['control']);control['control-manifest.json']=restored['control_manifest']
    output=Path(output)
    if own is None:output.mkdir(parents=True,exist_ok=True)
    else:ensure_owned_directory(own,output)
    result={}
    for name,files in [('main',main),('control',control)]:
        blob=pack_bytes(files);path=output/f'manifest-identical-{name}.tar.gz'
        if own is None:
            with path.open('xb') as stream:stream.write(blob)
        else:write_owned_bytes(own,path,blob)
        result[name]={'path':str(path),'bytes':len(blob),'sha256':digest(blob)}
    return result

def download_teacher(target,deadline,*,own):
    checked_path(own,target,reject_leaf_link=True)
    assert not os.path.lexists(target)
    assert time.monotonic()<deadline-30
    request=urllib.request.Request(TEACHER_URL,headers={'User-Agent':'SuperMimic-fixed-hash-stage'})
    with urllib.request.urlopen(request,timeout=min(15,deadline-time.monotonic()-20)) as response:
        assert response.url.startswith('https://')
        declared=response.headers.get('Content-Length')
        if declared is not None:assert int(declared)==TEACHER_BYTES
        size=0;sha=hashlib.sha256()
        with owned_file(own,target,exclusive=True) as stream:
            while True:
                assert time.monotonic()<deadline-20
                chunk=response.read(65536)
                if not chunk:break
                size+=len(chunk);assert size<=TEACHER_BYTES
                sha.update(chunk);stream.write(chunk)
        assert size==TEACHER_BYTES and sha.hexdigest()==TEACHER_SHA

def resource_limits(cpu_text,memory_text):
    quota,period=cpu_text.split();memory=memory_text.strip()
    assert quota!='max' and int(period)>0 and 0<int(quota)/int(period)<=2
    assert memory!='max' and 0<int(memory)<=8*1024**3
    return int(quota)/int(period),int(memory)

def run_cloud(args):
    started=time.monotonic();deadline=started+540
    def interrupted(signum,frame):raise TimeoutError('Finite CPU stage interrupted')
    signal.signal(signal.SIGTERM,interrupted)
    assert re.fullmatch(r'/mnt/afs/[0-9]{9}',args.afs_user_root)
    own=owner_directory(Path(args.afs_user_root))
    initial_layout=validate_stage_layout(own,'work');root=initial_layout['root']
    checked_path(own,args.transport,reject_leaf_link=True)
    # Check the actual hard limits before any write or model transfer.
    CPU,memory=resource_limits(Path('/sys/fs/cgroup/cpu.max').read_text(),Path('/sys/fs/cgroup/memory.max').read_text())
    os.umask(0o077);os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',WANDB_MODE='offline',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    restored=restore_transport(args.transport.read_bytes(),args.expected_transport_sha256,
                               afs_user_root=args.afs_user_root,local_workspace=args.restore_local_workspace)
    layout=prepare_stage_directory(own,'work',restored=restored)
    receipt_path=layout['receipt'];work=layout['work']
    result={'status':'running_CPU_only_stage','started_at':datetime.now(timezone.utc).isoformat(),
            'actual_CPU_cores':CPU,'actual_memory_bytes':memory,
            'GPU_requested':0,'GPU_allocation_must_be_verified_from_platform_configuration':True,
            'CUDA_VISIBLE_DEVICES':'','new_training_updates':0,'SSH_or_DNAT_used':False,'retries':0,**restored['identity']}
    first_save=True
    def save():
        nonlocal first_save
        write_owned_bytes(own,receipt_path,(json.dumps(result,indent=2)+'\n').encode(),exclusive=first_save)
        first_save=False;print('CPU_ACP_STAGE_RECEIPT '+json.dumps(result),flush=True)
    save()
    try:
        teacher=checked_path(own,layout['teacher'],reject_leaf_link=True)
        if os.path.lexists(teacher):assert teacher.is_file() and file_digest(teacher)==TEACHER_SHA
        else:
            teacher=work/'public-expert-policy.pt';download_teacher(teacher,deadline,own=own)
        archives=repack(restored,teacher,work,own=own);result['actual_repacked_archives']=archives
        helper=work/'stage_qualified_expert_v2.py'
        write_owned_bytes(own,helper,restored['helper'])
        assert file_digest(helper)==restored['identity']['restored_helper_sha256']
        python=own/'SuperMimic/lafan1-20261024/uni/.venv/bin/python';assert python.is_file()
        remaining=deadline-time.monotonic()-15;assert remaining>0
        command=[str(python),'-u',str(helper),'--archive',archives['main']['path'],
                 '--control-archive',archives['control']['path'],'--allow-manifest-identical-repack','--root',str(root)]
        validate_restore_targets(own,restored)
        with owned_file(own,work/'CPU-preflight.log',exclusive=True) as log:
            child=subprocess.Popen(command,cwd=root,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                code=child.wait(timeout=remaining)
                assert code==0,f'Existing CPU preflight failed with exitcode{code}'
            except BaseException:
                if child.poll() is None:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=5)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=3)
                raise
        proof=checked_path(own,layout['proof'],reject_leaf_link=True)
        verified=json.loads(proof.read_text());assert verified['new_training_updates']==0 and verified['CUDA_visible_devices']==0
        result.update(status='CPU_stage_and_existing_full_preflight_passed',CPU_preflight_sha256=file_digest(proof),new_GPU_job_submitted=False)
    except BaseException as error:
        result.update(status='CPU_stage_failed_partial_evidence_preserved',error=repr(error));raise
    finally:
        result.update(ended_at=datetime.now(timezone.utc).isoformat(),wall_seconds=time.monotonic()-started,
                      resource_release_requires_actual_ACP_terminal_observation=True);save()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute-CPU-stage',action='store_true')
    parser.add_argument('--transport',type=Path,required=True);parser.add_argument('--expected-transport-sha256',required=True)
    parser.add_argument('--afs-user-root',required=True);parser.add_argument('--restore-local-workspace',required=True)
    args=parser.parse_args();assert args.execute_CPU_stage,'Explicit CPU staging flag required'
    run_cloud(args)

if __name__=='__main__':main()
