"""Build a small sanitized transport and disabled, reviewable finite CPU-ACP plan."""
import hashlib,json,re,shlex,tarfile
from datetime import datetime,timezone
from pathlib import Path
import qualified_expert_v2_cpu_stage as transport
import stage_qualified_expert_v2 as stage

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'deployment/qualified-expert-v2-cpu-acp-preparation-v1'

def scan_public(files,private_root,local_root):
    forbidden=[private_root.encode(),private_root.rsplit('/',1)[-1].encode(),local_root.encode(),
               json.dumps(local_root)[1:-1].encode(),b'203.0.113.1',b'0197ee17-b6eb-7846-b2b4-a77c5f509b92']
    patterns=[rb'-----BEGIN [A-Z ]*PRIVATE KEY-----',rb'(?:hf_|ghp_|github_pat_|sk-)[A-Za-z0-9_\-]{20,}',
              rb'(?i)(?<![A-Za-z0-9])[A-Z]:[\\/]+Users[\\/]',rb'(?i)(?<![A-Za-z0-9])[A-Z]:[\\/]']
    for name,value in files.items():
        assert not name.endswith(('.pt','.csv','.npz','.log','.pem','.key')),name
        assert not any(word in value for word in forbidden),f'Private identifier in {name}'
        assert not any(re.search(pattern,value) for pattern in patterns),f'Private-data pattern in {name}'
    return {'files_scanned':len(files),'personal_AFS_and_account_ID_found':False,'private_keys_tokens_home_paths_and_known_private_endpoints_found':False,
            'model_binary_raw_motion_and_temporary_log_members':0}

def capsule(archive_sha,archive_bytes,bootstrap_sha):
    return f'''import hashlib,io,json,os,pathlib,re,sys,tarfile,urllib.request
root=pathlib.Path(os.environ['SUPERMIMIC_AFS_USER_ROOT'])
assert re.fullmatch(r'/mnt/afs/[0-9]{{9}}',str(root)) and root.resolve()==root and root.is_dir()
quota,period=pathlib.Path('/sys/fs/cgroup/cpu.max').read_text().split()
memory=pathlib.Path('/sys/fs/cgroup/memory.max').read_text().strip()
print('CPU_STAGE_ACTUAL_LIMITS '+json.dumps({{'cpu_max':quota+' '+period,'memory_max':memory,'GPU_requested':0}}),flush=True)
assert quota!='max' and 0<int(quota)/int(period)<=2 and memory!='max' and 0<int(memory)<=8*1024**3
source=os.environ['SUPERMIMIC_SOURCE_URL']
assert '@@' not in source and (source.startswith('https://github.com/skywhite1024/SuperMimic-Reproduction/releases/download/') or source.startswith('https://skywhite1024.github.io/SuperMimic-Reproduction/'))
os.umask(0o077)
incoming=root/'SuperMimic/acro-library-20261027-b/deployment/qualified-expert-v2-CPU-ACP-transport-v1'
request=urllib.request.Request(source,headers={{'User-Agent':'SuperMimic-fixed-source-stage'}})
with urllib.request.urlopen(request,timeout=15) as response:
 assert response.url.startswith('https://')
 blob=response.read({archive_bytes}+1)
assert len(blob)=={archive_bytes} and hashlib.sha256(blob).hexdigest()=='{archive_sha}'
with tarfile.open(fileobj=io.BytesIO(blob),mode='r:gz') as bundle:
 members=bundle.getmembers();assert len(members)==36 and all(m.isfile() for m in members)
 boot=bundle.extractfile('bootstrap/qualified_expert_v2_cpu_stage.py').read()
assert hashlib.sha256(boot).hexdigest()=='{bootstrap_sha}'
auditor={{'__name__':'_verified_CPU_stage_auditor'}}
exec(compile(boot,'<verified-CPU-stage>','exec'),auditor)
restored=auditor['restore_transport'](blob,'{archive_sha}',afs_user_root=str(root),local_workspace=os.environ['SUPERMIMIC_RESTORE_LOCAL_WORKSPACE'])
layout=auditor['prepare_stage_directory'](root,'incoming',restored=restored)
incoming=layout['incoming']
archive=incoming/'source-only-transport.tar.gz';auditor['write_owned_bytes'](root,archive,blob)
program=incoming/'CPU-stage.py';auditor['write_owned_bytes'](root,program,boot)
os.execv(sys.executable,[sys.executable,'-u',str(program),'--execute-CPU-stage','--transport',str(archive),'--expected-transport-sha256','{archive_sha}','--afs-user-root',str(root),'--restore-local-workspace',os.environ['SUPERMIMIC_RESTORE_LOCAL_WORKSPACE']])
'''

def startup(python,afs,local):
    env={'SUPERMIMIC_AFS_USER_ROOT':afs,'SUPERMIMIC_RESTORE_LOCAL_WORKSPACE':local,
         'SUPERMIMIC_SOURCE_URL':'@@STAGING_ARCHIVE_HTTPS_URL@@','CUDA_VISIBLE_DEVICES':'',
         'PYTHONUTF8':'1','HF_HUB_OFFLINE':'1','WANDB_MODE':'offline','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'}
    return ('set -euo pipefail\numask 077\n'+'\n'.join('export '+k+'='+shlex.quote(v) for k,v in env.items())
            +"\ntimeout --signal=TERM --kill-after=10s 580s python3 -u - <<'SUPERMIMIC_CPU_STAGE'\n"+python+'SUPERMIMIC_CPU_STAGE\n')

def main():
    main_archive=ROOT/'deployment/qualified-expert-v2-source-and-teacher.tar.gz'
    control_archive=ROOT/'deployment/qualified-expert-v2-control-supplement.tar.gz'
    main_manifest,main_data=stage.audited_members(main_archive)
    control_manifest,control_data=stage.audited_control(control_archive)
    with tarfile.open(main_archive,'r:gz') as t:main_raw=t.extractfile('bundle-manifest.json').read()
    with tarfile.open(control_archive,'r:gz') as t:control_raw=t.extractfile('control-manifest.json').read()
    plan=json.loads(main_data['qualified-expert-v2-plan.json'])
    afs=plan['new_teacher_cloud'].split('/SuperMimic/')[0];local=str(ROOT.parent.parent)
    local_json=json.dumps(local,ensure_ascii=False)[1:-1].encode()
    def sanitize(raw):
        assert transport.AFS_SLOT not in raw and transport.LOCAL_JSON_SLOT not in raw
        return raw.replace(afs.encode(),transport.AFS_SLOT).replace(local_json,transport.LOCAL_JSON_SLOT)
    files={'canonical/bundle-manifest.json':main_raw,'canonical/control-manifest.json':control_raw};rows=[]
    for group,data in [('main',main_data),('control',control_data)]:
        for name,raw in data.items():
            if name==transport.TEACHER_PATH:continue
            template=sanitize(raw);path=f'templates/{group}/{name}';files[path]=template
            rows.append({'group':group,'path':name,'template':path,'template_bytes':len(template),'template_sha256':transport.digest(template)})
    helper=(ROOT/'scripts/stage_qualified_expert_v2.py').read_bytes();helper_template=sanitize(helper)
    files['operations/stage_qualified_expert_v2.py.template']=helper_template
    bootstrap=(ROOT/'scripts/qualified_expert_v2_cpu_stage.py').read_bytes()
    files['bootstrap/qualified_expert_v2_cpu_stage.py']=bootstrap
    meta={'schema':transport.SCHEMA,'canonical_archives':{'main':transport.MAIN_SHA,'control':transport.CONTROL_SHA},
          'source_members':rows,'helper':{'template_sha256':transport.digest(helper_template),'restored_sha256':transport.digest(helper)},
          'bootstrap_sha256':transport.digest(bootstrap),
          'teacher':{'url':transport.TEACHER_URL,'bytes':transport.TEACHER_BYTES,'sha256':transport.TEACHER_SHA,'path':transport.TEACHER_PATH}}
    files['transport-manifest.json']=(json.dumps(meta,sort_keys=True,indent=2)+'\n').encode()
    assert len(files)==36
    privacy=scan_public(files,afs,local)
    blob=transport.pack_bytes(files);archive_sha=transport.digest(blob)
    public=OUT/'public';private=OUT/'private';public.mkdir(parents=True,exist_ok=True);private.mkdir(exist_ok=True)
    target=public/'qualified-expert-v2-source-only-transport.tar.gz';target.write_bytes(blob)
    (public/'transport-manifest.json').write_bytes(files['transport-manifest.json'])
    code=capsule(archive_sha,len(blob),transport.digest(bootstrap));compile(code,'CPU-stage-capsule','exec')
    public_command=startup(code,'@@AFS_USER_ROOT@@','@@LOCAL_WORKSPACE_RAW@@')
    scan_public({'CPU-ACP-startup.template.sh':public_command.encode()},afs,local)
    (public/'CPU-ACP-startup.template.sh').write_bytes(public_command.encode())
    private_command=startup(code,afs,local);(private/'CPU-ACP-startup.command.txt').write_bytes(private_command.encode())
    availability_path=ROOT/'evidence/qualified-expert-v2-CPU-ACP-availability.json'
    availability=json.loads(availability_path.read_text()) if availability_path.exists() else None
    report={'status':'offline_preparation_for_review_not_published_not_submitted','at':datetime.now(timezone.utc).isoformat(),
            'plan_kind':'one finite noninteractive CPU-only staging/preflight; no training or CUDA teacher trials',
            'request':{'cluster':'computing_cluster/share-cluster','resource':'standard CPU node','GPU':0,'CPU':2,'RAM_GiB':8,'replicas':1,'max_restarts':0},
            'maximum_program_seconds':600,'outer_timeout_seconds':580,'kill_after_seconds':10,'inner_stage_budget_seconds':540,
            'automatic_execution_allowed':False,'new_jobs_submitted':0,'new_training_updates':0,'source_transport_published':False,
            'source_transport_URL':None,'startup_has_unresolved_source_URL_placeholder':True,
            'CPU2_RAM8_platform_support_verified':False,'platform_availability_evidence':str(availability_path.relative_to(ROOT)),
            'platform_availability_observed_at':availability['at'] if availability else None,
            'platform_limitation':'Target computing_cluster standard CPU-node table returned no data; do not request a larger specification.',
            'transport':{'path':str(target.relative_to(ROOT)),'bytes':len(blob),'sha256':archive_sha,'members':len(files),**privacy},
            'canonical_main_archive_sha256':transport.MAIN_SHA,'canonical_control_archive_sha256':transport.CONTROL_SHA,
            'canonical_main_manifest_sha256':transport.MAIN_MANIFEST_SHA,'canonical_control_manifest_sha256':transport.CONTROL_MANIFEST_SHA,
            'scientific_plan_sha256_unchanged':transport.PLAN_SHA,'frozen_scientific_files':25,
            'operational_helper_SHA256':transport.digest(helper),'original_helper_75c_preserved':True,
            'public_teacher_URL':transport.TEACHER_URL,'public_teacher_SHA256':transport.TEACHER_SHA,'public_teacher_bytes':transport.TEACHER_BYTES,
            'public_startup_template_sha256':transport.digest(public_command.encode()),'private_command_sha256':transport.digest(private_command.encode()),
            'secrets_or_model_binary_published_or_uploaded_this_phase':False,
            'preserved_guards':['original archives are immutable','explicit repack switch required','both original manifest hashes pinned',
                               'every canonical member length and SHA exact','no altered25scientificfiles,plan,models,reference,55denominators,quality gates or CUDAteacher gate',
                               'actualcgroup<=2CPU/8GiB before download','all existing dependency/asset/parent trace/full model/6CPU proof checks retained',
                               'CPUonly:CUDA hidden,noGPUdriver,no training','one download attempt per public artifact; preserve conflicts/partial evidence',
                               'finite own process group termination; real platform terminal/resource verification still required'],
            'filesystem_guard_revision':2,
            'filesystem_guards':['capsule and runtime share prepare_stage_directory and owned_file',
                                 'all layout and restored/model target paths resolve inside the real personal AFS anchor before any writes or model reads',
                                 'missing legitimate deployment/evidence parents created one checked level at a time',
                                 'incoming/work leaves use exclusive mkdir; prior receipts/proofs and stale stage directories are rejected',
                                 'receipt leaves reject existing/dangling symlinks and junctions; repeated receipt writes recheck containment',
                                 'O_NOFOLLOW used where available; no claim of race-free Linux parent-directory traversal',
                                 'real host filesystem link coverage and unsupported cases recorded in offline test receipt'],
            'review_prerequisites':['confirm requested2CPU8GiB standard CPU-node spec is offered','approve any public transport publication separately; no push now',
                                    'fill fixed public URL and record final commandSHA after review','fresh academy network and actual account/resource checks',
                                    'inspect own previous job IDs before any single submission'],'full_goal_complete':False}
    (ROOT/'qualified-expert-v2-cpu-acp-stage-plan.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    (ROOT/'evidence/qualified-expert-v2-CPU-ACP-offline-preparation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'status':report['status'],'transport':report['transport'],'platform_support_verified':False,'original_scientific_plan_changed':False},ensure_ascii=False))

if __name__=='__main__':main()
