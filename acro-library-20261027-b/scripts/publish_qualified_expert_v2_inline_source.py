"""Publish only the reviewed four-file source payload; never touch SenseCore."""
import argparse,ast,json,os,re,shutil,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
import qualified_expert_v2_cpu_stage as cpu
import qualified_expert_v2_inline_init as inline
from bounded_inline_source import fetch_source
from prepare_qualified_expert_v2_cpu_acp import scan_public

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parent/'publication/SuperMimic-Reproduction'
REL=Path('artifacts/qualified-expert-v2-inline-init')
OUT=ROOT/'deployment/qualified-expert-v2-inline-init-v1'
RECEIPT=ROOT/'evidence/qualified-expert-v2-inline-source-publication.json'
EXPECTED='f7afc8dbe99bc5711a995b021c62dc862e1b2e17004c4d635fd81d770a3833dd'
BASE_COMMIT='8a160749b66fea84a5636d29b44b2478d805535c'
TAGS={'refs/tags/v0.1.0':'1cc5827a6b1e1927cff5071eccd7c3e1d63b1ef9','refs/tags/v0.2.0':'43c12c035f5ddbad3d0322c55604c7e98697f614'}
NAMES=['qualified-expert-v2-inline-init-envelope.tar.gz','inline-startup.template.sh','README.md','preparation-manifest.json']

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
def window():
    value=read(ROOT/'evidence/qualified-expert-v2-inline-publication-window.json')
    assert time.time()<value['deadline_epoch'],'Finite GitHub publication window expired'
    assert value['GPU']==value['training_updates']==value['SenseCore_changes']==0
    assert len(__import__('psutil').Process().cpu_affinity())<=2
    return value

def git(*args):
    seconds=min(60,window()['deadline_epoch']-time.time());assert seconds>0
    env=os.environ.copy()|{'GIT_TERMINAL_PROMPT':'0'}
    result=subprocess.run(['git','-c','http.lowSpeedLimit=1024','-c','http.lowSpeedTime=30',*args],cwd=REPO,env=env,
        capture_output=True,text=True,encoding='utf8',timeout=seconds)
    if result.returncode:
        error=ROOT/'evidence/qualified-expert-v2-inline-publication-git-failure.txt'
        error.write_text(result.stdout+'\n'+result.stderr,encoding='utf8')
        raise RuntimeError(f'git {args[0]} failed with exit{result.returncode}; private evidence saved')
    return result.stdout.strip()

def identities(files):return {name:{'bytes':len(data),'sha256':cpu.digest(data)} for name,data in files.items()}

def audited_payload():
    files={name:(REPO/REL/name).read_bytes() for name in NAMES}
    assert len(files[NAMES[0]])==76418 and cpu.digest(files[NAMES[0]])==EXPECTED
    assert files[NAMES[1]]==(OUT/'public'/NAMES[1]).read_bytes()
    plan=read(ROOT/'qualified-expert-v2-plan.json')
    assert cpu.file_digest(ROOT/'qualified-expert-v2-plan.json')==cpu.PLAN_SHA
    for name,digest in plan['files'].items():assert cpu.file_digest(ROOT/name)==digest,name
    assert cpu.file_digest(ROOT/'scripts/stage_qualified_expert_v2.py')==inline.CPU_HELPER_SHA
    assert cpu.file_digest(ROOT/'scripts/qualified_expert_v2_cpu_stage.py')==inline.BASE_BOOT_SHA
    assert cpu.file_digest(ROOT/'deployment/qualified-expert-v2-source-and-teacher.tar.gz')==cpu.MAIN_SHA
    assert cpu.file_digest(ROOT/'deployment/qualified-expert-v2-control-supplement.tar.gz')==cpu.CONTROL_SHA
    afs=plan['new_teacher_cloud'].split('/SuperMimic/')[0];local=str(ROOT.parent.parent)
    texts={name:data for name,data in files.items() if not name.endswith('.tar.gz')}
    envelope=cpu.transport_files(files[NAMES[0]])
    for name,data in envelope.items():
        if name.endswith('.tar.gz'):
            for nested,value in cpu.transport_files(data).items():texts[f'base/{nested}']=value
        else:texts[f'envelope/{name}']=data
    scan_public(texts,afs,local)
    assert set(p.name for p in (REPO/REL).iterdir())==set(NAMES)
    return files,len(texts)

def save(value):value.update(updated_at=datetime.now(timezone.utc).isoformat(),full_goal_complete=False);write(RECEIPT,value)

def prepare():
    w=window();assert git('status','--porcelain')=='' and git('branch','--show-current')=='main'
    assert git('rev-parse','HEAD')==BASE_COMMIT
    assert git('remote','get-url','origin')=='https://github.com/skywhite1024/SuperMimic-Reproduction.git'
    target=REPO/REL;assert not target.exists();target.mkdir()
    plan=read(ROOT/'qualified-expert-v2-inline-init-plan.json');tests=read(ROOT/'evidence/qualified-expert-v2-inline-offline-tests.json')
    assert tests['tests_run']==28 and tests['passed']==27 and tests['failures']==tests['errors']==0 and len(tests['skipped'])==1
    for name in NAMES[:2]:(target/name).write_bytes((OUT/'public'/name).read_bytes())
    manifest={'schema':'qualified-expert-v2-public-source-preparation-v1','status':'source_preparation_only_V2_untrained_and_unadopted',
        'archive':{'file':NAMES[0],'bytes':76418,'sha256':EXPECTED},'public_template':{'file':NAMES[1],'sha256':plan['public_startup_template_sha256']},
        'scientific_plan_sha256':cpu.PLAN_SHA,'scientific_files':25,'canonical_data_and_control_members':32,
        'canonical_original_archives_sha256':{'main':cpu.MAIN_SHA,'control':cpu.CONTROL_SHA},
        'resource_mode':inline.MODE,'future_job':{'GPU':1,'GPU_type':'H10080GB','CPU':4,'RAM_GiB':64,'replicas':1,'restarts':0},
        'GPU_remains_allocated_during_CUDA_hidden_CPU_preflight':True,'initialization_max_seconds':600,'whole_job_max_seconds':5400,
        'initialization_work_seconds':585,'initialization_cleanup_reserve_seconds':15,'one_origin_before_source_download':True,
        'maximum_training_updates':1000,'WM_only_updates':200,'strict_raw_and_calibrated_teacher_gate_before_any_update':True,
        'teacher_gate_failure_updates':0,'maximum_is_not_completed_updates':True,'V2_training_executed':False,'candidate_adopted':False,
        'full_goal':{'single_deployed_policy':True,'new_clips':15,'old_clips':40,'full_clips':55,'tracking_and_contact_quality_required':True,
            'continuous_trials_per_mode':3,'seconds_per_continuous_trial':120,'post_t0_resets_allowed':0,'complete':False},
        'offline_tests':{'run':28,'passed':27,'skipped_native_Linux':1,'GPU':0,'training_updates':0},
        'Linux_native_process_groups_and_SIGALRM_locally_validated':False,'mandatory_native_group_timeout_KILL_selftest_before_scientific_execution':True,
        'standalone_CPU2_RAM8_guard_unchanged':True,'source_URL_policy':plan['source_URL_policy'],'automatic_cloud_execution_allowed':False,
        'private_command_models_raw_motion_and_runtime_logs_in_this_publication':False,'BONES_route':'paused','full_goal_complete':False}
    write(target/NAMES[3],manifest)
    (target/'README.md').write_text('''# Qualified-expert V2 source preparation

Status: V2 has not been trained or adopted. The full experiment goal remains incomplete. This directory adds a reviewed source archive, a disabled public startup template and verification metadata. Existing published v0.1.0 and v0.2.0 results are unchanged.

The proposed finite job uses one H100, 4 CPU cores and 64 GiB RAM. One GPU remains allocated while the CPU preflight hides CUDA. Initialization includes downloading, unpacking and validation within the same 600-second deadline (585 seconds of work plus cleanup); the whole job is limited to 5,400 seconds. The strict raw and calibrated teacher checks must pass before any of at most 1,000 updates, including 200 world-model-only updates. A teacher-gate failure requires zero updates. No automatic retry or extra training block is permitted.

The unchanged goal is one deployed policy for all 55 clips (15 new plus 40 old), tracking and contact quality, and three 120-second continuous trials per evaluation mode with zero resets after the initial state. Preparation and process completion do not establish this acceptance.

The archive is 76,418 bytes. Its SHA256 is `f7afc8dbe99bc5711a995b021c62dc862e1b2e17004c4d635fd81d770a3833dd`. Pin the raw download URL to the containing 40-character Git commit. The public startup template retains placeholders and does not submit a job. Private commands, models, raw motion data and runtime logs are excluded.

Offline tests: 27 passed and one native Linux group test skipped on Windows. Windows leaf-process completion and mocked Linux signal logic do not prove native Linux behavior. The job must pass its native process-group timeout/KILL self-test before the GPU probe, CPU preflight or scientific execution. The original standalone CPU2/RAM8 guard and all scientific files, frozen plan, canonical members, physics, thresholds and evaluation denominators are preserved. Current account, network, resource and duplicate checks remain required before cloud execution. BONES remains paused.
''',encoding='utf8',newline='\n')
    files,text_count=audited_payload()
    save({'status':'source_only_payload_prepared_audited_not_committed','window':w,'base_commit':BASE_COMMIT,
        'public_directory':REL.as_posix(),'public_files':identities(files),'expanded_text_files_privacy_scanned':text_count,
        'GPU':0,'training_updates':0,'SenseCore_changes':0,'published':False,'pushed':False,'new_release_created':False,'old_tags':TAGS})
    print(json.dumps({'status':'prepared_and_scanned','files':identities(files),'expanded_texts':text_count,'published':False}))

def normalize_public_texts():
    value=read(RECEIPT);assert value['status']=='source_only_payload_prepared_audited_not_committed'
    backup=OUT/'private/before-public-text-LF-normalization';backup.mkdir(exist_ok=False)
    failure=ROOT/'evidence/qualified-expert-v2-inline-publication-git-failure.txt'
    (backup/'first-git-diff-check-failure.txt').write_bytes(failure.read_bytes())
    for name in ['README.md','preparation-manifest.json']:
        path=REPO/REL/name;old=path.read_bytes();assert cpu.digest(old)==value['public_files'][name]['sha256']
        (backup/name).write_bytes(old);path.write_bytes(old.replace(b'\r\n',b'\n'))
    files,count=audited_payload();value.update(public_files=identities(files),expanded_text_files_privacy_scanned=count,
        first_git_diff_check_failure={'preserved':str((backup/'first-git-diff-check-failure.txt').relative_to(ROOT)),
            'reason':'New README/manifest were written with Windows CRLF while the repository preserves bytes; Git reported CR as trailing whitespace',
            'fix':'Write only the two new public text files with LF; source archive and public startup template byte identities unchanged'})
    save(value);print(json.dumps({'status':'new_public_text_LF_fixed_and_rescanned','public_files':identities(files)}))

def commit_push():
    value=read(RECEIPT);assert value['status']=='source_only_payload_prepared_audited_not_committed'
    files,text_count=audited_payload();assert identities(files)==value['public_files']
    refs=dict((line.split()[1],line.split()[0]) for line in git('ls-remote','origin','refs/heads/main',*TAGS).splitlines())
    assert refs=={'refs/heads/main':BASE_COMMIT,**TAGS},'Remote changed: do not overwrite or force push'
    for name in NAMES:git('add',*(['-f'] if name.endswith('.tar.gz') else []),'--',(REL/name).as_posix())
    staged=git('diff','--cached','--name-only').splitlines()
    assert sorted(staged)==sorted((REL/name).as_posix() for name in NAMES)
    git('diff','--cached','--check')
    git('commit','-m','Publish reviewed source-only V2 initialization preparation')
    commit=git('rev-parse','HEAD');assert re.fullmatch('[a-f0-9]{40}',commit)
    value.update(status='source_only_committed_push_pending',commit=commit,staged_paths=staged);save(value)
    git('push','origin','HEAD:refs/heads/main')
    value.update(status='source_only_push_returned_remote_and_HTTP_verification_pending',pushed=True);save(value)
    print(json.dumps({'status':value['status'],'commit':commit,'public_files':len(files)}))

def verify(native_windows=False):
    w=window();value=read(RECEIPT);assert value['pushed'];commit=value['commit']
    assert git('rev-parse','HEAD')==commit and git('status','--porcelain')==''
    refs=dict((line.split()[1],line.split()[0]) for line in git('ls-remote','origin','refs/heads/main',*TAGS).splitlines())
    assert refs=={'refs/heads/main':commit,**TAGS}
    files,text_count=audited_payload();assert identities(files)==value['public_files']
    base=f'https://raw.githubusercontent.com/skywhite1024/SuperMimic-Reproduction/{commit}/{REL.as_posix()}/'
    readback=OUT/'private/HTTP-readback';readback.mkdir(exist_ok=True)
    proofs={}
    if native_windows:
        assert os.name=='nt' and shutil.which('pwsh')
        if value.get('HTTP_error_type'):
            prior={'error_type':value['HTTP_error_type'],'status':value['status'],'observed_at':value['updated_at'],
                'initial_download_stdout_evidence':{'status':'source_downloader_failed','child_pid':57544,'child_exitcode':1,'child_reaped':True,
                    'partial_bytes':0,'stderr_bytes':3673,'stderr_sha256':'988796c832d75ec0b1601a3abb20f19d1de5c132c9dee97d74b86dd3be1f347c'},
                'Python_HEAD_diagnostic':read(ROOT/'evidence/qualified-expert-v2-inline-public-HTTP-Python-diagnostic.json')}
            value.setdefault('preserved_HTTP_failures',[]).append(prior);save(value)
    try:
        for name,data in files.items():
            remaining=w['deadline_epoch']-time.time();assert remaining>0
            command=None
            if native_windows:command=[shutil.which('pwsh'),'-NoLogo','-NoProfile','-NonInteractive','-File',
                str(ROOT/'scripts/read_public_inline_asset_windows.ps1'),'-Url',base+name,'-MaximumBytes',str(len(data))]
            received=fetch_source(base+name,cpu.digest(data),len(data),time.monotonic()+min(30,remaining),command=command)
            assert received==data;(readback/name).write_bytes(received)
            proofs[name]={'url':base+name,'bytes':len(received),'sha256':cpu.digest(received),'exact_local_bytes':True,
                'HTTP_transport':'Windows_NET_HTTPClient_default_TLS' if native_windows else 'Python_urllib_default_TLS',
                'TLS_verification_disabled':False,'Linux_native_transport_validated':False}
            value['HTTP_readback']=proofs;save(value)
    except BaseException as error:
        value.update(status='pushed_but_HTTP_readback_incomplete',HTTP_error_type=type(error).__name__,published=False)
        if hasattr(error,'evidence'):value['latest_HTTP_failure_evidence']=error.evidence
        save(value);raise
    source_url=base+NAMES[0]
    draft=OUT/'private/inline-startup.command.txt';original=draft.read_bytes();assert original.count(b'@@PUBLIC_INLINE_ENVELOPE_HTTPS_URL@@')==1
    resolved=original.replace(b'@@PUBLIC_INLINE_ENVELOPE_HTTPS_URL@@',source_url.encode())
    code=resolved.decode().split("<<'INLINE_V2_INIT'\n",1)[1].rsplit('INLINE_V2_INIT',1)[0];compile(code,'resolved-capsule','exec')
    function=next(x for x in ast.parse(code).body if isinstance(x,ast.FunctionDef) and x.name=='source_url_allowed')
    scope={'re':re};exec(compile(ast.Module(body=[function],type_ignores=[]),'source-policy','exec'),scope);assert scope['source_url_allowed'](source_url)
    resolved_path=OUT/'private/inline-startup.resolved.command.txt';resolved_path.write_bytes(resolved);assert draft.read_bytes()==original
    plan_path=ROOT/'qualified-expert-v2-inline-init-plan.json';plan=read(plan_path)
    plan.update(published=True,public_URL_unresolved=False,source_publication={'commit':commit,'immutable_URL':source_url,
        'public_all4_HTTP_bytes_verified':True,'resolved_private_command':str(resolved_path.relative_to(ROOT)),
        'resolved_private_command_sha256':cpu.digest(resolved),'disabled_draft_preserved':True,'cloud_executed':False})
    write(plan_path,plan)
    value.update(status='source_only_publication_and_immutable_HTTP_readback_complete',published=True,remote_refs_verified=refs,
        immutable_source_URL=source_url,resolved_private_command={'path':str(resolved_path.relative_to(ROOT)),'sha256':cpu.digest(resolved),'executed':False},
        plan_sha256=cpu.file_digest(plan_path),disabled_draft_preserved=True,source_envelope_unchanged=True,
        tests=read(ROOT/'evidence/qualified-expert-v2-inline-offline-tests.json'),
        baseline=read(ROOT/'evidence/qualified-expert-v2-CPU-ACP-offline-tests.json'),Linux_native_validated=False,
        local_CPU_affinity=__import__('psutil').Process().cpu_affinity(),public_privacy_scan_texts=text_count,
        new_training_updates=0,new_GPU_jobs=0,CCI_starts=0,AFS_uploads=0,SenseCore_changes=0,
        source_wrapper_sha256=cpu.file_digest(ROOT/'scripts/qualified_expert_v2_inline_init.py'),
        preparation_builder_sha256=cpu.file_digest(ROOT/'scripts/prepare_qualified_expert_v2_inline.py'),
        current_tests_sha256=cpu.file_digest(ROOT/'scripts/test_qualified_expert_v2_inline.py'),
        frozen_scientific_plan_sha256=cpu.PLAN_SHA,scientific_files_unchanged=25,canonical_members_unchanged=32,
        existing_account_GPU_question_pending_no_repeat=True,BONES='paused',candidate_adopted=False)
    save(value)
    at=value['updated_at'];state=read(ROOT/'execution.json')
    state.update(stage='qualified_expert_v2_sanitized_source_published_no_cloud_execution',current_stage='qualified_expert_v2_sanitized_source_published_no_cloud_execution',updated_at=at,full_goal_complete=False)
    state['inline_source_publication']={key:value[key] for key in ['status','commit','immutable_source_URL','published','resolved_private_command','new_GPU_jobs','SenseCore_changes','full_goal_complete']}
    state['inline_source_publication']['receipt']=str(RECEIPT.relative_to(ROOT))
    state['next_actions']=['GitHub source-only publication and exact HTTP readback complete. Do not republish or rerun closed offline windows.',
        'Wait for already-asked whole-account queued/running GPU facts and current Oct31 presence; then fresh academy network/account/CCI/DNAT/quota/duplicate checks before any separately reviewed finite GPU operation.',
        'Full55 tracking/contact and continuous acceptance remain incomplete; preserve all failures, native Linux selftest and strict teacher gates. BONES paused.']
    write(ROOT/'execution.json',state)
    anchor_path=ROOT.parent/'monitoring/supermimic-active-experiment.json';anchor=read(anchor_path)
    anchor.update(updated_at=at,active_stage=state['stage'],current_execution_turn=w['current_execution_turn'],active_execution_turn=w['current_execution_turn'],full_goal_complete=False)
    anchor['executor_current_work_anchor']={**state['inline_source_publication'],'at':at,'current_execution_turn':w['current_execution_turn'],
        'next_actions':state['next_actions'],'local_deadline':w['deadline_at'],'BONES':'paused'}
    write(anchor_path,anchor)
    handoff_path=ROOT/'evidence/retention-1030-active-handoff.json';handoff=read(handoff_path)
    handoff.update(updated_at=at,stage=state['stage'],active_task=state['next_actions'][1],inline_source_publication=state['inline_source_publication'],full_goal_complete=False);write(handoff_path,handoff)
    w.update(status='source_only_GitHub_publication_complete_before_deadline',completed_at=at,commit=commit,receipt=str(RECEIPT.relative_to(ROOT)));write(ROOT/'evidence/qualified-expert-v2-inline-publication-window.json',w)
    print(json.dumps({'status':value['status'],'commit':commit,'source_url':source_url,'HTTP_files_verified':len(proofs),
        'new_GPU_jobs':0,'SenseCore_changes':0,'full_goal_complete':False}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','normalize-public-texts','commit-push','verify','verify-native-windows']);args=parser.parse_args()
    window();{'prepare':prepare,'normalize-public-texts':normalize_public_texts,'commit-push':commit_push,'verify':verify,
        'verify-native-windows':lambda:verify(native_windows=True)}[args.phase]()
