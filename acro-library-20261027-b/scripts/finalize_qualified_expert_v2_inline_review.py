"""Record only the finite offline review evidence; never publish or submit."""
import json,os,time
from datetime import datetime,timezone
from pathlib import Path
import qualified_expert_v2_cpu_stage as cpu
import qualified_expert_v2_inline_init as inline

ROOT=Path(__file__).resolve().parents[1]

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

def main():
    window=read(ROOT/'evidence/qualified-expert-v2-inline-local-window.json')
    assert time.time()<window['deadline_epoch'],'No extension of this finite local window'
    plan=read(ROOT/'qualified-expert-v2-inline-init-plan.json')
    tests=read(ROOT/'evidence/qualified-expert-v2-inline-offline-tests.json')
    baseline=read(ROOT/'evidence/qualified-expert-v2-CPU-ACP-offline-tests.json')
    assert tests['tests_run']==27 and tests['passed']==26 and tests['failures']==tests['errors']==0 and len(tests['skipped'])==1
    assert baseline['tests_run']==31 and baseline['tests_passed']==28 and baseline['failures']==baseline['errors']==0 and len(baseline['skipped_tests'])==3
    assert tests['local_CPU_affinity']==__import__('psutil').Process().cpu_affinity() and len(tests['local_CPU_affinity'])<=2
    frozen=read(ROOT/'qualified-expert-v2-plan.json');assert cpu.file_digest(ROOT/'qualified-expert-v2-plan.json')==cpu.PLAN_SHA
    assert len(frozen['files'])==25
    for name,digest in frozen['files'].items():assert cpu.file_digest(ROOT/name)==digest,name
    out=ROOT/'deployment/qualified-expert-v2-inline-init-v1'
    files=[ROOT/plan['envelope']['path'],out/'private/inline-startup.command.txt',out/'public/inline-startup.template.sh',
           out/'private/stage_qualified_expert_v2_inline.py',ROOT/'scripts/qualified_expert_v2_inline_init.py',
           ROOT/'scripts/bounded_inline_source.py',ROOT/'scripts/prepare_qualified_expert_v2_inline.py',ROOT/'scripts/test_qualified_expert_v2_inline.py',
           ROOT/'qualified-expert-v2-inline-init-plan.json',ROOT/'evidence/qualified-expert-v2-inline-offline-tests.json',
           ROOT/'evidence/qualified-expert-v2-CPU-ACP-offline-tests.json',ROOT/'deployment/qualified-expert-v2-source-and-teacher.tar.gz',
           ROOT/'deployment/qualified-expert-v2-control-supplement.tar.gz',ROOT/'scripts/stage_qualified_expert_v2.py',
           ROOT/'scripts/qualified_expert_v2_cpu_stage.py',ROOT/'deployment/qualified-expert-v2.command.txt',
           ROOT/'evidence/qualified-expert-v2-inline-independent-PID-assertion-failure.json',
           ROOT/'evidence/qualified-expert-v2-inline-independent-Windows-child-diagnostic.json',Path(__file__)]
    identities={str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'sha256':cpu.file_digest(p)} for p in files}
    assert cpu.file_digest(ROOT/plan['envelope']['path'])==plan['envelope']['sha256']
    assert cpu.file_digest(out/'private/inline-startup.command.txt')==plan['private_command_sha256']
    assert cpu.file_digest(out/'public/inline-startup.template.sh')==plan['public_startup_template_sha256']
    assert cpu.file_digest(ROOT/'scripts/qualified_expert_v2_inline_init.py')==plan['wrapper_sha256']
    assert cpu.file_digest(ROOT/'scripts/bounded_inline_source.py')==plan['source_download_supervisor_sha256']
    assert cpu.file_digest(ROOT/'scripts/prepare_qualified_expert_v2_inline.py')==plan['preparation_builder_sha256']
    assert cpu.file_digest(ROOT/'deployment/qualified-expert-v2-source-and-teacher.tar.gz')==cpu.MAIN_SHA
    assert cpu.file_digest(ROOT/'deployment/qualified-expert-v2-control-supplement.tar.gz')==cpu.CONTROL_SHA
    assert cpu.file_digest(ROOT/'scripts/stage_qualified_expert_v2.py')==inline.CPU_HELPER_SHA
    assert cpu.file_digest(ROOT/'scripts/qualified_expert_v2_cpu_stage.py')==inline.BASE_BOOT_SHA
    assert cpu.file_digest(ROOT/'deployment/qualified-expert-v2.command.txt')==inline.COMMAND_SHA
    at=datetime.now(timezone.utc).isoformat()
    window.update(status='finite_offline_preparation_completed_before_original_deadline',completed_at=at,
        current_inline_tests_passed=26,current_inline_tests_skipped=1,baseline_passed=28,baseline_skipped=3,
        review_handoff='evidence/qualified-expert-v2-inline-review-handoff.json',new_cloud_resources=0)
    review={'status':'finite_offline_inline600_fix_prepared_and_tested_native_Linux_pending_review',
        'at':at,'local_window':window,'local_CPU_affinity':tests['local_CPU_affinity'],'GPU':0,'new_training_updates':0,
        'whole_job_max_seconds':5400,'one_origin_before_source_download':True,
        'initialization':{'work_interrupt_seconds':585,'cleanup_reserve_seconds':15,'hard_cutoff_seconds':600,
            'source_child_work_cutoff_seconds':582,'source_child_reap_deadline_seconds':585,
            'covers':'source retrieval, unpacking, restoration, hash/path/model validation, repack and full CPU preflight',
            'Linux_SIGALRM_hard_cutoff_native_tested_locally':False},
        'resource_mode':inline.MODE,'future_request':plan['request'],'CPU_preflight_hides_CUDA_but_one_GPU_remains_allocated':True,
        'unchanged':{'scientific_file_count':25,'plan_sha256':cpu.PLAN_SHA,'canonical_data_and_control_members':32,
            'canonical_original_archives_verified':True,'original_CPU2_RAM8_default_rejection_retained':True,
            'strict_raw_and_calibrated_teacher_gate_failure_means_zero_updates':True,'maximum_updates':1000,'WM_only_updates':200,
            'all55_raw15_and_raw_calibrated_3x120s_denominators_and_physics_retained':True},
        'tests':{'inline':tests,'original_CPU_transport_regression':baseline},'identities':identities,
        'independent_PID_assertion_failure':{
            'preserved':'evidence/qualified-expert-v2-inline-independent-PID-assertion-failure.json',
            'diagnostic':'evidence/qualified-expert-v2-inline-independent-Windows-child-diagnostic.json',
            'old_result':{'run':25,'passed':23,'failed':1,'Linux_skipped':1},
            'observed_failure':'Popen poll had exited but PID existence asserted true; original run did not capture stable native handle evidence',
            'cause_boundary':'Cannot retrospectively prove PID reuse/transient visibility or classify the original event as a leak from that receipt alone',
            'correction':'Every actual failed/interrupted leaf now proves wait completion, exit code and same Popen Windows handle signaled; PID presence is recorded as observation only. Linux requires native process-group live members empty.',
            'new_fault_cases':'Real terminated child passes even with PID visibility forced true; real live child is rejected even with PID probe forced false',
            'runtime_download_termination_code_changed_for_this_assertion_fix':False,
            'old_failure_not_deleted_or_relabelled_as_pass':True},
        'preserved_prior_evidence':['deployment/qualified-expert-v2-inline-init-v1/private/before-600s-fix/',
            'evidence/qualified-expert-v2-CPU-ACP-offline-tests-first-failure.json',
            'evidence/qualified-expert-v2-CPU-ACP-offline-tests-before-inline600-regression.json',
            'evidence/qualified-expert-v2-inline-offline-tests-*.json'],
        'native_gaps':['Linux process groups, TERM/KILL escalation and SIGALRM delivery not natively exercised on Windows',
            'Linux full600s/whole5400s execution and actual public download/init timing unmeasured',
            'Windows3real file-symlink cases skipped for privilege1314;6real directory-junction cases passed'],
        'fail_closed_runtime_guard':'Mandatory native Linux leader+descendant timeout and KILL/empty-group selftest before GPU probe, preflight or scientific execution',
        'publication':False,'AFS_uploads':0,'CCI_starts':0,'ACP_submissions':0,'network_downloads_this_resumed_phase':0,
        'source_URL_unresolved':True,'automatic_execution_allowed':False,'new_GPU_job_submitted':False,
        'account_GPU_occupancy':'unknown; supervisor already asked; no repeat question and no zero assumption',
        'presence':'Oct30 statement is historical; current Oct31 presence/network must be checked before cloud action',
        'next':'Review this concrete disabled wrapper and its native gaps. No new cloud/publication action from this offline stage. Resolve existing account facts/current location-network/resource checks before any separately reviewed finite1H100 operation.',
        'BONES':'paused','candidate_adopted':False,'full_goal_complete':False}
    review_path=ROOT/'evidence/qualified-expert-v2-inline-review-handoff.json';write(review_path,review)
    state=read(ROOT/'execution.json')
    state.update(stage='qualified_expert_v2_inline600_offline_fix_tested_native_Linux_pending_review',
        current_stage='qualified_expert_v2_inline600_offline_fix_tested_native_Linux_pending_review',updated_at=at,full_goal_complete=False)
    state['qualified_expert_v2_inline_initialization']={k:review[k] for k in ['status','at','resource_mode','GPU','new_training_updates','whole_job_max_seconds','initialization','next','full_goal_complete']}
    state['qualified_expert_v2_inline_initialization'].update(review_handoff=str(review_path.relative_to(ROOT)),plan='qualified-expert-v2-inline-init-plan.json',
        inline_tests_passed=26,inline_tests_skipped=1,CPU_regression_passed=28,CPU_regression_skipped=3,envelope=plan['envelope'],
        retained_independent_review_failure='evidence/qualified-expert-v2-inline-independent-PID-assertion-failure.json',
        child_completion_proof='Same Popen wait/exitcode and nativeWindows handle signaled; PID existence is observation only; Linux still requires empty native group',
        original_independent_failure_cause_not_retrospectively_proven=True,
        public_URL_unresolved=True,automatic_execution_allowed=False,new_GPU_job_submitted=False)
    write(ROOT/'execution.json',state)
    handoff=read(ROOT/'evidence/retention-1030-active-handoff.json');handoff.update(updated_at=at,stage=state['stage'],
        active_task='Finite inline600 offline fix and regression complete; disabled concrete wrapper ready for review. NativeLinux and cloud prerequisites unresolved.',
        qualified_expert_v2_inline_initialization=state['qualified_expert_v2_inline_initialization'],full_goal_complete=False)
    write(ROOT/'evidence/retention-1030-active-handoff.json',handoff)
    anchor_path=ROOT.parent/'monitoring/supermimic-active-experiment.json';anchor=read(anchor_path)
    anchor.update(updated_at=at,active_stage=state['stage'],full_goal_complete=False)
    anchor['executor_current_work_anchor']={**state['qualified_expert_v2_inline_initialization'],
        'at':at,'completed_existing_local_phase_before_deadline':True,'local_deadline':'2026-10-31T00:50:51+08:00',
        'full55_tracking_contact_and_continuous_objective_remains_incomplete':True,'BONES':'paused','new_cloud_resources':0}
    write(anchor_path,anchor)
    window.update(status='finite_offline_preparation_completed_before_original_deadline',completed_at=at,
        current_inline_tests_passed=26,current_inline_tests_skipped=1,baseline_passed=28,baseline_skipped=3,
        review_handoff=str(review_path.relative_to(ROOT)),new_cloud_resources=0)
    write(ROOT/'evidence/qualified-expert-v2-inline-local-window.json',window)
    print(json.dumps({'status':review['status'],'inline_passed':26,'inline_skipped':1,'baseline_passed':28,'baseline_skipped':3,
        'envelope':plan['envelope'],'completed_at':at,'deadline_not_extended':True,'new_GPU_jobs':0,'full_goal_complete':False}))

if __name__=='__main__':main()
