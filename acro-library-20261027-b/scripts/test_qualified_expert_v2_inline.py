"""Finite offline budget/hash/fault checks; never invokes CUDA or training."""
import ast,contextlib,importlib.util,io,json,os,subprocess,sys,tempfile,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import qualified_expert_v2_cpu_stage as cpu
import qualified_expert_v2_inline_init as inline
import prepare_qualified_expert_v2_inline as builder
import bounded_inline_source as source_guard

ROOT=Path(__file__).resolve().parents[1]
DOWNLOAD_PROOFS=[]

class InlinePreparation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan=json.loads((ROOT/'qualified-expert-v2-plan.json').read_text())
        cls.afs=cls.plan['new_teacher_cloud'].split('/SuperMimic/')[0];cls.local=str(ROOT.parent.parent)
        cls.blob=(builder.OUT/'public/qualified-expert-v2-inline-init-envelope.tar.gz').read_bytes();cls.sha=cpu.digest(cls.blob)
        cls.restored,cls.helper,cls.meta=inline.verify_envelope(cls.blob,cls.sha,afs_user_root=cls.afs,local_workspace=cls.local)
        cls.helper_path=builder.OUT/'private/stage_qualified_expert_v2_inline.py'
        spec=importlib.util.spec_from_file_location('_explicit_inline_preflight',cls.helper_path)
        cls.module=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.module)

    def setUp(self):
        parent=ROOT/'tmp';parent.mkdir(exist_ok=True)
        self.temporary=tempfile.TemporaryDirectory(prefix='inline-offline-',dir=parent)
        self.base=Path(self.temporary.name).resolve();assert self.base.is_relative_to(parent.resolve())
        self.own=self.base/'own';self.own.mkdir();self.root=self.own/'SuperMimic/acro-library-20261027-b';self.root.mkdir(parents=True)
        self.work=self.own/'logs';self.work.mkdir()

    def tearDown(self):
        assert self.base.is_relative_to((ROOT/'tmp').resolve());self.temporary.cleanup()

    def test_single_budget_includes_initialization_and_does_not_reset(self):
        now=[100.];budget=inline.Budget(100.,lambda:now[0])
        now[0]=650.;self.assertEqual(budget.remaining(True),35);self.assertEqual(budget.remaining(False),4805)
        self.assertEqual(budget.init_hard_deadline,700)
        now[0]=700.
        with self.assertRaises(TimeoutError):budget.remaining(True)
        now[0]=5455.
        with self.assertRaises(TimeoutError):budget.remaining(False)

    def test_future_or_expired_start_origin_rejected(self):
        for start,now in [(101,100),(100,700),(100,800)]:
            with self.subTest(start=start,now=now):
                with self.assertRaises(AssertionError):inline.Budget(start,lambda:now)

    def download_failure(self,code,expected,seconds=.7):
        started=time.monotonic();output=io.StringIO();spawned=[];real_popen=source_guard.subprocess.Popen
        def tracked_popen(*args,**kwargs):
            child=real_popen(*args,**kwargs);spawned.append(child);return child
        with contextlib.redirect_stdout(output),mock.patch.object(source_guard.subprocess,'Popen',side_effect=tracked_popen):
            with self.assertRaises(source_guard.SourceDownloadFailure) as caught:
                source_guard.fetch_source('offline-fixture',cpu.digest(expected),len(expected),started+seconds,
                    command=[sys.executable,'-u','-c',code],cleanup_reserve=.15)
        proof=caught.exception.evidence;proof=dict(proof,elapsed_seconds=time.monotonic()-started,
            test=self.id(),native_host=sys.platform,network_used=False)
        self.assertEqual(len(spawned),1);proof['same_child_completion']=self.same_child_completion(spawned[0])
        DOWNLOAD_PROOFS.append(proof)
        self.assertTrue(proof['child_reaped'])
        self.assertLess(proof['elapsed_seconds'],2.5)
        self.assertIn(json.dumps(caught.exception.evidence),output.getvalue())
        return proof

    def same_child_completion(self,child):
        # PIDs may be reused or have transient visibility. The retained Popen
        # handle and wait result identify the exact process created by this test.
        self.assertIsNotNone(child.poll());code=child.wait(timeout=0)
        proof={'pid':child.pid,'wait_completed':True,'exitcode':code,
            'pid_exists_observation_only':__import__('psutil').pid_exists(child.pid)}
        self.assertEqual(code,child.returncode)
        if os.name=='nt':
            import _winapi
            status=_winapi.WaitForSingleObject(child._handle,0)
            self.assertEqual(status,_winapi.WAIT_OBJECT_0)
            proof.update(native_Windows_same_Popen_handle_signaled=True,native_wait_status=status,
                native_handle_exitcode=_winapi.GetExitCodeProcess(child._handle))
            self.assertEqual(proof['native_handle_exitcode'],code)
        else:
            members=inline.live_group(child.pid);self.assertEqual(members,[])
            proof['native_Linux_group_live_members_after']=members
        return proof

    def test_native_blocked_source_child_is_interrupted_reaped_and_failure_retained(self):
        proof=self.download_failure('import threading; threading.Event().wait(5)',b'x'*40)
        self.assertEqual(proof['status'],'source_absolute_deadline_interrupted_download')
        self.assertEqual(proof['partial_bytes'],0)

    def test_native_continuous_slow_source_cannot_extend_absolute_deadline(self):
        code="import sys,time\nfor i in range(40):\n sys.stdout.buffer.write(b'x');sys.stdout.buffer.flush();time.sleep(.05)\n"
        proof=self.download_failure(code,b'x'*40)
        self.assertEqual(proof['status'],'source_absolute_deadline_interrupted_download')
        self.assertGreater(proof['partial_bytes'],0);self.assertLess(proof['partial_bytes'],40)

    def test_native_source_same_length_wrong_SHA_rejected_after_child_reaped(self):
        proof=self.download_failure("import sys;sys.stdout.buffer.write(b'wrong')",b'right')
        self.assertEqual(proof['status'],'source_length_or_SHA_mismatch');self.assertEqual(proof['child_exitcode'],0)
        self.assertEqual(proof['partial_bytes'],5)

    def test_native_source_success_and_expired_origin_does_not_spawn(self):
        data=b'exact source'
        self.assertEqual(source_guard.fetch_source('offline-fixture',cpu.digest(data),len(data),time.monotonic()+2,
            command=[sys.executable,'-u','-c',"import sys;sys.stdout.buffer.write(b'exact source')"],cleanup_reserve=.1),data)
        with mock.patch.object(source_guard.subprocess,'Popen') as spawn:
            with self.assertRaises(source_guard.SourceDownloadFailure):
                source_guard.fetch_source('offline-fixture',cpu.digest(data),len(data),time.monotonic()-1)
            spawn.assert_not_called()

    def test_interrupted_source_supervisor_finally_reaps_real_leaf(self):
        # Inject the same Python exception raised by the Linux TERM/alarm handler.
        # Leaf kill/wait is native Windows; signal delivery is not exercised here.
        spawned=[];real_popen=source_guard.subprocess.Popen
        def interrupted_popen(*args,**kwargs):
            child=real_popen(*args,**kwargs);spawned.append(child)
            child.communicate=mock.Mock(side_effect=TimeoutError('offline injected initialization interrupt'))
            return child
        with mock.patch.object(source_guard.subprocess,'Popen',side_effect=interrupted_popen):
            with self.assertRaisesRegex(TimeoutError,'injected initialization interrupt'):
                source_guard.fetch_source('offline-fixture','0'*64,40,time.monotonic()+2,
                    command=[sys.executable,'-u','-c','import threading;threading.Event().wait(5)'],cleanup_reserve=.1)
        self.assertEqual(len(spawned),1);completion=self.same_child_completion(spawned[0])
        DOWNLOAD_PROOFS.append({'test':self.id(),'exception_injected':True,'native_signal_delivery_tested':False,
            'child_pid':spawned[0].pid,'child_exitcode':spawned[0].returncode,'child_reaped':True,'network_used':False,
            'same_child_completion':completion})

    def test_same_child_completion_uses_handle_when_PID_visibility_is_true(self):
        child=subprocess.Popen([sys.executable,'-u','-c','pass'],start_new_session=os.name=='posix')
        child.wait(timeout=2)
        with mock.patch('psutil.pid_exists',return_value=True):completion=self.same_child_completion(child)
        self.assertTrue(completion['pid_exists_observation_only'])
        DOWNLOAD_PROOFS.append({'test':self.id(),'PID_visibility_mocked':True,'real_leaf_process':True,
            'same_child_completion':completion})

    def test_same_child_completion_rejects_live_child_despite_PID_probe_false(self):
        child=subprocess.Popen([sys.executable,'-u','-c','import threading;threading.Event().wait(5)'],start_new_session=os.name=='posix')
        try:
            with mock.patch('psutil.pid_exists',return_value=False):
                with self.assertRaises(AssertionError):self.same_child_completion(child)
        finally:child.kill();child.wait(timeout=2)
        completion=self.same_child_completion(child)
        DOWNLOAD_PROOFS.append({'test':self.id(),'PID_visibility_mocked_during_live_check':True,
            'live_child_rejected':True,'same_child_completion_after_cleanup':completion})

    def timer_fixture(self):
        now=[100.];budget=inline.Budget(100.,lambda:now[0]);handlers={14:'previous'};timers=[]
        fake=SimpleNamespace(SIGALRM=14,SIG_DFL='default',ITIMER_REAL=0,
            getsignal=lambda sig:handlers[sig],signal=lambda sig,handler:handlers.__setitem__(sig,handler),
            setitimer=lambda kind,seconds:timers.append(seconds))
        return now,budget,handlers,timers,fake

    def test_mock_Linux_init_alarm_uses_original_585_and_600_deadlines(self):
        now,budget,handlers,timers,fake=self.timer_fixture()
        with mock.patch.object(inline,'sys',SimpleNamespace(platform='linux')),mock.patch.object(inline,'signal',fake):
            cancel=inline.arm_initialization_timer(budget);self.assertEqual(timers,[585])
            now[0]=685.
            with self.assertRaises(TimeoutError):handlers[14](14,None)
            self.assertEqual(timers[-1],15);self.assertEqual(handlers[14],'default')
            cancel();self.assertEqual(timers[-1],0);self.assertEqual(handlers[14],'previous')

    def test_mock_Linux_init_alarm_defers_exception_during_group_cleanup(self):
        now,budget,handlers,timers,fake=self.timer_fixture();aborted=[]
        with mock.patch.object(inline,'sys',SimpleNamespace(platform='linux')),mock.patch.object(inline,'signal',fake),mock.patch.object(inline,'_CLEANUP_ABORT_HOOK',lambda:aborted.append(True)):
            inline.arm_initialization_timer(budget);now[0]=687.;handlers[14](14,None)
        self.assertEqual(aborted,[True]);self.assertEqual(timers[-1],13)

    def test_explicit_new_resource_mode_and_unchanged_old_CPU_guard(self):
        self.assertEqual(inline.allocated_limits('400000 100000',str(64*1024**3),inline.MODE)['GPU_requested'],1)
        for spec in [('200000 100000',str(8*1024**3),inline.MODE),('400000 100000',str(64*1024**3),'cpu2-ram8-no-gpu')]:
            with self.assertRaises(AssertionError):inline.allocated_limits(*spec)
        with self.assertRaises(AssertionError):cpu.resource_limits('400000 100000',str(64*1024**3))
        self.assertEqual(cpu.file_digest(ROOT/'scripts/stage_qualified_expert_v2.py'),inline.CPU_HELPER_SHA)

    def test_generated_helper_preserves_all_archive_guard_functions(self):
        old=ast.parse((ROOT/'scripts/stage_qualified_expert_v2.py').read_text());new=ast.parse(self.helper.decode())
        pick=lambda tree,name:next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name==name)
        for name in ['sha','safe_member','audited_members','audited_control']:
            self.assertEqual(ast.dump(pick(old,name)),ast.dump(pick(new,name)))
        self.assertEqual(self.helper,builder.inline_helper((ROOT/'scripts/stage_qualified_expert_v2.py').read_bytes()))

    def evidence(self,**changes):
        value={'resource_mode':inline.MODE,'CPU':4,'memory_bytes':64*1024**3,'GPU_count':1,'GPU_name':'H100 80GB HBM3',
               'GPU_memory_bytes':80*1024**3,'probe_CUDA_visible':True};value.update(changes)
        path=self.root/'allocation.json';path.write_text(json.dumps(value))
        args=SimpleNamespace(resource_mode=inline.MODE,allocation_evidence=path,allocation_evidence_sha256=cpu.file_digest(path))
        return args

    def test_hidden_CUDA_preflight_requires_allocated_H100_attestation(self):
        with mock.patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}):
            self.module._require_allocated_init(self.evidence(),self.root,'400000','100000',str(64*1024**3))
            for bad in [{'GPU_count':0},{'GPU_count':2},{'GPU_name':'A100'},{'GPU_memory_bytes':40*1024**3},{'probe_CUDA_visible':False}]:
                with self.subTest(bad=bad):
                    with self.assertRaises(AssertionError):self.module._require_allocated_init(self.evidence(**bad),self.root,'400000','100000',str(64*1024**3))
        with mock.patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0'}):
            with self.assertRaises(AssertionError):self.module._require_allocated_init(self.evidence(),self.root,'400000','100000',str(64*1024**3))

    def test_attestation_hash_mismatch_rejected(self):
        args=self.evidence();args.allocation_evidence_sha256='0'*64
        with mock.patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}):
            with self.assertRaises(AssertionError):self.module._require_allocated_init(args,self.root,'400000','100000',str(64*1024**3))

    def test_actual_H100_probe_schema_rejects_wrong_hardware(self):
        good={'GPU_count':1,'GPU_name':'H100','GPU_memory_bytes':80*1024**3};self.assertEqual(inline.verify_probe(good),good)
        for bad in [good|{'GPU_count':0},good|{'GPU_count':2},good|{'GPU_name':'A100'},good|{'GPU_memory_bytes':40*1024**3}]:
            with self.assertRaises(AssertionError):inline.verify_probe(bad)

    def test_envelope_and_inner_source_hash_failures_rejected(self):
        with self.assertRaises(AssertionError):inline.verify_envelope(self.blob,'0'*64,afs_user_root=self.afs,local_workspace=self.local)
        files=cpu.transport_files(self.blob)
        for name in ['base-source-only-transport.tar.gz','inline-preflight-helper.py.template']:
            with self.subTest(name=name):
                wrong=dict(files);wrong[name]+=b'changed';blob=cpu.pack_bytes(wrong)
                with self.assertRaises(AssertionError):inline.verify_envelope(blob,cpu.digest(blob),afs_user_root=self.afs,local_workspace=self.local)

    def test_envelope_resource_mode_cannot_claim_zero_GPU(self):
        files=cpu.transport_files(self.blob);meta=json.loads(files['envelope-manifest.json']);meta['resource_mode']='cpu2-ram8-no-gpu'
        files['envelope-manifest.json']=json.dumps(meta).encode();blob=cpu.pack_bytes(files)
        with self.assertRaises(AssertionError):inline.verify_envelope(blob,cpu.digest(blob),afs_user_root=self.afs,local_workspace=self.local)

    def test_frozen_science_original_CPU_packet_and_commands_unchanged(self):
        self.assertEqual(cpu.file_digest(ROOT/'qualified-expert-v2-plan.json'),cpu.PLAN_SHA)
        for name,digest in self.plan['files'].items():self.assertEqual(cpu.file_digest(ROOT/name),digest,name)
        self.assertEqual(cpu.file_digest(ROOT/'deployment/qualified-expert-v2.command.txt'),inline.COMMAND_SHA)
        base=ROOT/'deployment/qualified-expert-v2-cpu-acp-preparation-v1/public/qualified-expert-v2-source-only-transport.tar.gz'
        self.assertEqual(cpu.file_digest(base),inline.BASE_SHA)
        self.assertEqual(self.plan['new_denominator'],15);self.assertEqual(self.plan['old_denominator'],40)

    def test_unfinished_receipt_does_not_misreport_zero_updates(self):
        result=inline.interpret_science({'status':'failed_or_interrupted_all_partial_evidence_retained','new_training_updates':0})
        self.assertIsNone(result['confirmed_new_updates'])
        self.assertEqual(inline.interpret_science({'status':'teacher_pretraining_control_failed_no_training','new_training_updates':0})['confirmed_new_updates'],0)
        with self.assertRaises(AssertionError):inline.interpret_science({'status':'teacher_pretraining_control_failed_no_training','new_training_updates':1})

    def test_candidate_finished_is_not_full_goal_or_adoption(self):
        result=inline.interpret_science({'status':'finite_qualified_expert_v2_candidate_and_all_CUDA_assessments_complete','new_training_updates':1000})
        self.assertEqual(result['confirmed_new_updates'],1000);self.assertNotIn('candidate_adopted',result);self.assertNotIn('full_goal_complete',result)

    def runner_fault(self,wait):
        runner=inline.PhaseRunner(self.own,self.work,inline.Budget(time.monotonic()))
        child=mock.Mock(pid=12345)
        if isinstance(wait,BaseException):child.wait.side_effect=wait
        else:child.wait.return_value=wait
        cleanup={'live_members_after':[],'interrupted_during_cleanup':False}
        with mock.patch.object(inline.subprocess,'Popen',return_value=child),mock.patch.object(inline,'stop_group',return_value=cleanup) as stop:
            with self.assertRaises((subprocess.TimeoutExpired,RuntimeError)):runner.run('fault',['never-executed'],os.environ.copy(),initialization=True)
            stop.assert_called_once_with(child)
        self.assertEqual(runner.events[-1]['termination']['live_members_after'],[])

    def test_timeout_always_enters_group_cleanup_logic(self):self.runner_fault(subprocess.TimeoutExpired('fixture',.01))
    def test_nonzero_exit_always_enters_group_cleanup_logic(self):self.runner_fault(7)

    def test_selftest_rejects_missing_descendant_or_missing_timeout(self):
        runner=mock.Mock();runner.events=[{'termination':{'live_members_after':[],'members_before':[1],'signals':['SIGKILL']}}]
        runner.run.side_effect=subprocess.TimeoutExpired('fixture',.5)
        with self.assertRaises(AssertionError):inline.group_selftest(runner,'not-executed',{})
        runner.run.side_effect=None
        with self.assertRaises(AssertionError):inline.group_selftest(runner,'not-executed',{})

    def test_cleanup_logic_escalates_to_KILL_and_confirms_empty(self):
        child=mock.Mock(pid=12345,returncode=-9);child.poll.return_value=-9
        state={'alive':True};sent=[]
        def killpg(pgid,kind):
            sent.append(kind)
            if kind==inline.signal.SIGKILL:state['alive']=False
        with mock.patch.object(inline.signal,'SIGKILL',9,create=True),mock.patch.object(inline,'live_group',side_effect=lambda pgid:[12345,12346] if state['alive'] else []),mock.patch.object(inline.os,'killpg',side_effect=killpg,create=True):
            result=inline.stop_group(child,grace=0)
        self.assertEqual(sent,[inline.signal.SIGTERM,9]);self.assertEqual(result['live_members_after'],[])

    def test_disabled_startup_uses_one_origin_and_total_deadline(self):
        command=(builder.OUT/'private/inline-startup.command.txt').read_text()
        self.assertIn('@@PUBLIC_INLINE_ENVELOPE_HTTPS_URL@@',command)
        self.assertIn('timeout --signal=TERM --kill-after=10s 5380s',command)
        self.assertNotIn("export CUDA_VISIBLE_DEVICES=''",command)
        code=command.split("<<'INLINE_V2_INIT'\n",1)[1].rsplit('INLINE_V2_INIT',1)[0];compile(code,'inline-capsule','exec')
        self.assertLess(code.index('started=time.monotonic()'),code.index('urllib.request.urlopen'))
        self.assertLess(code.index('signal.setitimer(signal.ITIMER_REAL'),code.index('blob=fetch_source('))
        self.assertIn('started+585',code);self.assertIn('started+600',code)
        self.assertIn('INLINE_CAPSULE_FAILURE',code)
        self.assertIn('signal.signal(signal.SIGTERM,initialization_terminated)',code)
        self.assertIn((ROOT/'scripts/bounded_inline_source.py').read_text(encoding='utf8'),code)
        self.assertIn(self.sha,code);self.assertIn(cpu.file_digest(ROOT/'scripts/qualified_expert_v2_inline_init.py'),code)
        plan=json.loads((ROOT/'qualified-expert-v2-inline-init-plan.json').read_text(encoding='utf8'))
        self.assertFalse(plan['automatic_execution_allowed']);self.assertTrue(plan['GPU_allocated_during_CPU_preflight'])
        self.assertFalse(plan['CPU_preflight_is_zero_GPU_job']);self.assertEqual(plan['maximum_total_program_seconds'],5400)

    def test_capsule_source_URL_allows_only_exact_owned_immutable_raw_asset_or_existing_release(self):
        command=(builder.OUT/'private/inline-startup.command.txt').read_text()
        code=command.split("<<'INLINE_V2_INIT'\n",1)[1].rsplit('INLINE_V2_INIT',1)[0]
        function=next(x for x in ast.parse(code).body if isinstance(x,ast.FunctionDef) and x.name=='source_url_allowed')
        scope={'re':__import__('re')};exec(compile(ast.Module(body=[function],type_ignores=[]),'exact-source-allowlist','exec'),scope)
        allowed=scope['source_url_allowed'];commit='a'*40
        raw=f'https://raw.githubusercontent.com/skywhite1024/SuperMimic-Reproduction/{commit}/artifacts/qualified-expert-v2-inline-init/qualified-expert-v2-inline-init-envelope.tar.gz'
        self.assertTrue(allowed(raw));self.assertTrue(allowed('https://github.com/skywhite1024/SuperMimic-Reproduction/releases/download/v0.3.0/source.tar.gz'))
        for wrong in [raw.replace(commit,'main'),raw.replace(commit,'a'*39),raw.replace(commit,'g'*40),
            raw.replace('skywhite1024','other'),raw.replace('SuperMimic-Reproduction/','other/'),
            raw.replace('raw.githubusercontent.com','raw.githubusercontent.com.evil.example'),
            raw.replace('https://','http://'),raw+'?other=1',raw.replace('inline-init-envelope.tar.gz','other.tar.gz'),
            raw.replace(commit,'@@COMMIT@@')]:
            with self.subTest(wrong=wrong):self.assertFalse(allowed(wrong))

    @unittest.skipUnless(sys.platform=='linux','Native Linux process groups unavailable on this Windows host; WSL is not installed')
    def test_native_process_group_timeout_selftest(self):
        runner=inline.PhaseRunner(self.own,self.work,inline.Budget(time.monotonic()))
        proof=inline.group_selftest(runner,sys.executable,os.environ.copy())
        self.assertFalse(proof['termination']['live_members_after'])

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(InlinePreparation))
    receipt={'status':'offline_inline_logic_tests_passed_with_native_Linux_pending' if result.wasSuccessful() else 'offline_inline_tests_failed',
             'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':[{'test':x.id(),'reason':why} for x,why in result.skipped],
             'passed':result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),'Linux_native_process_groups_tested':sys.platform=='linux',
             'local_CPU_affinity':__import__('psutil').Process().cpu_affinity(),'local_GPU':0,'local_training_updates':0,'cloud_executed':False,
             'download_substitute_native_host':sys.platform,'download_substitute_proofs':DOWNLOAD_PROOFS,
             'Linux_initialization_signal_alarm_tests_are_mocked_not_native':sys.platform!='linux',
             'failure_details':[{'test':x.id(),'traceback':trace} for x,trace in result.failures+result.errors],
             'mocked_group_cleanup_is_logic_coverage_not_native_termination_proof':True}
    stamp=str(time.time_ns())
    (ROOT/f'evidence/qualified-expert-v2-inline-offline-tests-{stamp}.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (ROOT/'evidence/qualified-expert-v2-inline-offline-tests.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
    raise SystemExit(not result.wasSuccessful())
