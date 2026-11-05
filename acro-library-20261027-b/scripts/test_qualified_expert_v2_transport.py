"""Offline integrity/privacy/adversarial tests; no network, CUDA or cloud actions."""
import copy,io,json,os,platform,re,shutil,subprocess,sys,tarfile,tempfile,unittest
from unittest import mock
from datetime import datetime,timezone
from pathlib import Path
import qualified_expert_v2_cpu_stage as cpu
import stage_qualified_expert_v2 as stage
import prepare_qualified_expert_v2_cpu_acp as builder

ROOT=Path(__file__).resolve().parents[1]

class TransportIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main_path=ROOT/'deployment/qualified-expert-v2-source-and-teacher.tar.gz'
        cls.control_path=ROOT/'deployment/qualified-expert-v2-control-supplement.tar.gz'
        _,cls.main=stage.audited_members(cls.main_path);_,cls.control=stage.audited_control(cls.control_path)
        cls.plan=json.loads(cls.main['qualified-expert-v2-plan.json'])
        cls.afs=cls.plan['new_teacher_cloud'].split('/SuperMimic/')[0];cls.local=str(ROOT.parent.parent)
        cls.packet_path=builder.OUT/'public/qualified-expert-v2-source-only-transport.tar.gz'
        cls.blob=cls.packet_path.read_bytes();cls.packet_sha=cpu.digest(cls.blob)
        cls.files=cpu.transport_files(cls.blob)
        cls.restored=cpu.restore_transport(cls.blob,cls.packet_sha,afs_user_root=cls.afs,local_workspace=cls.local)
        temp_root=ROOT/'tmp';temp_root.mkdir(exist_ok=True)
        cls.temporary=tempfile.TemporaryDirectory(prefix='qualified-v2-transport-',dir=temp_root)
        cls.work=Path(cls.temporary.name).resolve();assert cls.work.is_relative_to(temp_root.resolve())
        cls.teacher=ROOT.parent/'interactive-g1-20261027-a/policy/acro-h100-2000.pt'
        cls.repacked=cpu.repack(cls.restored,cls.teacher,cls.work/'good')

    @classmethod
    def tearDownClass(cls):
        assert cls.work.is_relative_to((ROOT/'tmp').resolve())
        cls.temporary.cleanup()

    def altered_archive(self,name,files):
        target=self.work/name;target.write_bytes(cpu.pack_bytes(files));return target

    def restore_altered(self,files,**kwargs):
        blob=cpu.pack_bytes(files)
        return cpu.restore_transport(blob,cpu.digest(blob),afs_user_root=kwargs.get('afs_user_root',self.afs),
                                     local_workspace=kwargs.get('local_workspace',self.local))

    def test_original_default_archive_path_and_preserved_helper(self):
        self.assertEqual(len(stage.audited_members(self.main_path)[1]),30)
        self.assertEqual(len(stage.audited_control(self.control_path)[1]),2)
        old=ROOT/'deployment/stage_qualified_expert_v2.original-75c4780c.py'
        self.assertEqual(cpu.file_digest(old),'75c4780c9ec82f50e6fe24907f7a434c453866944465147406a24131abc76985')
        result=subprocess.run([sys.executable,'-X','utf8',str(old),'--archive',str(self.main_path),
                               '--control-archive',str(self.control_path),'--audit-local-only'],capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_all_restored_members_are_original_bytes(self):
        self.assertEqual(self.restored['main'],{k:v for k,v in self.main.items() if k!=cpu.TEACHER_PATH})
        self.assertEqual(self.restored['control'],self.control)
        self.assertEqual(self.restored['helper'],(ROOT/'scripts/stage_qualified_expert_v2.py').read_bytes())
        self.assertEqual(cpu.digest(self.restored['main_manifest']),cpu.MAIN_MANIFEST_SHA)
        self.assertEqual(cpu.digest(self.restored['control_manifest']),cpu.CONTROL_MANIFEST_SHA)
        self.assertEqual(self.files['bootstrap/qualified_expert_v2_cpu_stage.py'],(ROOT/'scripts/qualified_expert_v2_cpu_stage.py').read_bytes())

    def test_repack_default_rejects_alternate_compression(self):
        self.assertNotEqual(self.repacked['main']['sha256'],cpu.MAIN_SHA)
        self.assertNotEqual(self.repacked['control']['sha256'],cpu.CONTROL_SHA)
        with self.assertRaises(AssertionError):stage.audited_members(Path(self.repacked['main']['path']))
        with self.assertRaises(AssertionError):stage.audited_control(Path(self.repacked['control']['path']))

    def test_explicit_repack_accepts_all_exact_canonical_bytes(self):
        self.assertEqual(stage.audited_members(Path(self.repacked['main']['path']),allow_manifest_identical_repack=True)[1],self.main)
        self.assertEqual(stage.audited_control(Path(self.repacked['control']['path']),allow_manifest_identical_repack=True)[1],self.control)

    def test_repack_CLI_requires_explicit_opt_in(self):
        command=[sys.executable,'-X','utf8',str(ROOT/'scripts/stage_qualified_expert_v2.py'),
                 '--archive',self.repacked['main']['path'],'--control-archive',self.repacked['control']['path'],'--audit-local-only']
        denied=subprocess.run(command,capture_output=True,text=True,timeout=30)
        self.assertNotEqual(denied.returncode,0)
        accepted=subprocess.run([*command,'--allow-manifest-identical-repack'],capture_output=True,text=True,timeout=30)
        self.assertEqual(accepted.returncode,0,accepted.stderr)
        result=json.loads(accepted.stdout)
        self.assertTrue(result['manifest_identical_repack_explicitly_enabled'])
        self.assertEqual(result['actual_archive_sha256'],self.repacked['main']['sha256'])
        self.assertEqual(result['canonical_archive_sha256'],cpu.MAIN_SHA)

    def test_changed_scientific_source_rejected(self):
        files=dict(self.main);files['bundle-manifest.json']=self.restored['main_manifest']
        files['scripts/run_qualified_expert_v2.py']+=b'\n# changed\n'
        target=self.altered_archive('changed-source.tar.gz',files)
        with self.assertRaises(AssertionError):stage.audited_members(target,allow_manifest_identical_repack=True)

    def test_changed_main_or_control_manifest_rejected(self):
        for group,source,manifest in [('main',self.main,self.restored['main_manifest']),('control',self.control,self.restored['control_manifest'])]:
            with self.subTest(group=group):
                files=dict(source);files['bundle-manifest.json' if group=='main' else 'control-manifest.json']=manifest+b' '
                target=self.altered_archive(f'changed-{group}-manifest.tar.gz',files)
                with self.assertRaises(AssertionError):
                    (stage.audited_members if group=='main' else stage.audited_control)(target,allow_manifest_identical_repack=True)

    def test_extra_and_missing_members_rejected(self):
        for kind in ['extra','missing']:
            with self.subTest(kind=kind):
                files=dict(self.control);files['control-manifest.json']=self.restored['control_manifest']
                if kind=='extra':files['unexpected.py']=b'print(1)\n'
                else:files.pop('deployment/qualified-expert-v2.command.txt')
                target=self.altered_archive(f'{kind}.tar.gz',files)
                with self.assertRaises(AssertionError):stage.audited_control(target,allow_manifest_identical_repack=True)

    def unsafe_tar(self,name,kind=tarfile.REGTYPE):
        out=io.BytesIO()
        with tarfile.open(fileobj=out,mode='w:gz') as archive:
            item=tarfile.TarInfo(name);item.type=kind
            if kind in [tarfile.SYMTYPE,tarfile.LNKTYPE]:item.linkname='../outside'
            archive.addfile(item,io.BytesIO(b''))
        return out.getvalue()

    def test_path_traversal_and_absolute_paths_rejected(self):
        for index,name in enumerate(['../escape.py','/absolute.py','scripts/../escape.py','scripts\\escape.py','C:/escape.py']):
            with self.subTest(name=name):
                blob=self.unsafe_tar(name);target=self.work/f'unsafe-{index}.tar.gz';target.write_bytes(blob)
                with self.assertRaises(AssertionError):stage.audited_members(target,allow_manifest_identical_repack=True)
                with self.assertRaises(AssertionError):cpu.transport_files(blob)

    def test_symlink_and_hardlink_rejected(self):
        for index,kind in enumerate([tarfile.SYMTYPE,tarfile.LNKTYPE]):
            with self.subTest(kind=kind):
                blob=self.unsafe_tar('scripts/linked.py',kind);target=self.work/f'linked-{index}.tar.gz';target.write_bytes(blob)
                with self.assertRaises(AssertionError):stage.audited_members(target,allow_manifest_identical_repack=True)
                with self.assertRaises(AssertionError):cpu.transport_files(blob)

    def test_transport_source_change_even_with_updated_template_metadata_rejected(self):
        files=dict(self.files);name='templates/main/scripts/train_qualified_expert_v2.py';files[name]+=b'\n# changed\n'
        meta=json.loads(files['transport-manifest.json']);row=next(x for x in meta['source_members'] if x['template']==name)
        row.update(template_bytes=len(files[name]),template_sha256=cpu.digest(files[name]))
        files['transport-manifest.json']=json.dumps(meta).encode()
        with self.assertRaises(AssertionError):self.restore_altered(files)

    def test_transport_canonical_manifest_change_rejected(self):
        files=dict(self.files);files['canonical/bundle-manifest.json']+=b' '
        with self.assertRaises(AssertionError):self.restore_altered(files)

    def test_wrong_placeholder_values_rejected_before_writes(self):
        for kwargs in [{'afs_user_root':'/mnt/afs/000000000'},{'local_workspace':self.local+'-wrong'}]:
            with self.subTest(kind=next(iter(kwargs))):
                with self.assertRaises(AssertionError):self.restore_altered(self.files,**kwargs)

    def test_actual_transport_digest_required(self):
        with self.assertRaises(AssertionError):cpu.restore_transport(self.blob,'0'*64,afs_user_root=self.afs,local_workspace=self.local)

    def test_public_package_independent_privacy_scan(self):
        self.assertEqual(len(self.files),36)
        self.assertTrue(all(not name.endswith(('.pt','.npz','.csv','.log','.pem','.key')) for name in self.files))
        forbidden=[self.afs.encode(),self.afs.rsplit('/',1)[-1].encode(),self.local.encode(),json.dumps(self.local)[1:-1].encode()]
        patterns=[rb'-----BEGIN [A-Z ]*PRIVATE KEY-----',rb'(?:hf_|ghp_|github_pat_|sk-)[A-Za-z0-9_\-]{20,}',rb'(?i)(?<![A-Za-z0-9])[A-Z]:[\\/]']
        for name,value in self.files.items():
            with self.subTest(name=name):
                self.assertFalse(any(word in value for word in forbidden))
                self.assertFalse(any(re.search(pattern,value) for pattern in patterns))
        for name in ['scripts/train_qualified_expert_v2.py','scripts/run_qualified_expert_v2.py','qualified-expert-v2-plan.json']:
            self.assertIn(cpu.AFS_SLOT,self.files['templates/main/'+name])

    def test_wrong_teacher_bytes_rejected(self):
        wrong=self.work/'wrong-teacher.pt';value=bytearray(self.teacher.read_bytes());value[0]^=1;wrong.write_bytes(value)
        with self.assertRaises(AssertionError):cpu.repack(self.restored,wrong,self.work/'bad-teacher')
        self.assertFalse((self.work/'bad-teacher').exists())

    def test_private_path_rule_detects_drives_without_matching_HTTPS(self):
        pattern=rb'(?i)(?<![A-Za-z0-9])[A-Z]:[\\/]'
        self.assertIsNotNone(re.search(pattern,b'"[local-user-path]"'))
        self.assertIsNotNone(re.search(pattern,b'"[local-user-path]"'))
        self.assertIsNone(re.search(pattern,b'https://github.com/example/source'))

    def test_resource_limits_reject_overallocation_and_unlimited(self):
        self.assertEqual(cpu.resource_limits('200000 100000','8589934592'),(2,8589934592))
        for cpu_value,memory in [('800000 100000',str(32*1024**3)),('max 100000','8589934592'),('200000 100000','max'),('0 100000','8589934592')]:
            with self.subTest(cpu=cpu_value,memory=memory):
                with self.assertRaises(AssertionError):cpu.resource_limits(cpu_value,memory)

    def test_all_frozen_scientific_sources_and_archives_unchanged(self):
        self.assertEqual(cpu.file_digest(self.main_path),cpu.MAIN_SHA);self.assertEqual(cpu.file_digest(self.control_path),cpu.CONTROL_SHA)
        self.assertEqual(cpu.file_digest(ROOT/'qualified-expert-v2-plan.json'),cpu.PLAN_SHA)
        self.assertEqual(len(self.plan['files']),25)
        for name,sha in self.plan['files'].items():self.assertEqual(cpu.file_digest(ROOT/name),sha,name)
        for name,value in self.restored['main'].items():
            if name.endswith('.py'):compile(value,name,'exec')

    def test_startup_remains_disabled_finite_CPU_only_draft(self):
        command=(builder.OUT/'private/CPU-ACP-startup.command.txt').read_text()
        self.assertIn('@@STAGING_ARCHIVE_HTTPS_URL@@',command)
        self.assertIn('timeout --signal=TERM --kill-after=10s 580s',command)
        self.assertIn("export CUDA_VISIBLE_DEVICES=''",command)
        code=command.split("<<'SUPERMIMIC_CPU_STAGE'\n",1)[1].rsplit('SUPERMIMIC_CPU_STAGE',1)[0]
        compile(code,'finite-capsule','exec')
        self.assertIn(self.packet_sha,code)
        self.assertIn(cpu.file_digest(ROOT/'scripts/qualified_expert_v2_cpu_stage.py'),code)
        self.assertNotIn('--execute-reviewed-plan',code)
        plan=json.loads((ROOT/'qualified-expert-v2-cpu-acp-stage-plan.json').read_text(encoding='utf8'))
        self.assertEqual(plan['transport']['sha256'],self.packet_sha)
        self.assertFalse(plan['automatic_execution_allowed']);self.assertFalse(plan['CPU2_RAM8_platform_support_verified'])
        self.assertEqual(plan['new_jobs_submitted'],0);self.assertEqual(plan['maximum_program_seconds'],600)

class StageFilesystem(unittest.TestCase):
    link_cases=[]

    def setUp(self):
        parent=ROOT/'tmp';parent.mkdir(exist_ok=True)
        self.temporary=tempfile.TemporaryDirectory(prefix='qualified-v2-paths-',dir=parent)
        self.base=Path(self.temporary.name).resolve();assert self.base.is_relative_to(parent.resolve())
        self.own,self.root,self.outside=self.fixture('default')

    def tearDown(self):
        assert self.base.is_relative_to((ROOT/'tmp').resolve())
        self.temporary.cleanup()

    def fixture(self,label):
        own=self.base/f'own-{label}';outside=self.base/f'outside-{label}'
        own.mkdir();outside.mkdir();root=own/'SuperMimic/acro-library-20261027-b';root.mkdir(parents=True)
        (outside/'untouched.txt').write_bytes(b'outside sentinel')
        return own,root,outside

    def tree(self,directory):
        return {str(p.relative_to(directory)):p.read_bytes() for p in directory.rglob('*') if p.is_file()}

    def link(self,path,target,*,directory,case):
        assert path.absolute().is_relative_to(self.base) and target.resolve().is_relative_to(self.base)
        try:
            os.symlink(target,path,target_is_directory=directory)
            kind='directory_symlink' if directory else 'file_symlink'
        except OSError as error:
            if not directory or os.name!='nt':
                self.skipTest(f'Real {"directory" if directory else "file"} symlink unavailable on {platform.system()}: errno={error.errno},winerror={getattr(error,"winerror",None)}; Linux behavior not covered')
            executable=shutil.which('powershell.exe') or shutil.which('pwsh')
            if executable is None:self.skipTest('No PowerShell available for a real Windows junction; Linux symlinks not covered')
            quote=lambda value:"'"+str(value).replace("'","''")+"'"
            command="$ErrorActionPreference='Stop'; New-Item -ItemType Junction -Path "+quote(path)+' -Target '+quote(target)+' | Out-Null'
            made=subprocess.run([executable,'-NoProfile','-NonInteractive','-Command',command],capture_output=True,text=True,timeout=10,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            if made.returncode!=0:self.skipTest('Real Windows directory junction unavailable; Linux symlink behavior not covered')
            self.assertTrue(path.is_junction());kind='Windows_directory_junction'
        self.link_cases.append({'case':case,'kind':kind,'host_platform':platform.system()})
        return kind

    def test_missing_deployment_and_evidence_parents_created(self):
        self.assertFalse((self.root/'deployment').exists());self.assertFalse((self.root/'evidence').exists())
        incoming=cpu.prepare_stage_directory(self.own,'incoming')
        self.assertTrue(incoming['incoming'].is_dir());self.assertTrue(incoming['evidence'].is_dir())
        work=cpu.prepare_stage_directory(self.own,'work')
        self.assertTrue(work['work'].is_dir())
        for key in ['deployment','evidence','incoming','work']:self.assertTrue(work[key].resolve().is_relative_to(self.own))

    def test_existing_stage_directories_are_not_reused(self):
        for phase in ['incoming','work']:
            created=cpu.prepare_stage_directory(self.own,phase)
            marker=created[phase]/'prior.txt';marker.write_bytes(b'keep')
            with self.assertRaises(AssertionError):cpu.prepare_stage_directory(self.own,phase)
            self.assertEqual(marker.read_bytes(),b'keep')

    def test_outside_or_parent_traversal_paths_rejected_without_writes(self):
        before=self.tree(self.outside)
        for path in [self.outside/'new.txt',self.own/'..'/self.outside.name/'new.txt']:
            with self.subTest(path=str(path)):
                with self.assertRaises(AssertionError):cpu.write_owned_bytes(self.own,path,b'forbidden')
        self.assertEqual(self.tree(self.outside),before)

    def test_root_deployment_evidence_directory_links_outside_rejected(self):
        for name in ['root','deployment','evidence']:
            with self.subTest(name=name):
                own,root,outside=self.fixture(name);target=outside/'target';target.mkdir()
                if name=='root':root.rmdir();link=root
                else:link=root/name
                self.link(link,target,directory=True,case=name+'_outside')
                before=self.tree(outside)
                with self.assertRaisesRegex(AssertionError,'Resolved path escapes'):cpu.prepare_stage_directory(own,'incoming')
                self.assertEqual(self.tree(outside),before)

    def test_owner_anchor_directory_link_rejected(self):
        alias=self.base/'owner-alias';self.link(alias,self.outside,directory=True,case='owner_anchor_outside')
        before=self.tree(self.outside)
        with self.assertRaises(AssertionError):cpu.prepare_stage_directory(alias,'incoming')
        self.assertEqual(self.tree(self.outside),before)

    def test_teacher_parent_link_outside_rejected_before_model_open(self):
        target=self.outside/'policy';target.mkdir();model=target/'qualified-expert-8808.pt';model.write_bytes(b'outside model sentinel')
        self.link(self.root/'policy',target,directory=True,case='teacher_parent_outside')
        before=self.tree(self.outside);opened=[];original=Path.open
        def observed(path,*args,**kwargs):
            if path.resolve()==model.resolve():opened.append(str(path))
            return original(path,*args,**kwargs)
        with mock.patch.object(Path,'open',observed):
            with self.assertRaisesRegex(AssertionError,'Resolved path escapes'):cpu.prepare_stage_directory(self.own,'incoming')
        self.assertEqual(opened,[]);self.assertEqual(self.tree(self.outside),before)
        self.assertFalse((self.root/'deployment').exists())

    def test_teacher_file_symlink_outside_rejected(self):
        (self.root/'policy').mkdir();target=self.outside/'model.pt';target.write_bytes(b'outside model')
        self.link(self.root/cpu.TEACHER_PATH,target,directory=False,case='teacher_file_outside')
        before=self.tree(self.outside)
        with self.assertRaises(AssertionError):cpu.prepare_stage_directory(self.own,'incoming')
        self.assertEqual(self.tree(self.outside),before)

    def receipt_link(self,*,dangling):
        (self.root/'evidence').mkdir();target=self.own/'receipt-target.json'
        if not dangling:target.write_bytes(b'prior receipt')
        receipt=self.root/'evidence'/f'{cpu.LABEL}.json'
        self.link(receipt,target,directory=False,case='receipt_dangling_inside' if dangling else 'receipt_existing_inside')
        before=target.read_bytes() if target.exists() else None
        with self.assertRaisesRegex(AssertionError,'Leaf symlinks'):cpu.prepare_stage_directory(self.own,'incoming')
        with self.assertRaisesRegex(AssertionError,'Leaf symlinks'):cpu.write_owned_bytes(self.own,receipt,b'forbidden')
        self.assertEqual(target.read_bytes() if target.exists() else None,before)
        self.assertFalse((self.root/'deployment').exists())

    def test_existing_receipt_file_symlink_inside_rejected(self):self.receipt_link(dangling=False)
    def test_dangling_receipt_file_symlink_inside_rejected(self):self.receipt_link(dangling=True)

    def test_existing_regular_receipt_preserved(self):
        (self.root/'evidence').mkdir();receipt=self.root/'evidence'/f'{cpu.LABEL}.json';receipt.write_bytes(b'prior receipt')
        with self.assertRaises(AssertionError):cpu.prepare_stage_directory(self.own,'incoming')
        self.assertEqual(receipt.read_bytes(),b'prior receipt');self.assertFalse((self.root/'deployment').exists())

    def test_parent_alias_resolving_inside_owner_is_allowed(self):
        target=self.own/'owned-deployment';target.mkdir()
        self.link(self.root/'deployment',target,directory=True,case='deployment_alias_inside')
        layout=cpu.prepare_stage_directory(self.own,'incoming')
        self.assertTrue(layout['incoming'].resolve().is_relative_to(self.own))
        self.assertTrue((target/layout['incoming'].name).is_dir())

if __name__=='__main__':
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(TransportIntegrity),
                             unittest.defaultTestLoader.loadTestsFromTestCase(StageFilesystem)])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    receipt={'status':'offline_integrity_privacy_and_adversarial_tests_passed' if result.wasSuccessful() else 'offline_tests_failed',
             'at':datetime.now(timezone.utc).isoformat(),'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
             'network_downloads_performed':0,'cloud_jobs_submitted':0,'GPU_or_training_executed':False,
             'actual_cloud_CPU_preflight_executed':False,'runtime_network_and_resource_release_not_validated_by_these_offline_tests':True,
             'host_platform':platform.system(),'Linux_real_filesystem_behavior_tested':platform.system()=='Linux',
             'skipped_tests':[{'test':test.id(),'reason':reason} for test,reason in result.skipped],
             'tests_passed':result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
             'real_filesystem_link_cases_exercised':StageFilesystem.link_cases,
             'actual_transport_sha256':getattr(TransportIntegrity,'packet_sha',None),
             'actual_test_repacked_archives':{k:{f:v[f] for f in ['bytes','sha256']} for k,v in getattr(TransportIntegrity,'repacked',{}).items()},
             'canonical_main_archive_sha256':cpu.MAIN_SHA,'canonical_control_archive_sha256':cpu.CONTROL_SHA,
             'first_privacy_rule_false_positive_preserved':'qualified-expert-v2-CPU-ACP-offline-tests-first-failure.json'}
    (ROOT/'evidence/qualified-expert-v2-CPU-ACP-offline-tests.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt));raise SystemExit(not result.wasSuccessful())
