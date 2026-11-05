"""Audited own-AFS installation and CPU-only complete preflight for the frozenV2bundle."""
import argparse,hashlib,importlib.metadata,json,os,platform,sys,tarfile,time
from datetime import datetime,timezone
from pathlib import Path,PurePosixPath

ARCHIVE_SHA='3edc27b42a47a16adfb5ab1c8f1e95ec8ec754a345ac07e2ec17b2fe0f2d031f'
PLAN_SHA='bc5e8b397accd4f585732803385516b755538f924afebc9e73fbc205dc763476'
CONTROL_SHA='0b809a4011fbc6b66fe197e776d59104be545cfc08d45362e8145d516cc14712'
MANIFEST_SHA='1dfb777ef0e80c53e0b6db05db97a3d9197828a5f86893be4e222751b7733d5d'
CONTROL_MANIFEST_SHA='566caae65cfc0436a526a34849e48d1e43af6b49d346c8e2a7e8e9ea948f9666'
PARENT_TRACES={
    'retention-v1-parent-bound20-corrected-cuda-all55':'f3bdc12ed7f4394491d0542eb7b8d39d31669ef3f9038d680e5bca5934b2c751',
    'retention-v1-parent-bound20-raw-cuda-new15':'2cefb6c4d53a75926b83bad5e84b9e2a58e5607d480faf671a0a6dac9bd185ab',
    'retention-v1-parent-bound20-raw-continuous-cuda':'7752de78f8e2f3d26074d26890ee960c783d2581954ea940f980324990855775',
    'retention-v1-parent-bound20-corrected-continuous-cuda':'dde915ce40e3ae17d0908eb66a8576f272b5c7ae1518b429d70dac1c4054990b'}

def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def safe_member(member):
    name=member.name
    return (member.isfile() and bool(name) and not name.startswith('/') and '\\' not in name
            and ':' not in name and all(part not in ('','..','.') for part in name.split('/'))
            and 0<=member.size<=64*1024**2)


def audited_members(archive, *, allow_manifest_identical_repack=False):
    if not allow_manifest_identical_repack:assert sha(archive)==ARCHIVE_SHA
    with tarfile.open(archive,'r:gz') as bundle:
        members=bundle.getmembers();assert all(safe_member(member) for member in members)
        names=[member.name for member in members];assert len(names)==len(set(names)) and len(names)==31
        manifest_raw=bundle.extractfile('bundle-manifest.json').read();assert hashlib.sha256(manifest_raw).hexdigest()==MANIFEST_SHA
        manifest=json.loads(manifest_raw);assert len(manifest['files'])==30
        expected=[row['path'] for row in manifest['files']]
        assert len(expected)==len(set(expected)) and set(names)==set(expected)|{'bundle-manifest.json'}
        data={}
        for row in manifest['files']:
            value=bundle.extractfile(row['path']).read();assert len(value)==row['bytes'] and hashlib.sha256(value).hexdigest()==row['sha256'],row['path']
            data[row['path']]=value
        assert hashlib.sha256(data['qualified-expert-v2-plan.json']).hexdigest()==PLAN_SHA
    return manifest,data


def audited_control(archive, *, allow_manifest_identical_repack=False):
    if not allow_manifest_identical_repack:assert sha(archive)==CONTROL_SHA
    with tarfile.open(archive,'r:gz') as bundle:
        members=bundle.getmembers();expected={'control-manifest.json','deployment/qualified-expert-v2.command.txt','deployment/run_qualified_expert_v2_guarded.py'}
        assert len(members)==3 and {member.name for member in members}==expected and all(safe_member(member) for member in members)
        raw=bundle.extractfile('control-manifest.json').read();assert hashlib.sha256(raw).hexdigest()==CONTROL_MANIFEST_SHA
        control=json.loads(raw);assert len(control['files'])==2
        assert {row['path'] for row in control['files']}==expected-{'control-manifest.json'}
        data={}
        for row in control['files']:
            value=bundle.extractfile(row['path']).read();assert len(value)==row['bytes'] and hashlib.sha256(value).hexdigest()==row['sha256']
            data[row['path']]=value
    return control,data


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--archive',type=Path,required=True);parser.add_argument('--control-archive',type=Path,required=True);parser.add_argument('--audit-local-only',action='store_true');parser.add_argument('--root',type=Path)
    parser.add_argument('--allow-manifest-identical-repack',action='store_true',help='Accept alternate compression only with both exact canonical manifests and every exact original member')
    args=parser.parse_args();started=time.monotonic();manifest,data=audited_members(args.archive.resolve(),allow_manifest_identical_repack=args.allow_manifest_identical_repack)
    control,control_data=audited_control(args.control_archive.resolve(),allow_manifest_identical_repack=args.allow_manifest_identical_repack)
    assert not set(data)&set(control_data);data.update(control_data)
    archive_identity={'actual_archive_sha256':sha(args.archive),'canonical_archive_sha256':ARCHIVE_SHA,
                      'actual_control_archive_sha256':sha(args.control_archive),'canonical_control_archive_sha256':CONTROL_SHA,
                      'canonical_manifest_sha256':MANIFEST_SHA,'canonical_control_manifest_sha256':CONTROL_MANIFEST_SHA,
                      'manifest_identical_repack_explicitly_enabled':args.allow_manifest_identical_repack}
    if args.audit_local_only:
        print(json.dumps({'status':'all30_bundle_and2_control_members_verified_no_installation','archive_sha256':sha(args.archive),'control_sha256':sha(args.control_archive),'plan_sha256':PLAN_SHA,'bytes':args.archive.stat().st_size,'GPU':0,**archive_identity}));return
    root=args.root.resolve();assert root==Path('/mnt/afs/USER_ID/SuperMimic/acro-library-20261027-b')
    base=root.parent/'lafan1-20261024';uni=base/'uni';assert Path(sys.prefix).resolve()==(uni/'.venv').resolve()
    quota,period=Path('/sys/fs/cgroup/cpu.max').read_text().split();memory=Path('/sys/fs/cgroup/memory.max').read_text().strip()
    assert quota!='max' and int(quota)/int(period)<=2 and memory!='max' and int(memory)<=8*1024**3
    os.umask(0o077)
    # Check every existing destination before writing any new file; no replacement of conflicts.
    for name,value in data.items():
        target=root/name;assert target.resolve().is_relative_to(root)
        if target.exists():assert target.is_file() and not target.is_symlink() and target.read_bytes()==value,'Preserve conflicting AFS file: '+name
    written=[]
    for name,value in data.items():
        target=root/name
        if not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True)
            assert target.resolve().is_relative_to(root)
            with target.open('xb') as stream:stream.write(value)
            written.append(name)
        assert sha(target)==hashlib.sha256(value).hexdigest(),name
    plan=json.loads((root/'qualified-expert-v2-plan.json').read_text())
    for name,digest in plan['files'].items():assert sha(root/name)==digest,name
    for key in ['initial_full','old_teacher','new_teacher']:assert sha(Path(plan[key+'_cloud']))==plan[key+'_sha256'],key
    for name,digest in plan['matched_parent_result_sha256'].items():assert sha(root/name)==digest,name
    for mode,digest in PARENT_TRACES.items():
        folder=root/'runs'/mode;assert sha(folder/'trajectory.npz')==digest,mode
        summary=json.loads((folder/'summary.json').read_text())
        assert summary['checkpoint_sha256']==plan['initial_full_sha256'] and summary['device']=='cuda'
        assert summary['post_initialization_resets']==0 and summary['controller_variant']=='action-output-bound-20-v1'
        assert summary['reference_variant']==('ground-calibration-v1' if 'corrected' in mode else 'raw')
        if 'continuous' in mode:
            assert len(summary['trials'])==3 and all(row['elapsed_seconds']==120 and row['post_initialization_resets']==0 for row in summary['trials'])
        else:
            assert summary['backend']=='mjwarp' and summary['observation_dim']==101
            assert len(summary['clips'])==(55 if 'all55' in mode else 15)
            assert summary['groups']['new']['denominator']==15
            if 'all55' in mode:assert summary['groups']['lafan']['denominator']==40
    assets=base/'evidence/asset-downloads.json';sources=base/'evidence/source-files.json'
    assert sha(assets)=='7b11d15769d720d858230bb862f7c78765657cef816500f4455466b1d2947ffb'
    assert sha(sources)=='ac553ceb0f0edfd39e5fd16c91dd6f0ab4ae62080fa7056e8a4c8bce7f86fcf6'
    asset_rows=json.loads(assets.read_text())['files'];source_rows=json.loads(sources.read_text())
    assert source_rows['commit']=='faf7818e8abac3ec6abb8f00e6e904d3577f800b'
    for row in asset_rows:assert sha(uni/row['path'])==row['sha256'],row['path']
    for name,digest in source_rows['files'].items():assert sha(uni/name)==digest,name
    for name in ['library-manifest.json','ground-calibration-v1/manifest.json']:
        for row in json.loads((root/name).read_text())['files']:
            path=root/row['path'];assert path.resolve().is_relative_to(root) and sha(path)==row['sha256'],row['path']
    expected={'torch':'2.14.1','numpy':'2.5.4','mujoco':'3.11.0','mujoco-warp':'3.11.0','warp-lang':'1.16.0','unilab':'1.5.1','hydra-core':'1.3.7','scipy':'1.17.0'}
    versions={name:importlib.metadata.version(name) for name in expected};assert versions==expected
    assert sys.version_info[:2]==(3,12) and platform.machine()=='x86_64'
    os.environ.update(MUJOCO_GL='disable',CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',WANDB_MODE='offline',SUPERMIMIC_UNI=str(uni),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    sys.path.insert(0,str(root/'scripts'));sys.path.insert(0,str(uni/'src'))
    import torch,numpy as np
    from scipy.spatial.transform import Rotation
    from qualified_retention_learner import QualifiedRetentionLearner
    import evaluate_qualified_expert_v2
    assert torch.cuda.device_count()==0 and np.array_equal(Rotation.from_quat([0,0,0,1]).as_matrix(),np.eye(3))
    torch.set_num_threads(2)
    for key,dimension in [('initial_full',101),('old_teacher',99),('new_teacher',99)]:
        saved=torch.load(Path(plan[key+'_cloud']),map_location='cpu',weights_only=True);assert saved['pi']['obs_norm.mean'].numel()==dimension
        if key=='initial_full':assert saved['iteration']==30000
        del saved
    from evaluate_sequences import sequence_env
    lengths={}
    for folder,digest in [('continuous-frozen-v1','483ad3d2913eeca3d92d03e2b83805f011f145bf646ddb00e249a86a482f9542'),('continuous-corrected-v1','92fdae6b84705548718fb90a09eb018158bf15d66d6c3b0028c5ed4d791ca799')]:
        index=root/folder/'index.json';assert sha(index)==digest
        frozen=json.loads(index.read_text());assert frozen['seeds']==[0,1,2];files=[]
        for row in frozen['sequences']:
            path=root/folder/f'seed-{row["seed"]}'/'sequence.csv';assert sha(path)==row['reference_sha256']
            matrix=np.loadtxt(path,delimiter=',');assert matrix.shape[0]>=6001 and matrix.shape[1]==36 and np.isfinite(matrix).all();files.append(path)
        environment=sequence_env(files,101,'cpu')
        try:
            value=environment.supermimic_mdp.motion.clip_len.tolist();assert value==[6026]*3;lengths[folder]=value
        finally:environment.close();environment.backend.close()
    proof=json.loads((root/plan['six_update_engineering']).read_text());assert proof['new_updates']==6 and proof['normalizer_tensors_unchanged']
    assert proof['retention']['teacher_parameters_and_norms_unchanged'] and proof['retention']['teacher_gradients_absent']
    assert json.loads((root/plan['independent_readback']).read_text())['exact_output_equality']
    for name,digest in plan['files'].items():assert sha(root/name)==digest,name
    result={'status':'own_AFS_bundle_dependencies_models_all_inputs_parent_controls_and_sequence_constructors_verified',
            'checked_at':datetime.now(timezone.utc).isoformat(),'archive_sha256':sha(args.archive),'plan_sha256':PLAN_SHA,'installed_files':written,**archive_identity,
            'all30_bundle_files_hash_verified':True,'all2_control_files_hash_verified':True,'control_supplement_sha256':sha(args.control_archive),'actual_CPU_limit':int(quota)/int(period),'actual_memory_bytes':int(memory),'GPU_requested':0,
            'CUDA_visible_devices':0,'packages':versions,'all_original_asset_files_verified':len(asset_rows),'all_upstream_source_files_verified':len(source_rows['files']),
            'old_LAFAN_denominator':40,'new_raw_denominator':15,'new_calibrated_denominator':15,'fixed_sequence_constructor_lengths':lengths,
            'parent_iteration':30000,'engineering_checkpoint_not_used_as_parent':True,'new_training_updates':0,'new_GPU_job_submitted':False,
            'all4_parent_summary_and_full_trajectory_hashes_verified':True,'matched_parent_protocol_device_bound_reference_denominators_verified':True,
            'wall_seconds':time.monotonic()-started,'manual_stop_empty_DNAT_unbind_original_command_restore_still_required':True}
    out=root/'evidence/qualified-expert-v2-cloud-CPU-preflight.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n')
    print('QUALIFIED_EXPERT_V2_CLOUD_PREFLIGHT_PASS '+json.dumps(result),flush=True)

if __name__=='__main__':main()
