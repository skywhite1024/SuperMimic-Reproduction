"""Finite own-AFS CPU collection after the ACP candidate reaches a terminal state."""
import argparse,ast,hashlib,importlib.metadata,json,platform,sys,tarfile
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
MODES=['retention-v1-parent-bound20-corrected-cuda-all55','retention-v1-parent-bound20-raw-cuda-new15',
       'retention-v1-parent-bound20-raw-continuous-cuda','retention-v1-parent-bound20-corrected-continuous-cuda',
       'retention-v1-corrected-cuda-all55','retention-v1-raw-cuda-new15',
       'retention-v1-raw-continuous-cuda','retention-v1-corrected-continuous-cuda']
def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--platform-status',choices=['success','failed','stopped'],required=True)
    parser.add_argument('--attempt',choices=['attempt-a','resume-b'],required=True);args=parser.parse_args()
    label='retention-v1-1000-'+args.attempt
    assert ROOT.resolve().is_relative_to(Path('/mnt/afs/USER_ID'))
    plan=json.loads((ROOT/'retention-trial-v1-plan.json').read_text())
    for name,digest in plan['files'].items():assert sha(ROOT/name)==digest,name
    base=ROOT.parent/'lafan1-20261024';assets=base/'evidence/asset-downloads.json';source_manifest=base/'evidence/source-files.json'
    assert sha(assets)=='7b11d15769d720d858230bb862f7c78765657cef816500f4455466b1d2947ffb'
    old=[row for row in json.loads(assets.read_text())['files'] if '/lafan1_supermimic/' in row['path']];assert len(old)==40
    for row in old:assert sha(base/'uni'/row['path'])==row['sha256'],row['path']
    assert sha(source_manifest)=='ac553ceb0f0edfd39e5fd16c91dd6f0ab4ae62080fa7056e8a4c8bce7f86fcf6'
    source=json.loads(source_manifest.read_text());assert source['commit']=='faf7818e8abac3ec6abb8f00e6e904d3577f800b'
    for name,digest in source['files'].items():assert sha(base/'uni'/name)==digest,name
    for manifest_name in ['library-manifest.json','ground-calibration-v1/manifest.json']:
        for row in json.loads((ROOT/manifest_name).read_text())['files']:assert sha(ROOT/row['path'])==row['sha256'],row['path']
    assert sha(ROOT/plan['initial_full'])==plan['initial_full_sha256']
    for key in ['old_teacher','new_teacher']:assert sha(Path(plan[key+'_cloud']))==plan[key+'_sha256']
    stage_file=ROOT/('evidence/gpu-retention-v1-1000.json' if args.attempt=='attempt-a' else 'evidence/gpu-retention-v1-1000-resume1030b.json');stage=json.loads(stage_file.read_text()) if stage_file.exists() else {'status':'missing_program_receipt'}
    run=ROOT/'runs/unified-retention-v1-h100-1000';complete=stage['status']=='finite_candidate_and_all_cuda_assessments_complete'
    if args.platform_status=='success':assert complete, 'Successful platform job requires the complete finite program receipt'
    proof_path=run/'proof.json'
    if complete:
        proof=json.loads(proof_path.read_text());assert proof['new_updates']==1000 and proof['normalizer_tensors_unchanged']
        assert proof['functional_bounded_policy_readback']['exact_output_equality'] and proof['retention']['teacher_parameters_and_norms_unchanged']
        assert sha(run/'final-full.pt')==stage['full_checkpoint_sha256']
        assert sha(ROOT/'policy/unified-retention-v1-h100-1000.pt')==stage['policy_sha256']
        for mode in MODES:assert (ROOT/'runs'/mode/'summary.json').exists(),mode
    checkpoints=[{'path':str(path.relative_to(ROOT)),'bytes':path.stat().st_size,'sha256':sha(path)} for path in sorted(run.glob('*.pt'))]
    inventory=ROOT/f'evidence/checkpoint-inventory-{label}.json';assert not inventory.exists()
    inventory.write_text(json.dumps({'checkpoints_retained_on_personal_AFS':checkpoints,'program_complete':complete},indent=2)+'\n')
    learner=base/'uni/src/unisupermimic/rl/algos/supermimic/learner.py'
    fields=next(ast.literal_eval(node.value) for node in ast.parse(learner.read_text()).body if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id=='_LOG_FIELDS' for target in node.targets))
    log=run/'train_log.csv'
    if log.exists():
        lines=log.read_text().splitlines();assert lines and all(len(line.split(','))==len(fields) for line in lines)
        header=','.join(fields);has_header=lines[0]==header
        (run/'train_log_with_header.csv').write_text(('' if has_header else header+'\n')+log.read_text())
        (run/'train_log_schema.json').write_text(json.dumps({'fields':fields,'raw_file_had_header':has_header,'raw_file_preserved':True},indent=2)+'\n')
    versions={}
    for name in ['torch','numpy','mujoco','mujoco-warp','warp-lang','unilab','hydra-core']:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    runtime=ROOT/f'evidence/cloud-runtime-versions-{label}.json';runtime.write_text(json.dumps({'python':sys.version,'platform':platform.platform(),'packages':versions,'recorded_from':'Finite CPU collection using same AFS venv','source_and_all_reference_hashes_verified':True},indent=2)+'\n')
    names=set(plan['files'])|{'retention-trial-v1-plan.json','library-manifest.json','ground-calibration-v1/manifest.json',
          'continuous-frozen-v1/index.json','continuous-corrected-v1/index.json','scripts/run_retention_trial.py','scripts/audit_sequence_contacts.py',
          str(inventory.relative_to(ROOT)),str(runtime.relative_to(ROOT))}
    if stage_file.exists():names.add(str(stage_file.relative_to(ROOT)))
    if (run/'final-full.pt').exists():names.add(str((run/'final-full.pt').relative_to(ROOT)))
    elif (run/'interrupted-full.pt').exists():names.add(str((run/'interrupted-full.pt').relative_to(ROOT)))
    policy=ROOT/'policy/unified-retention-v1-h100-1000.pt'
    if policy.exists():names.add(str(policy.relative_to(ROOT)))
    for folder in [run,*[ROOT/'runs'/mode for mode in MODES]]:
        for path in folder.glob('*'):
            if path.is_file() and path.suffix in ['.json','.csv','.npz']:names.add(str(path.relative_to(ROOT)))
    files=[]
    for name in sorted(names):
        path=ROOT/name;assert path.resolve().is_relative_to(ROOT) and path.is_file(),name
        files.append({'path':name,'bytes':path.stat().st_size,'sha256':sha(path)})
    manifest={'status':'collected_not_local_transfer_verified','platform_status_from_attended_controller':args.platform_status,
              'program_receipt':stage,'program_complete':complete,'files':files,'file_count':len(files),
              'intermediate_checkpoints_retained_on_personal_AFS':True,'raw_data_or_credentials_included':False,
              'all_existing_matched_parent_candidate_cuda_trajectories_and_contact_samples_included':True}
    manifest_name=f'evidence/results-manifest-{label}.json'
    manifest_file=ROOT/manifest_name;assert not manifest_file.exists();manifest_file.write_text(json.dumps(manifest,indent=2)+'\n')
    archive=ROOT/f'deployment/{label}-results.tar.gz';assert not archive.exists()
    # The archive is built from the exact allowlist, never a recursive environment directory.
    with tarfile.open(archive,'w:gz',compresslevel=3) as bundle:
        for name in [*sorted(names),manifest_name]:bundle.add(ROOT/name,arcname=name,recursive=False)
    result={'archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),'manifest_files':len(files),
            'program_complete':complete,'all_file_hashes_recorded':True,'local_transfer_verified':False,
            'manual_CCI_stop_and_DNAT_unbind_still_required':True}
    (ROOT/f'evidence/archive-receipt-{label}.json').write_text(json.dumps(result,indent=2)+'\n')
    print('RETENTION_RESULTS_ARCHIVE_READY '+json.dumps(result),flush=True)

if __name__=='__main__':main()
