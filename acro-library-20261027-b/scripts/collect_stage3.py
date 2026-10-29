"""Finite CPU-only recovery of the third stage, preserving raw and calibrated tests."""
import ast,hashlib,importlib.metadata,json,platform,sys,tarfile
from datetime import datetime,timezone
from pathlib import Path
import torch
from collect_results import digest,export_policy
ROOT=Path(__file__).resolve().parents[1]

def main():
    assert ROOT.resolve().is_relative_to(Path('/mnt/afs/USER_ID'))
    stage=json.loads((ROOT/'evidence/gpu-stage3-4000.json').read_text());assert stage['status']=='updates_corrected55_and_raw15_evaluation_complete'
    base=ROOT.parent/'lafan1-20261024';assets=base/'evidence/asset-downloads.json';source_manifest=base/'evidence/source-files.json'
    assert digest(assets)=='7b11d15769d720d858230bb862f7c78765657cef816500f4455466b1d2947ffb'
    old=[row for row in json.loads(assets.read_text())['files'] if '/lafan1_supermimic/' in row['path']];assert len(old)==40
    for row in old:assert digest(base/'uni'/row['path'])==row['sha256'],row['path']
    assert digest(source_manifest)=='ac553ceb0f0edfd39e5fd16c91dd6f0ab4ae62080fa7056e8a4c8bce7f86fcf6'
    source=json.loads(source_manifest.read_text());assert source['commit']=='faf7818e8abac3ec6abb8f00e6e904d3577f800b'
    for name,expected in source['files'].items():assert digest(base/'uni'/name)==expected,name
    for file in ['library-manifest.json','ground-calibration-v1/manifest.json']:
        for row in json.loads((ROOT/file).read_text())['files']:assert digest(ROOT/row['path'])==row['sha256'],row['path']
    run=ROOT/'runs/unified-h100-stage3-4000';proof=json.loads((run/'proof.json').read_text())
    assert proof['normalizer_tensors_unchanged'] and proof['new_updates']==4000 and len(proof['sampling']['clip_reset_samples'])==55
    assert min(proof['sampling']['clip_reset_samples'].values())>0
    assert digest(run/'final-full.pt')==stage['full_checkpoint_sha256']
    export=export_policy(run/'final-full.pt',ROOT/'policy/unified-h100-stage3-4000.pt')
    checkpoints=[{'path':str(path),'bytes':path.stat().st_size,'sha256':digest(path),'filesystem_modified_at_utc':datetime.fromtimestamp(path.stat().st_mtime,timezone.utc).isoformat()} for path in sorted(run.glob('*.pt'))]
    (ROOT/'evidence/checkpoint-inventory-stage3-4000.json').write_text(json.dumps({'checkpoints_retained_on_personal_AFS':checkpoints,'inference_export':export},indent=2)+'\n')
    learner=base/'uni/src/unisupermimic/rl/algos/supermimic/learner.py'
    fields=next(ast.literal_eval(node.value) for node in ast.parse(learner.read_text()).body if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id=='_LOG_FIELDS' for target in node.targets))
    log=run/'train_log.csv';lines=log.read_text().splitlines();assert lines and all(len(line.split(','))==len(fields) for line in lines)
    header=','.join(fields);has_header=lines[0]==header
    (run/'train_log_with_header.csv').write_text(('' if has_header else header+'\n')+log.read_text())
    (run/'train_log_schema.json').write_text(json.dumps({'fields':fields,'raw_file_had_header':has_header,'raw_file_preserved':True},indent=2)+'\n')
    versions={}
    for name in ['torch','numpy','mujoco','mujoco-warp','warp-lang','unilab','hydra-core']:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    (ROOT/'evidence/cloud-runtime-versions-stage3.json').write_text(json.dumps({'python':sys.version,'platform':platform.platform(),'packages':versions,
        'recorded_from':'CPU retrieval container using the same AFS venv','source_commit_verified_by_file_manifest':source['commit'],
        'old_lafan_40_and_raw15_and_corrected15_hashes_verified':True},indent=2)+'\n')
    names=['evidence/gpu-stage3-4000.json','evidence/checkpoint-inventory-stage3-4000.json','evidence/cloud-runtime-versions-stage3.json',
           'stage3-plan.json','library-manifest.json','ground-calibration-v1/manifest.json','runs/unified-h100-stage3-4000/final-full.pt','policy/unified-h100-stage3-4000.pt']
    for directory,pattern in [('scripts','*.py'),('data','*.csv'),('ground-calibration-v1/data','*.csv'),('runs/unified-h100-stage3-4000','*.json'),
                              ('runs/unified-h100-stage3-4000','*.csv'),('runs/unified-stage3-corrected-cuda-all55','*'),('runs/unified-stage3-raw-cuda-new15','*')]:
        names.extend(str(path.relative_to(ROOT)).replace('\\','/') for path in (ROOT/directory).glob(pattern) if path.is_file())
    names=sorted(set(names));files=[]
    for name in names:
        path=ROOT/name;assert path.resolve().is_relative_to(ROOT);files.append({'path':name,'bytes':path.stat().st_size,'sha256':digest(path)})
    manifest={'status':'collected_not_local_transfer_verified','gpu_stage':stage,'inference_export':export,'files':files,'file_count':len(files),
              'corrected_all55_and_raw_new15_full_trajectories_included':True,'intermediate_checkpoints_retained_on_personal_AFS':True,'credentials_included':False}
    receipt=ROOT/'evidence/results-manifest-stage3-4000.json';assert not receipt.exists();receipt.write_text(json.dumps(manifest,indent=2)+'\n')
    target=ROOT/'deployment/unified-stage3-4000-results.tar.gz';assert not target.exists()
    with tarfile.open(target,'w:gz',compresslevel=3) as archive:
        for name in names+['evidence/results-manifest-stage3-4000.json']:archive.add(ROOT/name,arcname=name,recursive=False)
    result={'archive':str(target),'bytes':target.stat().st_size,'sha256':digest(target),'manifest_files':len(files),'all_file_hashes_recorded':True,
            'local_transfer_verified':False,'manual_CCI_stop_and_DNAT_unbind_still_required':True}
    (ROOT/'evidence/archive-receipt-stage3-4000.json').write_text(json.dumps(result,indent=2)+'\n');print('UNIFIED_STAGE3_RESULTS_ARCHIVE_READY '+json.dumps(result),flush=True)

if __name__=='__main__':main()
