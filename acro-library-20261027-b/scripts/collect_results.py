"""Finite CPU-only collection, hash inventory, exact inference exports and archive."""
import ast,hashlib,importlib.metadata,json,platform,sys,tarfile
from datetime import datetime,timezone
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def export_policy(source,target):
    assert not target.exists();saved=torch.load(source,map_location='cpu',weights_only=True)
    torch.save({key:saved[key] for key in ['pi','args','iteration','variant']},target)
    loaded=torch.load(target,map_location='cpu',weights_only=True)
    assert all(torch.equal(value,loaded['pi'][key]) for key,value in saved['pi'].items())
    return {'source':str(source),'source_sha256':digest(source),'output':str(target.relative_to(ROOT)),'output_sha256':digest(target),'exact_pi_tensor_readback':True,'observation_dim':saved['pi']['obs_norm.mean'].numel()}


def main():
    assert ROOT.resolve().is_relative_to(Path('/mnt/afs/USER_ID'))
    stage=json.loads((ROOT/'evidence/gpu-stage-4000.json').read_text())
    assert stage['status']=='training_and_full55_evaluation_complete','Collect partial runs with a separately explicit inventory'
    base=ROOT.parent/'lafan1-20261024';assets=base/'evidence/asset-downloads.json'
    assert digest(assets)=='7b11d15769d720d858230bb862f7c78765657cef816500f4455466b1d2947ffb'
    old=[row for row in json.loads(assets.read_text())['files'] if '/lafan1_supermimic/' in row['path']];assert len(old)==40
    for row in old:assert digest(base/'uni'/row['path'])==row['sha256'],row['path']
    source_manifest=base/'evidence/source-files.json';assert digest(source_manifest)=='ac553ceb0f0edfd39e5fd16c91dd6f0ab4ae62080fa7056e8a4c8bce7f86fcf6'
    source=json.loads(source_manifest.read_text());assert source['commit']=='faf7818e8abac3ec6abb8f00e6e904d3577f800b'
    for name,expected in source['files'].items():assert digest(base/'uni'/name)==expected,name
    versions={}
    for name in ['torch','numpy','mujoco','mujoco-warp','warp-lang','unilab','hydra-core']:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    (ROOT/'evidence/cloud-runtime-versions-4000.json').write_text(json.dumps({'python':sys.version,'platform':platform.platform(),'packages':versions,
            'recorded_from':'CPU retrieval container reusing the same personal AFS venv as the GPU stage','source_commit_verified_by_file_manifest':source['commit'],
            'old_lafan_40_match_original_frozen_manifest':True,'old_lafan_manifest_sha256':digest(assets)},indent=2)+'\n')
    run=ROOT/'runs/unified-h100-4000';proof=json.loads((run/'proof.json').read_text());assert proof['normalizer_tensors_unchanged']
    policies=ROOT/'policy';policies.mkdir(exist_ok=True)
    exports=[export_policy(ROOT/'runs/initial-18k-101d/full.pt',policies/'initial-18k-101d.pt'),export_policy(run/'final-full.pt',policies/'unified-h100-4000.pt')]
    checkpoints=[{'path':str(path),'bytes':path.stat().st_size,'sha256':digest(path),'filesystem_modified_at_utc':datetime.fromtimestamp(path.stat().st_mtime,timezone.utc).isoformat()} for path in sorted(run.glob('*.pt'))]
    (ROOT/'evidence/checkpoint-inventory-4000.json').write_text(json.dumps({'checkpoints_retained_on_personal_AFS':checkpoints,'exports':exports},indent=2)+'\n')
    learner=ROOT.parent/'lafan1-20261024/uni/src/unisupermimic/rl/algos/supermimic/learner.py'
    fields=next(ast.literal_eval(node.value) for node in ast.parse(learner.read_text()).body if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id=='_LOG_FIELDS' for target in node.targets))
    log=run/'train_log.csv';lines=log.read_text().splitlines();assert lines and all(len(line.split(','))==len(fields) for line in lines)
    header=','.join(fields);has_header=lines[0]==header
    (run/'train_log_with_header.csv').write_text(('' if has_header else header+'\n')+log.read_text())
    (run/'train_log_schema.json').write_text(json.dumps({'fields':fields,'raw_file_had_header':has_header,'raw_file_preserved':True},indent=2)+'\n')
    names=['evidence/gpu-stage-4000.json','evidence/preparation-ready.json','evidence/checkpoint-inventory-4000.json','evidence/parent-18k-receipt.json','evidence/cloud-runtime-versions-4000.json',
           'library-manifest.json','library-audit.json','sources/motion-descriptions.json','runs/unified-h100-4000/final-full.pt']
    for directory,pattern in [('scripts','*.py'),('data','*.csv'),('policy','*.pt'),('runs/unified-h100-4000','*.json'),('runs/unified-h100-4000','*.csv'),
                              ('runs/initial101d-cuda-all55','*'),('runs/unified4000-cuda-all55','*')]:
        names.extend(str(path.relative_to(ROOT)).replace('\\','/') for path in (ROOT/directory).glob(pattern) if path.is_file())
    names=sorted(set(names));files=[]
    for name in names:
        path=ROOT/name;assert path.resolve().is_relative_to(ROOT)
        files.append({'path':name,'bytes':path.stat().st_size,'sha256':digest(path)})
    manifest={'status':'collected_not_local_transfer_verified','gpu_stage':stage,'inference_exports':exports,'files':files,'file_count':len(files),
              'all_55_trajectories_included':True,'intermediate_checkpoints_retained_on_personal_AFS':True,'credentials_included':False}
    receipt=ROOT/'evidence/results-manifest-4000.json';assert not receipt.exists();receipt.write_text(json.dumps(manifest,indent=2)+'\n')
    target=ROOT/'deployment/unified-h100-4000-results.tar.gz';target.parent.mkdir(exist_ok=True);assert not target.exists()
    with tarfile.open(target,'w:gz',compresslevel=3) as archive:
        for name in names+['evidence/results-manifest-4000.json']:archive.add(ROOT/name,arcname=name,recursive=False)
    result={'archive':str(target),'bytes':target.stat().st_size,'sha256':digest(target),'manifest_files':len(files),'all_file_hashes_recorded':True,
            'local_transfer_verified':False,'manual_CCI_stop_and_DNAT_unbind_still_required':True}
    (ROOT/'evidence/archive-receipt-4000.json').write_text(json.dumps(result,indent=2)+'\n');print('UNIFIED_RESULTS_ARCHIVE_READY '+json.dumps(result),flush=True)


if __name__=='__main__':main()
