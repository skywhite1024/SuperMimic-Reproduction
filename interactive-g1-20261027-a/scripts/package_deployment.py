"""Package a scoped inference deployment, without environments, credentials or full weights."""
import hashlib
import json
import tarfile
from runtime import ROOT


def main():
    relative=['deployment/global-adapter-600-delta.pt','deployment/reconstruct_global_policy.py',
              'deployment/delta-receipt.json','scripts/runtime.py','scripts/global_model.py',
              'scripts/evaluate_continuous.py','scripts/evaluate_acro.py']
    for seed in range(12):
        relative.extend(f'references-lookahead/seed-{seed}/{name}' for name in ['sequence.csv','manifest.json'])
    rows=[{'path':name,'bytes':(ROOT/name).stat().st_size,'sha256':hashlib.sha256((ROOT/name).read_bytes()).hexdigest()} for name in relative]
    manifest={'purpose':'exact accepted basic-policy inference and fixed-reference evaluation',
              'not_complete_experiment_archive':True,'files':rows,'contains_credentials':False,
              'full_baseline_required_sha256':'0afcca9af97bd9db1f2dd083875898f962acab0b0a70c7313b221e575b70b05b',
              'old_environment_modification_required':False,'BONES_included':False}
    listing=ROOT/'deployment/manifest.json';listing.write_text(json.dumps(manifest,indent=2)+'\n')
    archive=ROOT/'deployment/basic-inference-frozen-12.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=6) as tar:
        for name in relative+['deployment/manifest.json']:tar.add(ROOT/name,arcname=name,recursive=False)
    # Every packaged byte is checked after reading the archive independently.
    with tarfile.open(archive,'r:gz') as tar:
        assert len(tar.getmembers())==len(relative)+1
        for row in rows:assert hashlib.sha256(tar.extractfile(row['path']).read()).hexdigest()==row['sha256']
    receipt={'path':str(archive),'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'files':len(relative)+1,'all_packaged_file_hashes_verified':True,'uploaded':False}
    (ROOT/'deployment/package-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))


if __name__=='__main__':main()
