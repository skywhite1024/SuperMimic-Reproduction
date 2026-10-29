"""Finite CPU result export from this run's own AFS directory."""
import hashlib,json,tarfile
from pathlib import Path
import torch
ROOT=Path('/mnt/afs/USER_ID/SuperMimic/interactive-g1-20261027-a')


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    receipt=json.loads((ROOT/'evidence/acro-gpu-result-receipt.json').read_text())
    full=ROOT/'runs/acro-h100-2000/final-full.pt';assert digest(full)==receipt['full_checkpoint_sha256']
    trajectory=ROOT/'runs/acro-h100-2000-retest/trajectory.npz';assert digest(trajectory)==receipt['trajectory_sha256']
    model=torch.load(full,map_location='cpu',weights_only=True)
    out=ROOT/'exports/acro-h100-2000';out.mkdir(parents=True,exist_ok=True)
    target=out/'policy.pt'
    if not target.exists():torch.save({'pi':model['pi'],'args':model['args'],'variant':model['variant']},target)
    loaded=torch.load(target,map_location='cpu',weights_only=True)
    assert all(torch.equal(v,loaded['pi'][k]) for k,v in model['pi'].items())
    relative=['evidence/acro-gpu-result-receipt.json','evidence/acro-stage-ready.json','runs/acro-h100-2000/proof.json',
              'runs/acro-h100-2000/args.json','runs/acro-h100-2000-retest/summary.json','runs/acro-h100-2000-retest/trajectory.npz',
              'scripts/train_acro.py','scripts/evaluate_acro.py','assets/acro-candidates/88_08_stageii_unitree_g1.csv','exports/acro-h100-2000/policy.pt']
    manifest={'purpose':'acro H100 result recovery; no full optimizer checkpoint included',
              'files':[{'path':name,'bytes':(ROOT/name).stat().st_size,'sha256':digest(ROOT/name)} for name in relative],
              'full_checkpoint_retained_on_AFS':str(full),'full_checkpoint_sha256':receipt['full_checkpoint_sha256'],
              'inference_pi_readback_exact':True,'cgroup_cpu_max':Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
              'cgroup_memory_max':Path('/sys/fs/cgroup/memory.max').read_text().strip()}
    listing=out/'manifest.json';listing.write_text(json.dumps(manifest,indent=2)+'\n')
    archive=ROOT/'exports/acro-h100-2000-results.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=3) as tar:
        for name in relative+['exports/acro-h100-2000/manifest.json']:tar.add(ROOT/name,arcname=name,recursive=False)
    print('ACRO_RESULTS_ARCHIVE '+json.dumps({'path':str(archive),'bytes':archive.stat().st_size,'sha256':digest(archive),'summary':receipt['summary']}),flush=True)


if __name__=='__main__':main()
