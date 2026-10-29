"""Finite CPU preparation using the existing pinned runtime and accepted full checkpoint."""
import base64,hashlib,json,os,subprocess,sys,urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/mnt/afs/USER_ID/SuperMimic/lafan1-20261024')
PARENT=Path('/mnt/afs/USER_ID/SuperMimic/interactive-g1-20261027-a/runs/acro-h100-2000/final-full.pt')


def main():
    assert ROOT.resolve().is_relative_to(Path('/mnt/afs/USER_ID'))
    ready=ROOT/'evidence/preparation-ready.json';assert not ready.exists(),'Inspect existing preparation before retrying'
    parent_receipt=json.loads((ROOT/'evidence/parent-18k-receipt.json').read_text())
    expected=parent_receipt['full_checkpoint_sha256'];assert len(expected)==64
    with PARENT.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==expected
    manifest=json.loads((ROOT/'library-manifest.json').read_text());assert len(manifest['files'])==15
    for row in manifest['files']:
        target=ROOT/row['path'];target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            request=urllib.request.Request('https://api.github.com/repos/lez666/chunwan-flip/git/blobs/'+row['git_blob'],headers={'User-Agent':'SuperMimic-reproduction','Accept':'application/vnd.github+json'})
            with urllib.request.urlopen(request,timeout=45) as response:blob=json.load(response)
            raw=base64.b64decode(blob['content']);assert len(raw)==row['bytes']
            assert hashlib.sha256(raw).hexdigest()==row['sha256']
            assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==row['git_blob']
            target.write_bytes(raw)
        assert target.stat().st_size==row['bytes'] and hashlib.sha256(target.read_bytes()).hexdigest()==row['sha256']
        print('UNIFIED_REFERENCE_VERIFIED '+target.name,flush=True)
    env=os.environ.copy();env.update(SUPERMIMIC_UNI=str(BASE/'uni'),MUJOCO_GL='disable',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',WANDB_MODE='offline',PYTHONPATH=str(ROOT/'scripts')+':'+str(BASE/'uni/src'))
    os.environ.update(env)
    from global_model import convert_checkpoint
    initial=ROOT/'runs/initial-18k-101d/full.pt';assert not initial.exists()
    convert_checkpoint(PARENT,initial)
    py=str(BASE/'uni/.venv/bin/python')
    subprocess.run([py,'-u',str(ROOT/'scripts/train_unified.py'),'--initial',str(initial),'--label','cloud-cpu-engineering',
                    '--device','cpu','--steps','6','--wm-warmup','2','--num-envs','4','--engineering'],env=env,check=True,timeout=120)
    proof=json.loads((ROOT/'runs/cloud-cpu-engineering/proof.json').read_text())
    assert proof['normalizer_tensors_unchanged'] and proof['new_updates']==6
    subprocess.run([py,'-u',str(ROOT/'scripts/verify_score_contract.py'),'--checkpoint',str(initial),
                    '--label','cloud-initial-score-contract-v2','--clips','88_08_stageii_unitree_g1'],env=env,check=True,timeout=120)
    contract=json.loads((ROOT/'evidence/cloud-initial-score-contract-v2.json').read_text());assert contract['passed']
    result={'status':'cpu_preparation_passed','gpu_count':0,'new_clips_verified':15,'old_clips_expected':40,
            'parent_checkpoint':str(PARENT),'initial_101d_checkpoint':str(initial),'initial_sha256':hashlib.sha256(initial.read_bytes()).hexdigest(),
            'conversion':json.loads(initial.with_suffix('.json').read_text()),'cpu_engineering':proof,
            'original_score_contract':contract,'new_dependencies_installed':False,'BONES_route':'paused'}
    ready.parent.mkdir(parents=True,exist_ok=True);ready.write_text(json.dumps(result,indent=2)+'\n')
    print('UNIFIED_CPU_PREPARATION_PASS '+json.dumps(result),flush=True)


if __name__=='__main__':main()
