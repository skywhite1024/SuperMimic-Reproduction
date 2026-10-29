"""Prepare one finite stage3 ACP command using existing AFS data and checkpoint."""
import base64,gzip,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    proof=json.loads((ROOT/'runs/engineering-stage3-cpu-6/proof.json').read_text())
    plan=json.loads((ROOT/'stage3-plan.json').read_text())
    assert proof['new_updates']==6 and proof['device']=='cpu' and proof['normalizer_tensors_unchanged']
    assert all(proof['before'][key]!=proof['after'][key] for key in ['wm','pi','critic'])
    assert proof['parent_sha256']==plan['initial_full_sha256']
    for name in ['wm_starts','pi_consumed']:assert proof['replay_sampling']['observed_sampling'][name]['count']>0
    tests=json.loads((ROOT/'evidence/balanced-replay-tests.json').read_text());assert tests['status']=='passed'
    names=['scripts/balanced_replay.py','scripts/train_replay_stage3.py','scripts/run_gpu_stage3.py','stage3-plan.json',
           'evidence/balanced-replay-tests.json','runs/engineering-stage3-cpu-6/proof.json',
           'acceptance-contract.json','contact-acceptance-contract.json','continuous-acceptance-contract-v1.json']
    prior=json.loads((ROOT/'stage2-plan.json').read_text())['files']
    assert all(name in names or name in prior or name=='stage2-plan.json' for name in plan['files']), 'Missing bootstrap file coverage'
    payload={name:base64.b64encode((ROOT/name).read_bytes()).decode() for name in names}
    encoded=base64.b64encode(gzip.compress(json.dumps(payload).encode(),mtime=0)).decode()
    command="""set -euo pipefail
umask 077
timeout --signal=TERM --kill-after=30s 7200s python3 -u - <<'PY'
import base64,gzip,hashlib,json,os,pathlib
root=pathlib.Path('/mnt/afs/USER_ID/SuperMimic/acro-library-20261027-b')
assert root.resolve().is_relative_to(pathlib.Path('/mnt/afs/USER_ID'))
payload=json.loads(gzip.decompress(base64.b64decode('PAYLOAD')))
for name,encoded in payload.items():
    target=root/name;assert target.resolve().is_relative_to(root);raw=base64.b64decode(encoded)
    assert not target.exists() or target.read_bytes()==raw,name
    target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
plan=json.loads((root/'stage3-plan.json').read_text())
for name,expected in plan['files'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
base=pathlib.Path('/mnt/afs/USER_ID/SuperMimic/lafan1-20261024')
assets=base/'evidence/asset-downloads.json';source_manifest=base/'evidence/source-files.json'
assert hashlib.sha256(assets.read_bytes()).hexdigest()=='7b11d15769d720d858230bb862f7c78765657cef816500f4455466b1d2947ffb'
old=[row for row in json.loads(assets.read_text())['files'] if '/lafan1_supermimic/' in row['path']];assert len(old)==40
for row in old:assert hashlib.sha256((base/'uni'/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
assert hashlib.sha256(source_manifest.read_bytes()).hexdigest()=='ac553ceb0f0edfd39e5fd16c91dd6f0ab4ae62080fa7056e8a4c8bce7f86fcf6'
source=json.loads(source_manifest.read_text());assert source['commit']=='faf7818e8abac3ec6abb8f00e6e904d3577f800b'
for name,expected in source['files'].items():assert hashlib.sha256((base/'uni'/name).read_bytes()).hexdigest()==expected,name
for filename in ['library-manifest.json','ground-calibration-v1/manifest.json']:
    for row in json.loads((root/filename).read_text())['files']:assert hashlib.sha256((root/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
print('STAGE3_PINNED_SOURCE_40OLD_15RAW_15CALIBRATED_VERIFIED',flush=True)
os.environ.update(SUPERMIMIC_UNI=str(base/'uni'),MUJOCO_GL='disable',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',WANDB_MODE='offline',PYTHONPATH=str(root/'scripts')+':'+str(base/'uni/src'))
python=str(base/'uni/.venv/bin/python');os.execv(python,[python,'-u',str(root/'scripts/run_gpu_stage3.py')])
PY
""".replace('PAYLOAD',encoded)
    compile(command.split("<<'PY'\n",1)[1].rsplit('\nPY',1)[0],'<stage3-launch>','exec')
    target=ROOT/'deployment/gpu-stage3-1027h2.command.txt';assert not target.exists();target.write_text(command,newline='\n')
    receipt={'status':'prepared_not_submitted','new_updates':4000,'wm_only_updates':400,'gpu_count':1,'maximum_seconds':7200,
             'bytes':len(command.encode()),'command_sha256':hashlib.sha256(command.encode()).hexdigest(),
             'plan_sha256':hashlib.sha256((ROOT/'stage3-plan.json').read_bytes()).hexdigest(),
             'program_sha256':hashlib.sha256((ROOT/'scripts/run_gpu_stage3.py').read_bytes()).hexdigest(),
             'real_CPU_six_updates_and_readback_verified':True,'normalizers_fixed':True,'credentials_included':False}
    (ROOT/'deployment/gpu-stage3-command-h2.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))


if __name__=='__main__':main()
