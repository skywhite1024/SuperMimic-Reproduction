"""Reviewable finite ACP startup command; preparation only, no resource creation."""
import base64,gzip,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    plan=json.loads((ROOT/'retention-trial-v1-plan.json').read_text())
    proof=json.loads((ROOT/'runs/engineering-retention-v2-cpu-6/proof.json').read_text())
    assert proof['new_updates']==6 and proof['device']=='cpu' and proof['normalizer_tensors_unchanged']
    assert proof['functional_bounded_policy_readback']['exact_output_equality'] and proof['retention']['teacher_parameters_and_norms_unchanged']
    assert proof['parent_sha256']==plan['initial_full_sha256']
    for name,expected in plan['files'].items():assert sha(ROOT/name)==expected,name
    names=list(plan['files'])+['retention-trial-v1-plan.json','scripts/run_retention_trial.py','scripts/audit_sequence_contacts.py',
                            'runs/engineering-retention-v2-cpu-6/proof.json','evidence/retention-native-method-diff.json']
    manifest={name:sha(ROOT/name) for name in names}
    payload={name:base64.b64encode((ROOT/name).read_bytes()).decode() for name in names}
    encoded=base64.b64encode(gzip.compress(json.dumps(payload).encode(),mtime=0)).decode()
    command="""set -euo pipefail
umask 077
timeout --signal=TERM --kill-after=30s 5400s python3 -u - <<'PY'
import base64,gzip,hashlib,json,os,pathlib
root=pathlib.Path('/mnt/afs/USER_ID/SuperMimic/acro-library-20261027-b')
assert root.resolve().is_relative_to(pathlib.Path('/mnt/afs/USER_ID'))
payload=json.loads(gzip.decompress(base64.b64decode('PAYLOAD')))
manifest=json.loads('MANIFEST')
for name,encoded in payload.items():
    target=root/name;assert target.resolve().is_relative_to(root)
    raw=base64.b64decode(encoded);assert hashlib.sha256(raw).hexdigest()==manifest[name]
    assert not target.exists() or target.read_bytes()==raw,name
    target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
plan=json.loads((root/'retention-trial-v1-plan.json').read_text())
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
for key in ['old_teacher','new_teacher']:assert hashlib.sha256(pathlib.Path(plan[key+'_cloud']).read_bytes()).hexdigest()==plan[key+'_sha256']
assert hashlib.sha256((root/plan['initial_full']).read_bytes()).hexdigest()==plan['initial_full_sha256']
print('RETENTION_PINNED_SOURCE_40OLD_15RAW_15CALIBRATED_2TEACHERS_VERIFIED',flush=True)
os.environ.update(SUPERMIMIC_UNI=str(base/'uni'),MUJOCO_GL='disable',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',WANDB_MODE='offline',PYTHONPATH=str(root/'scripts')+':'+str(base/'uni/src'))
python=str(base/'uni/.venv/bin/python');os.execv(python,[python,'-u',str(root/'scripts/run_retention_trial.py')])
PY
""".replace('PAYLOAD',encoded).replace('MANIFEST',json.dumps(manifest))
    compile(command.split("<<'PY'\n",1)[1].rsplit('\nPY',1)[0],'<retention-launch>','exec')
    target=ROOT/'deployment/gpu-retention-v1-1028.command.txt';assert not target.exists();target.write_text(command,newline='\n')
    receipt={'status':'prepared_not_submitted','command':str(target.relative_to(ROOT)),'bytes':len(command.encode()),
             'command_sha256':sha(target),'plan_sha256':sha(ROOT/'retention-trial-v1-plan.json'),
             'payload_files':manifest,'gpu_count':1,'maximum_seconds':5400,'maximum_new_updates':1000,
             'engineering_six_updates_and_functional_readback_verified':True,'credentials_included':False,
             'prerequisites_pending':'Current academy location/network and fresh all-account resource checks; preparation does not establish eligibility or resource availability.'}
    (ROOT/'deployment/gpu-retention-v1-command.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({key:value for key,value in receipt.items() if key!='payload_files'}),flush=True)

if __name__=='__main__':main()
