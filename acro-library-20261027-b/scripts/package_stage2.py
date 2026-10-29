"""Prepare a bounded ACP command from pinned scripts and local data transformations."""
import base64,gzip,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    proof=json.loads((ROOT/'runs/engineering-stage2-cpu-6/proof.json').read_text())
    assert proof['new_updates']==6 and proof['normalizer_tensors_unchanged'] and proof['device']=='cpu'
    assert all(proof['before'][key]!=proof['after'][key] for key in ['wm','pi','critic'])
    plan=json.loads((ROOT/'stage2-plan.json').read_text());assert proof['parent_sha256']==plan['initial_full_sha256']
    names=['scripts/train_ground_stage2.py','scripts/stage2_references.py','scripts/evaluate_ground_variant.py','scripts/run_gpu_stage2.py',
           'stage2-plan.json','ground-calibration-v1/manifest.json']
    payload={name:base64.b64encode((ROOT/name).read_bytes()).decode() for name in names}
    encoded=base64.b64encode(gzip.compress(json.dumps(payload).encode(),mtime=0)).decode()
    variant=json.loads((ROOT/'ground-calibration-v1/manifest.json').read_text())
    for row in variant['files']:
        source=ROOT/'data'/Path(row['path']).name;data=source.read_bytes();assert hashlib.sha256(data).hexdigest()==row['raw_sha256']
        if row['constant_root_z_shift_m']:
            lines=[]
            for line in data.decode().splitlines():
                columns=line.split(',');columns[2]=format(float(columns[2])+row['constant_root_z_shift_m'],'.12g');lines.append(','.join(columns))
            data=('\n'.join(lines)+'\n').encode()
        assert hashlib.sha256(data).hexdigest()==row['sha256']
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
variant=json.loads((root/'ground-calibration-v1/manifest.json').read_text())
for row in variant['files']:
    source=root/'data'/pathlib.Path(row['path']).name;data=source.read_bytes();assert hashlib.sha256(data).hexdigest()==row['raw_sha256']
    if row['constant_root_z_shift_m']:
        lines=[]
        for line in data.decode().splitlines():
            columns=line.split(',');columns[2]=format(float(columns[2])+row['constant_root_z_shift_m'],'.12g');lines.append(','.join(columns))
        data=('\\n'.join(lines)+'\\n').encode()
    assert hashlib.sha256(data).hexdigest()==row['sha256'],row['clip']
    target=root/row['path'];assert not target.exists() or target.read_bytes()==data
    target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
plan=json.loads((root/'stage2-plan.json').read_text())
for name,expected in plan['files'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
base=pathlib.Path('/mnt/afs/USER_ID/SuperMimic/lafan1-20261024')
os.environ.update(SUPERMIMIC_UNI=str(base/'uni'),MUJOCO_GL='disable',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',WANDB_MODE='offline',PYTHONPATH=str(root/'scripts')+':'+str(base/'uni/src'))
python=str(base/'uni/.venv/bin/python');os.execv(python,[python,'-u',str(root/'scripts/run_gpu_stage2.py')])
PY
""".replace('PAYLOAD',encoded)
    compile(command.split("<<'PY'\n",1)[1].rsplit('\nPY',1)[0],'<stage2-launch>','exec')
    target=ROOT/'deployment/gpu-stage2-1027g1.command.txt';assert not target.exists();target.write_text(command,newline='\n')
    receipt={'status':'prepared_not_submitted','new_updates':4000,'wm_only_updates':400,'gpu_count':1,'maximum_seconds':7200,
             'bytes':len(command.encode()),'command_sha256':hashlib.sha256(command.encode()).hexdigest(),
             'plan_sha256':hashlib.sha256((ROOT/'stage2-plan.json').read_bytes()).hexdigest(),
             'program_sha256':hashlib.sha256((ROOT/'scripts/run_gpu_stage2.py').read_bytes()).hexdigest(),
             'cpu_engineering_six_real_updates_passed':True,'ground_transform_all_15_exact_hashes_passed':True,'credentials_included':False}
    (ROOT/'deployment/gpu-stage2-command.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
