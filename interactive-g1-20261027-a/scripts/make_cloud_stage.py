"""Build a small hash-verified CPU ACP deployment, reusing the existing AFS environment."""
import base64,gzip,hashlib,json
from pathlib import Path
from runtime import ROOT,UNI


def main():
    names=['runtime.py','compose_sequences.py','global_model.py','train_global.py']
    files={'scripts/'+n:(ROOT/'scripts'/n).read_text(encoding='utf-8') for n in names}
    hashes={n:hashlib.sha256(b.encode()).hexdigest() for n,b in files.items()}
    core={n:hashlib.sha256((UNI/n).read_text(encoding='utf-8').encode()).hexdigest() for n in ['src/unisupermimic/rl/algos/supermimic/learner.py','src/unisupermimic/rl/algos/supermimic/networks.py','src/unisupermimic/tasks/motion_tracking/g1/supermimic/mdp.py']}
    refs={str(s):hashlib.sha256((ROOT/'references'/f'seed-{s}/sequence.csv').read_text().encode()).hexdigest() for s in range(6)}
    body=base64.b64encode(gzip.compress(json.dumps({'files':files,'hashes':hashes,'core':core,'references_normalized_text_sha256':refs}).encode(),mtime=0)).decode()
    code="""import os,pathlib,json,gzip,base64,hashlib,subprocess,urllib.request
os.umask(0o077)
root=pathlib.Path('/mnt/afs/USER_ID/SuperMimic/interactive-g1-20261027-a')
base=pathlib.Path('/mnt/afs/USER_ID/SuperMimic/lafan1-20261024')
assert root.resolve().is_relative_to(pathlib.Path('/mnt/afs/USER_ID'))
root.mkdir(parents=True,exist_ok=True);(root/'evidence').mkdir(exist_ok=True)
assert not (root/'evidence/cloud-stage-ready.json').exists(),'Inspect existing completed preparation before rerun'
bundle=json.loads(gzip.decompress(base64.b64decode('PAYLOAD')))
for n,digest in bundle['core'].items():
    assert hashlib.sha256((base/'uni'/n).read_text().encode()).hexdigest()==digest,n
for name,text in bundle['files'].items():
    data=text.encode();assert hashlib.sha256(data).hexdigest()==bundle['hashes'][name]
    p=root/name;assert p.resolve().is_relative_to(root)
    assert not p.exists() or p.read_bytes()==data,'Existing source differs: '+name
    p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
full=base/'experiments/continuation-20261025-a/policy/full_016000.pt'
with full.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()=='0afcca9af97bd9db1f2dd083875898f962acab0b0a70c7313b221e575b70b05b'
env=os.environ.copy();env.update(SUPERMIMIC_UNI=str(base/'uni'),SUPERMIMIC_FULL=str(full),HF_HUB_OFFLINE='1',WANDB_MODE='offline',MUJOCO_GL='disable',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONPATH=str(root/'scripts')+':'+str(base/'uni/src'))
py=base/'uni/.venv/bin/python'
subprocess.run([str(py),'-u','-c','from compose_sequences import compose; print([compose(s) for s in range(6)])'],env=env,check=True,timeout=120)
for seed,digest in bundle['references_normalized_text_sha256'].items():
    p=root/'references'/('seed-'+seed)/'sequence.csv'
    assert hashlib.sha256(p.read_text().encode()).hexdigest()==digest,'Numeric reference changed: '+seed
subprocess.run([str(py),'-u',str(root/'scripts/global_model.py')],env=env,check=True,timeout=120)
result={'status':'world_position_sources_and_frozen_references_ready','gpu_count':0,'source_hashes':bundle['hashes'],'reference_text_hashes':bundle['references_normalized_text_sha256'],'cpu_max':pathlib.Path('/sys/fs/cgroup/cpu.max').read_text().strip(),'memory_max':pathlib.Path('/sys/fs/cgroup/memory.max').read_text().strip(),'BONES_requested':False,'original_baseline_modified':False}
(root/'evidence/cloud-stage-ready.json').write_text(json.dumps(result,indent=2)+'\\n')
print('CONTINUOUS_CPU_STAGE_PASS '+json.dumps(result),flush=True)
""".replace('PAYLOAD',body)
    command="set -euo pipefail\numask 077\ntimeout --signal=TERM --kill-after=20s 420s python3 -u - <<'PY'\n"+code+"PY\n"
    p=ROOT/'cloud-stage-20261027a.command.txt';p.write_text(command,encoding='utf-8',newline='\n')
    result={'status':'prepared_not_submitted','command':p.name,'bytes':len(command.encode()),'sha256':hashlib.sha256(command.encode()).hexdigest(),'gpu_count':0,'maximum_seconds':420,'payload_hashes':hashes,'new_raw_dataset_download':False}
    (ROOT/'evidence/cloud-stage-command.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='payload_hashes'}))


if __name__=='__main__':main()
