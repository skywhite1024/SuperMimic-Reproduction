"""Create a hash-checked small source payload for finite ACP preparation."""
import base64,gzip,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    names=['scripts/'+name for name in ['runtime.py','global_model.py','balanced_sampling.py','train_unified.py','evaluate_library.py','verify_score_contract.py','prepare_cloud.py']]
    names+=['library-manifest.json','library-audit.json','sources/motion-descriptions.json','evidence/parent-18k-receipt.json']
    previous=json.loads((ROOT/'deployment/cpu-preparation-command.json').read_text())['payload_hashes']
    payload={name:base64.b64encode((ROOT/name).read_bytes()).decode() for name in names}
    raw=json.dumps(payload,separators=(',',':')).encode();encoded=base64.b64encode(gzip.compress(raw,mtime=0)).decode()
    script="""set -euo pipefail
umask 077
timeout --signal=TERM --kill-after=30s 900s python3 -u - <<'PY'
import base64,gzip,hashlib,json,os,pathlib
root=pathlib.Path('/mnt/afs/USER_ID/SuperMimic/acro-library-20261027-b')
assert root.resolve().is_relative_to(pathlib.Path('/mnt/afs/USER_ID'))
data=gzip.decompress(base64.b64decode('PAYLOAD'));assert hashlib.sha256(data).hexdigest()=='PAYLOAD_SHA'
for name,value in json.loads(data).items():
    path=root/name;assert path.resolve().is_relative_to(root)
    raw=base64.b64decode(value);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.read_bytes()!=raw:
        old_sha=hashlib.sha256(path.read_bytes()).hexdigest();assert old_sha==PREVIOUS.get(name),'Unexpected differing source '+name
        backup=root/'evidence/source-before-1027e2'/name;backup.parent.mkdir(parents=True,exist_ok=True)
        assert not backup.exists() or backup.read_bytes()==path.read_bytes();backup.write_bytes(path.read_bytes())
    path.write_bytes(raw)
base=pathlib.Path('/mnt/afs/USER_ID/SuperMimic/lafan1-20261024')
os.environ.update(SUPERMIMIC_UNI=str(base/'uni'),MUJOCO_GL='disable',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',WANDB_MODE='offline',PYTHONPATH=str(root/'scripts')+':'+str(base/'uni/src'))
os.execv(str(base/'uni/.venv/bin/python'),[str(base/'uni/.venv/bin/python'),'-u',str(root/'scripts/prepare_cloud.py')])
PY
""".replace('PREVIOUS',repr(previous)).replace('PAYLOAD_SHA',hashlib.sha256(raw).hexdigest()).replace('PAYLOAD',encoded)
    out=ROOT/'deployment';out.mkdir(exist_ok=True)
    (out/'cpu-preparation-1027e2.command.txt').write_text(script,newline='\n')
    receipt={'status':'prepared_not_submitted','gpu_count':0,'maximum_seconds':900,'command_bytes':len(script.encode()),'command_sha256':hashlib.sha256(script.encode()).hexdigest(),'payload_hashes':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}}
    (out/'cpu-preparation-command.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))


if __name__=='__main__':main()
