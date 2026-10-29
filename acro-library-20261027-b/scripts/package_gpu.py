"""Small finite GPU command; all data and environment must already exist on AFS."""
import base64,gzip,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    hashes=json.loads((ROOT/'deployment/cpu-preparation-command.json').read_text())['payload_hashes']
    raw=(ROOT/'scripts/run_gpu_stage.py').read_bytes();encoded=base64.b64encode(gzip.compress(raw,mtime=0)).decode()
    command="""set -euo pipefail
umask 077
timeout --signal=TERM --kill-after=30s 7200s python3 -u - <<'PY'
import base64,gzip,hashlib,json,os,pathlib
root=pathlib.Path('/mnt/afs/USER_ID/SuperMimic/acro-library-20261027-b')
assert root.resolve().is_relative_to(pathlib.Path('/mnt/afs/USER_ID'))
for name,expected in HASHES.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
raw=gzip.decompress(base64.b64decode('ENCODED'));assert hashlib.sha256(raw).hexdigest()=='SCRIPT_SHA'
script=root/'scripts/run_gpu_stage.py';assert not script.exists() or script.read_bytes()==raw;script.write_bytes(raw)
base=pathlib.Path('/mnt/afs/USER_ID/SuperMimic/lafan1-20261024')
os.environ.update(SUPERMIMIC_UNI=str(base/'uni'),MUJOCO_GL='disable',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',WANDB_MODE='offline',PYTHONPATH=str(root/'scripts')+':'+str(base/'uni/src'))
os.execv(str(base/'uni/.venv/bin/python'),[str(base/'uni/.venv/bin/python'),'-u',str(script)])
PY
""".replace('HASHES',repr(hashes)).replace('SCRIPT_SHA',hashlib.sha256(raw).hexdigest()).replace('ENCODED',encoded)
    (ROOT/'deployment/gpu-unified-1027f1.command.txt').write_text(command,newline='\n')
    receipt={'status':'prepared_not_submitted','gpu_count':1,'maximum_seconds':7200,'new_updates':4000,'wm_only_updates':800,
             'bytes':len(command.encode()),'sha256':hashlib.sha256(command.encode()).hexdigest(),'source_hashes':hashes,
             'program_sha256':hashlib.sha256(raw).hexdigest(),'full55_evaluation_before_and_after':True}
    (ROOT/'deployment/gpu-unified-command.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))


if __name__=='__main__':main()
