"""Package the tested acrobatic trainer and a finite isolated CPU preparation command."""
import base64,gzip,hashlib,json
from runtime import ROOT


def main():
    code=(ROOT/'scripts/train_acro.py').read_text()
    payload=base64.b64encode(gzip.compress(code.encode(),mtime=0)).decode()
    expected=json.loads((ROOT/'evidence/cloud-stage-command.json').read_text())['payload_hashes']
    script="""set -euo pipefail
umask 077
timeout --signal=TERM --kill-after=20s 900s python3 -u - <<'PY'
import base64,gzip,hashlib,json,os,pathlib,subprocess,urllib.request
root=pathlib.Path('/mnt/afs/USER_ID/SuperMimic/interactive-g1-20261027-a')
base=pathlib.Path('/mnt/afs/USER_ID/SuperMimic/lafan1-20261024')
assert root.resolve().is_relative_to(pathlib.Path('/mnt/afs/USER_ID'))
assert not (root/'evidence/acro-stage-ready.json').exists()
for name,sha in EXPECTED.items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha,name
py=base/'uni/.venv/bin/python';env=os.environ.copy()
full=base/'experiments/continuation-20261025-a/policy/full_016000.pt'
with full.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()=='0afcca9af97bd9db1f2dd083875898f962acab0b0a70c7313b221e575b70b05b'
env.update(SUPERMIMIC_UNI=str(base/'uni'),SUPERMIMIC_FULL=str(full),HF_HUB_OFFLINE='1',WANDB_MODE='offline',MUJOCO_GL='disable',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONPATH=str(root/'scripts')+':'+str(base/'uni/src'))
data=gzip.decompress(base64.b64decode('PAYLOAD'));assert hashlib.sha256(data).hexdigest()=='TRAIN_SHA'
p=root/'scripts/train_acro.py';assert not p.exists() or p.read_bytes()==data;p.write_bytes(data)
source=root/'assets/acro-candidates/88_08_stageii_unitree_g1.csv';source.parent.mkdir(parents=True,exist_ok=True)
if not source.exists():
    request=urllib.request.Request('https://api.github.com/repos/lez666/chunwan-flip/git/blobs/8e477657ff4fdc61ef4fc44466ceeb3f0052eb07',headers={'User-Agent':'SuperMimic-reproduction','Accept':'application/vnd.github+json'})
    with urllib.request.urlopen(request,timeout=45) as response:blob=json.load(response)
    raw=base64.b64decode(blob['content']);assert len(raw)==131459 and hashlib.sha256(raw).hexdigest()=='35d1e39cfdb18e8f644752add1d24ce1e4616b0eb735ecd256bb6a647113884d'
    source.write_bytes(raw)
assert hashlib.sha256(source.read_bytes()).hexdigest()=='35d1e39cfdb18e8f644752add1d24ce1e4616b0eb735ecd256bb6a647113884d'
subprocess.run([str(py),'-u',str(p),'--device','cpu','--label','acro-cloud-cpu-engineering','--steps','6','--world-model-warmup','2','--num-envs','4','--engineering'],env=env,check=True,timeout=120)
proof=json.loads((root/'runs/acro-cloud-cpu-engineering/proof.json').read_text());assert proof['status']=='updates_and_save_verified' and proof['normalizer_tensors_unchanged']
result={'status':'acro_cpu_preparation_passed','gpu_count':0,'reused_environment':str(py),'BONES_downloaded':False,'acro_reference_bytes':source.stat().st_size,'cpu_engineering':proof,'old_environment_modified':False,'new_dependencies_installed':False}
(root/'evidence/acro-stage-ready.json').write_text(json.dumps(result,indent=2)+'\\n');print('ACRO_CPU_STAGE_PASS '+json.dumps(result),flush=True)
PY
""".replace('EXPECTED',repr(expected)).replace('PAYLOAD',payload).replace('TRAIN_SHA',hashlib.sha256(code.encode()).hexdigest())
    p=ROOT/'acro-cpu-stage-20261027c.command.txt';p.write_text(script,newline='\n')
    receipt={'status':'prepared_not_submitted','bytes':len(script.encode()),'sha256':hashlib.sha256(script.encode()).hexdigest(),'name':p.name,'gpu_count':0,'maximum_seconds':900}
    (ROOT/'evidence/acro-cloud-stage-command.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))


if __name__=='__main__':main()
