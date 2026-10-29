"""Prepare one finite H100 training job only after the CPU preparation receipt is verified."""
import base64,gzip,hashlib,json
from runtime import ROOT


def main():
    code=(ROOT/'scripts/evaluate_acro.py').read_text();digest=hashlib.sha256(code.encode()).hexdigest()
    body=base64.b64encode(gzip.compress(code.encode(),mtime=0)).decode()
    text="""set -euo pipefail
umask 077
timeout --signal=TERM --kill-after=30s 7200s python3 -u - <<'PY'
import base64,gzip,hashlib,json,os,pathlib,subprocess
root=pathlib.Path('/mnt/afs/USER_ID/SuperMimic/interactive-g1-20261027-a')
base=pathlib.Path('/mnt/afs/USER_ID/SuperMimic/lafan1-20261024')
stage=json.loads((root/'evidence/acro-stage-ready.json').read_text())
assert stage['status']=='acro_cpu_preparation_passed' and stage['cpu_engineering']['normalizer_tensors_unchanged']
assert hashlib.sha256((root/'scripts/train_acro.py').read_bytes()).hexdigest()=='1032a810417382605e208221380293986ec5e0b487934365503181f4a9e600c3'
assert not (root/'runs/acro-h100-2000/args.json').exists(),'Inspect existing run before any duplicate GPU submission'
data=gzip.decompress(base64.b64decode('PAYLOAD'));assert hashlib.sha256(data).hexdigest()=='DIGEST'
evaluation=root/'scripts/evaluate_acro.py';assert not evaluation.exists() or evaluation.read_bytes()==data;evaluation.write_bytes(data)
env=os.environ.copy();env.update(SUPERMIMIC_UNI=str(base/'uni'),SUPERMIMIC_FULL=str(base/'experiments/continuation-20261025-a/policy/full_016000.pt'),HF_HUB_OFFLINE='1',WANDB_MODE='offline',MUJOCO_GL='disable',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONPATH=str(root/'scripts')+':'+str(base/'uni/src'))
py=str(base/'uni/.venv/bin/python')
print('ACRO_H100_FINITE_TRAINING_START '+json.dumps({'gpu_count_requested':1,'iterations':2000,'world_model_warmup':400,'environments':512,'maximum_program_seconds':7200,'BONES_downloaded':False,'old_baseline_modified':False}),flush=True)
subprocess.run([py,'-u',str(root/'scripts/train_acro.py'),'--device','cuda','--label','acro-h100-2000','--steps','2000','--world-model-warmup','400','--num-envs','512'],env=env,check=True,timeout=6800)
proof=json.loads((root/'runs/acro-h100-2000/proof.json').read_text());assert proof['status']=='updates_and_save_verified' and proof['device']=='cuda' and proof['normalizer_tensors_unchanged']
print('ACRO_H100_TRAINING_PROOF '+json.dumps(proof),flush=True)
full=root/'runs/acro-h100-2000/final-full.pt'
subprocess.run([py,'-u',str(evaluation),'--mode','acro-h100-2000-retest','--checkpoint',str(full)],env=env,check=True,timeout=120)
result=root/'runs/acro-h100-2000-retest';summary=json.loads((result/'summary.json').read_text())
raw=(result/'trajectory.npz').read_bytes();encoded=base64.b64encode(raw).decode();parts=[encoded[i:i+3800] for i in range(0,len(encoded),3800)]
for i,part in enumerate(parts):print('ACRO_TRAJECTORY_CHUNK '+str(i+1)+'/'+str(len(parts))+' '+part,flush=True)
receipt={'summary':summary,'trajectory_bytes':len(raw),'trajectory_sha256':hashlib.sha256(raw).hexdigest(),'chunks':len(parts),'full_checkpoint_sha256':hashlib.file_digest(full.open('rb'),'sha256').hexdigest()}
(root/'evidence/acro-gpu-result-receipt.json').write_text(json.dumps(receipt,indent=2)+'\\n')
print('ACRO_GPU_RESULT_RECEIPT '+json.dumps(receipt),flush=True)
PY
""".replace('PAYLOAD',body).replace('DIGEST',digest)
    path=ROOT/'acro-h100-20261027d1.command.txt';path.write_text(text,newline='\n')
    receipt={'status':'prepared_not_submitted','name':path.name,'bytes':len(text.encode()),'sha256':hashlib.sha256(text.encode()).hexdigest(),
             'gpu_count':1,'replicas':1,'cpu_count':4,'memory_gib':64,'max_restarts':0,'maximum_seconds':7200,'no_new_training_dependencies':True}
    (ROOT/'evidence/acro-h100-command.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))


if __name__=='__main__':main()
