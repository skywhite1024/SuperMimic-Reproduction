"""Build hash-checked commands without editing the fixed baseline directory."""
from pathlib import Path
import base64
import gzip
import hashlib
import json

OUT = Path(__file__).resolve().parent
LOCAL = OUT.parent
BASE = '/mnt/afs/USER_ID/SuperMimic/lafan1-20261024'
REMOTE = BASE + '/experiments/continuation-20261025-a'
base_files = [
    'evidence/source-files.json', 'evidence/asset-downloads.json', 'evidence/environment.json',
    'runs/baseline-seed0/last.pt', 'runs/baseline-seed0/ckpt_012000.pt',
    'runs/baseline-seed0/args.json', 'runs/baseline-seed0/run_config.json',
    'runs/baseline-evaluation.json',
    'evaluate_audited_v2.py', 'train_audited_v2.py', 'run_gpu_v3.py',
]


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    frozen = {'source_commit': 'faf7818e8abac3ec6abb8f00e6e904d3577f800b',
        'dataset_revision': 'ce1572906efe6157840e8474d5a0d7aa87481e74',
        'base_files': {name: sha((LOCAL/'downloaded'/name) if (LOCAL/'downloaded'/name).is_file() else LOCAL/name) for name in base_files},
        'only_training_change': 'extend total iterations to 24000 from full 12000 checkpoint',
        'seed': 0, 'num_envs': 4096, 'horizon': 32, 'amp': 'bf16', 'compile': True,
        'lr': 0.0002, 'lr_decay_iters': 250000, 'evaluation_iterations': [16000, 20000, 24000],
        'community_acceptance_target': {'macro_at_least': 0.82, 'n_failed': 0},
        'target_is_not_attributed_to_paper_failure_count': True,
        'gpu_budget_seconds_first_program': 10740, 'gpu_hours_total_cap_new_experiments': 6}
    assert frozen['base_files']['runs/baseline-seed0/last.pt'] == '8770016fac48e4938203f77c8e2cd55b553171d2bc3134b7cbb40b2c53f6b1f1'
    (OUT/'frozen.json').write_text(json.dumps(frozen, indent=2)+'\n', encoding='utf-8', newline='\n')
    names = ['audit_common.py', 'preflight.py', 'train_continuation.py', 'run_experiment.py', 'diagnose.py', 'collect_continuation.py', 'frozen.json']
    payload = {name: (OUT/name).read_text(encoding='utf-8') for name in names}
    hashes = {name: sha(OUT/name) for name in names}
    encoded = base64.b64encode(gzip.compress(json.dumps(payload).encode(), mtime=0)).decode()
    bootstrap = f'''import base64,gzip,hashlib,json,pathlib
root=pathlib.Path({REMOTE!r})
root.mkdir(parents=True,exist_ok=True)
assert not (root/'experiment.json').exists(),'Existing experiment requires explicit inspection;refusing duplicate'
payload=json.loads(gzip.decompress(base64.b64decode({encoded!r})))
expected={hashes!r}
for name,body in payload.items():
    data=body.encode()
    assert hashlib.sha256(data).hexdigest()==expected[name],name
    p=root/name
    assert not p.exists() or p.read_bytes()==data,name
    p.write_bytes(data)
print('CONTINUATION_DEPLOY_VERIFIED',flush=True)
'''
    env = f'''set -e
export SUPERMIMIC_BASE={BASE}
export MUJOCO_GL=disable WANDB_MODE=offline HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4 MAX_JOBS=4
unset PYOPENGL_PLATFORM
cd {BASE}/uni
'''
    cpu = env + "python3 - <<'PY'\n" + bootstrap + "PY\n" + f'timeout --signal=TERM --kill-after=10s 180s .venv/bin/python -u {REMOTE}/preflight.py\n'
    gpucheck = f'''import hashlib,json,pathlib
root=pathlib.Path({REMOTE!r})
expected={hashes!r}
for name,digest in expected.items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
assert json.loads((root/'evidence/preflight.json').read_text())['passed']
assert not (root/'experiment.json').exists(),'Refusing duplicate GPU run'
print('CONTINUATION_PAYLOAD_AND_CPU_PREFLIGHT_VERIFIED',flush=True)
'''
    gpu = env + "python3 - <<'PY'\n" + gpucheck + "PY\n" + f'''export DEBIAN_FRONTEND=noninteractive
timeout --signal=TERM --kill-after=10s 300s bash -c 'set -euo pipefail; apt-get -o Acquire::Retries=1 -o Acquire::http::Timeout=30 update; apt-get -o Acquire::Retries=1 -o Acquire::http::Timeout=30 install -y --no-install-recommends libosmesa6=25.1.7-1ubuntu2~24.04.2'
timeout --signal=TERM --kill-after=30s 10380s .venv/bin/python -u {REMOTE}/run_experiment.py
'''
    for name,body in [('cpu-preflight.command.txt',cpu),('gpu-continuation.command.txt',gpu)]:
        (OUT/name).write_text(body, encoding='utf-8', newline='\n')
    receipt = {'status': 'prepared_not_submitted', 'cloud_root': REMOTE, 'files': hashes,
               'cpu_command_sha256': sha(OUT/'cpu-preflight.command.txt'),
               'gpu_command_sha256': sha(OUT/'gpu-continuation.command.txt'),
               'gpu_config': {'gpus': 1, 'cpus': 4, 'memory_gib': 64, 'replicas': 1, 'max_restarts': 0, 'node_ssh': False},
               'finite_cpu_seconds': 190, 'finite_gpu_program_upper_seconds': 10740}
    (OUT/'preparation.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({k: receipt[k] for k in ['status','cloud_root','cpu_command_sha256','gpu_command_sha256']}))


if __name__ == '__main__': main()
