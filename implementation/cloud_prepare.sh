#!/usr/bin/env bash
set -euo pipefail
umask 077
ROOT=/mnt/afs/USER_ID/SuperMimic/lafan1-20261024
test -d /mnt/afs/USER_ID
test -d "$ROOT"
cd "$ROOT"
mkdir -p evidence cache
export UV_CACHE_DIR="$ROOT/cache/uv"
export HF_HOME="$ROOT/cache/huggingface"
export UV_LINK_MODE=copy
export WANDB_MODE=offline
export PIP_DISABLE_PIP_VERSION_CHECK=1
if [ ! -x "$ROOT/bootstrap/bin/uv" ]; then
  python3 -m venv "$ROOT/bootstrap"
  "$ROOT/bootstrap/bin/python" -m pip install uv==0.12.2
fi
if [ ! -d uni/.git ]; then
  git clone https://github.com/Motphys/Uni-SuperMimic.git uni
fi
git -C uni checkout --detach faf7818e8abac3ec6abb8f00e6e904d3577f800b
test "$(git -C uni rev-parse HEAD)" = faf7818e8abac3ec6abb8f00e6e904d3577f800b
cd uni
"$ROOT/bootstrap/bin/uv" sync --frozen --no-dev --python python3
.venv/bin/python -u "$ROOT/prepare_assets.py"
.venv/bin/python -u tools/prepare_supermimic_assets.py --validate-only
.venv/bin/python - <<'PY'
import hashlib, importlib.metadata, json, pathlib, platform, subprocess
root=pathlib.Path('/mnt/afs/USER_ID/SuperMimic/lafan1-20261024')
actual=json.loads((root/'evidence/asset-downloads.json').read_text())
observed={x['remote']:(x['bytes'],x['sha256']) for x in actual['files']}
assert hashlib.sha256(json.dumps(observed,sort_keys=True).encode()).hexdigest()=='fdfcda4c9ad74aa39ce0be0740c10b4cd511312bfb6ec8f052cc08067df70883', 'Downloaded assets differ from local verified snapshot'
import torch, mujoco, unilab, warp
assert torch.__version__.split('+')[0]=='2.14.1'
assert mujoco.__version__=='3.11.0'
assert importlib.metadata.version('unilab')=='1.5.1'
record={'status':'environment_and_assets_ready','python':platform.python_version(),
 'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
 'torch':torch.__version__, 'torch_cuda':torch.version.cuda,
 'unilab':importlib.metadata.version('unilab'), 'mujoco':mujoco.__version__,
 'warp':importlib.metadata.version('warp-lang'),
 'packages':{p.metadata['Name']:p.version for p in importlib.metadata.distributions()},
 'assets_verified':len(observed),'gpu_training_executed':False}
for n in ('cpu.max','memory.max'):
 p=pathlib.Path('/sys/fs/cgroup')/n
 record[n]=p.read_text().strip() if p.exists() else None
(root/'evidence/environment.json').write_text(json.dumps(record,indent=2)+'\n')
print('SUPERMIMIC_CPU_PREP_PASS '+json.dumps({k:v for k,v in record.items() if k!='packages'}),flush=True)
PY
