#!/usr/bin/env bash
set -euo pipefail
umask 077
ROOT=/mnt/afs/USER_ID/SuperMimic/lafan1-20261024
cd "$ROOT"
export UV_CACHE_DIR="$ROOT/cache/uv"
export UV_LINK_MODE=copy
export UV_HTTP_TIMEOUT=60
export UV_HTTP_RETRIES=1
export UV_CONCURRENT_DOWNLOADS=6
export WANDB_MODE=offline
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
python3 -u - <<'PY'
import hashlib,json,pathlib,tarfile
root=pathlib.Path('/mnt/afs/USER_ID/SuperMimic/lafan1-20261024').resolve()
archive=root/'supermimic-lafan1-stage.tar.gz'
expected='1881b940c9e98790248c01fbc4f4618320d01079c343df95bfbb474b8bcf7883'
assert hashlib.sha256(archive.read_bytes()).hexdigest()==expected
with tarfile.open(archive) as tar:
 for item in tar.getmembers():
  target=root/item.name
  assert target.resolve().is_relative_to(root) and item.isfile(),item.name
  if target.exists():
   assert hashlib.sha256(target.read_bytes()).digest()==hashlib.sha256(tar.extractfile(item).read()).digest(),item.name
 tar.extractall(root,filter='data')
for manifest,prefix in [('evidence/source-files.json','uni')]:
 for name,sha in json.loads((root/manifest).read_text())['files'].items():
  assert hashlib.sha256((root/prefix/name).read_bytes()).hexdigest()==sha,name
for row in json.loads((root/'evidence/asset-downloads.json').read_text())['files']:
 assert hashlib.sha256((root/'uni'/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
print('SUPERMIMIC_STAGED_SOURCE_AND_ASSETS_PASS',flush=True)
PY
chmod 700 "$ROOT/bin/uv"
UV="$ROOT/bin/uv"
"$UV" --version
cd uni
if [ ! -x .venv/bin/python ]; then "$UV" venv --python python3 .venv; fi
"$UV" pip sync --python .venv/bin/python --require-hashes --index-url https://mirrors.aliyun.com/pypi/simple/ "$ROOT/requirements-upstream.lock.txt"
"$UV" pip install --python .venv/bin/python --index-url https://mirrors.aliyun.com/pypi/simple/ uv_build==0.12.2
"$UV" pip install --python .venv/bin/python --no-deps --no-build-isolation --editable .
.venv/bin/python -u tools/prepare_supermimic_assets.py --validate-only
.venv/bin/python -u - <<'PY'
import importlib.metadata,json,pathlib,platform
import torch,mujoco,unilab,warp
root=pathlib.Path('/mnt/afs/USER_ID/SuperMimic/lafan1-20261024')
assert torch.__version__.split('+')[0]=='2.14.1'
assert mujoco.__version__=='3.11.0'
assert importlib.metadata.version('unilab')=='1.5.1'
d={'status':'environment_and_assets_ready','source_commit':'faf7818e8abac3ec6abb8f00e6e904d3577f800b',
   'python':platform.python_version(),'torch':torch.__version__,'torch_cuda':torch.version.cuda,
   'mujoco':mujoco.__version__,'unilab':importlib.metadata.version('unilab'),'warp':importlib.metadata.version('warp-lang'),
   'packages':{p.metadata['Name']:p.version for p in importlib.metadata.distributions()},'gpu_training_executed':False}
(root/'evidence/environment.json').write_text(json.dumps(d,indent=2)+'\n')
print('SUPERMIMIC_CPU_PREP_PASS '+json.dumps({k:v for k,v in d.items() if k!='packages'}),flush=True)
PY
