#!/usr/bin/env bash
set -euo pipefail
umask 077
ROOT=/mnt/afs/USER_ID/SuperMimic/lafan1-20261024
cd "$ROOT"
export UV_CACHE_DIR="$ROOT/cache/uv"
export UV_LINK_MODE=hardlink
export UV_COMPILE_BYTECODE=0
export UV_HTTP_TIMEOUT=60
export UV_HTTP_RETRIES=1
export WANDB_MODE=offline
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
python3 -u - <<'PY'
import importlib.metadata,json,os,pathlib,uuid
root=pathlib.Path('/mnt/afs/USER_ID/SuperMimic/lafan1-20261024')
name='.hardlink-probe-'+uuid.uuid4().hex
source=root/name
target=root/(name+'.link')
try:
 source.write_bytes(b'supermimic-private-afs-hardlink-probe')
 os.link(source,target)
 assert source.stat().st_ino==target.stat().st_ino
 print('SUPERMIMIC_AFS_HARDLINK_PASS',flush=True)
finally:
 target.unlink(missing_ok=True)
 source.unlink(missing_ok=True)
site=root/'uni/.venv/lib/python3.12/site-packages'
installed={p.metadata['Name']:p.version for p in importlib.metadata.distributions(path=[str(site)])}
print('SUPERMIMIC_PARTIAL_ENV '+json.dumps({'installed_count':len(installed),'packages':installed}),flush=True)
PY
UV="$ROOT/bin/uv"
echo SUPERMIMIC_OFFLINE_HARDLINK_SYNC_BEGIN
"$UV" pip sync --offline --python uni/.venv/bin/python --require-hashes --link-mode hardlink --index-url https://mirrors.aliyun.com/pypi/simple/ requirements-upstream.lock.txt
echo SUPERMIMIC_OFFLINE_HARDLINK_SYNC_END
"$UV" pip install --python uni/.venv/bin/python --index-url https://mirrors.aliyun.com/pypi/simple/ uv_build==0.12.2
cd uni
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
   'packages':{p.metadata['Name']:p.version for p in importlib.metadata.distributions()},'gpu_training_executed':False,
   'install_method':'hash-checked cached packages linked within personal AFS'}
(root/'evidence/environment.json').write_text(json.dumps(d,indent=2)+chr(10))
print('SUPERMIMIC_CPU_PREP_PASS '+json.dumps({k:v for k,v in d.items() if k!='packages'}),flush=True)
PY
