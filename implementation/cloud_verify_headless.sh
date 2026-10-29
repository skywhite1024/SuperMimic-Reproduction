#!/usr/bin/env bash
set -euo pipefail
umask 077
ROOT=/mnt/afs/USER_ID/SuperMimic/lafan1-20261024
cd "$ROOT/uni"
export WANDB_MODE=offline
export MUJOCO_GL=disable
unset PYOPENGL_PLATFORM
.venv/bin/python -u - <<'PY'
import importlib.metadata,json,os,pathlib,platform,runpy
import torch,mujoco,unilab,warp
root=pathlib.Path('/mnt/afs/USER_ID/SuperMimic/lafan1-20261024')
assert torch.__version__.split('+')[0]=='2.14.1'
assert mujoco.__version__=='3.11.0'
assert importlib.metadata.version('unilab')=='1.5.1'
assert importlib.metadata.version('warp-lang')=='1.16.0'
assert torch.cuda.device_count()==0
asset=runpy.run_path(str(root/'uni/tools/prepare_supermimic_assets.py'))
for function in ('validate_motions','validate_robot','validate_scene'): asset[function]()
d={'status':'environment_and_assets_ready_without_graphics','source_commit':'faf7818e8abac3ec6abb8f00e6e904d3577f800b',
   'python':platform.python_version(),'torch':torch.__version__,'torch_cuda':torch.version.cuda,
   'mujoco':mujoco.__version__,'unilab':importlib.metadata.version('unilab'),'warp':importlib.metadata.version('warp-lang'),
   'packages':{p.metadata['Name']:p.version for p in importlib.metadata.distributions()},'gpu_training_executed':False,
   'graphics_mode_for_simulation':'disable'}
(root/'evidence/environment.json').write_text(json.dumps(d,indent=2)+chr(10))
print('SUPERMIMIC_CPU_CORE_PASS '+json.dumps({k:v for k,v in d.items() if k!='packages'}),flush=True)
PY
export DEBIAN_FRONTEND=noninteractive
timeout --signal=TERM --kill-after=10s 300s bash -c 'set -euo pipefail; apt-get -o Acquire::Retries=1 -o Acquire::http::Timeout=30 update; apt-get -o Acquire::Retries=1 -o Acquire::http::Timeout=30 install -y --no-install-recommends libosmesa6'
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
.venv/bin/python -u - <<'PY'
import hashlib,json,pathlib,subprocess
import imageio.v2 as imageio
import mujoco,numpy as np
from PIL import Image
root=pathlib.Path('/mnt/afs/USER_ID/SuperMimic/lafan1-20261024')
m=mujoco.MjModel.from_xml_path(str(root/'uni/src/unisupermimic/assets/robots/g1/scene_supermimic.xml'))
d=mujoco.MjData(m)
key=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_KEY,'stand')
mujoco.mj_resetDataKeyframe(m,d,key)
mujoco.mj_forward(m,d)
renderer=mujoco.Renderer(m,height=240,width=320)
try:
 renderer.update_scene(d)
 frame=renderer.render().copy()
 assert frame.shape==(240,320,3) and float(frame.std())>1.0
 Image.fromarray(frame).save(root/'evidence/cpu-render-smoke.png')
 with imageio.get_writer(root/'evidence/cpu-render-smoke.mp4',fps=2,codec='libx264',macro_block_size=None) as writer:
  writer.append_data(frame)
  writer.append_data(frame)
finally: renderer.close()
version=subprocess.check_output(['dpkg-query','-W','-f=${Version}','libosmesa6'],text=True).strip()
receipt={'status':'passed','renderer':'OSMesa software','libosmesa6':version,'shape':list(frame.shape),'pixel_std':float(frame.std()),
         'png_sha256':hashlib.sha256((root/'evidence/cpu-render-smoke.png').read_bytes()).hexdigest()}
(root/'evidence/headless-render.json').write_text(json.dumps(receipt,indent=2)+chr(10))
print('SUPERMIMIC_HEADLESS_RENDER_PASS '+json.dumps(receipt),flush=True)
print('SUPERMIMIC_CPU_PREP_PASS',flush=True)
PY
