"""Freeze a small source payload for one finite ACP engineering job."""
import base64
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import argparse

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=Path('C:/project/sensecore/SuperMimic/research/sports-video-execution-20261110/cloud')
OUT.mkdir(parents=True,exist_ok=True)
parser=argparse.ArgumentParser()
parser.add_argument('--attempt',choices=['a','b','c'],default='a')
attempt=parser.parse_args().attempt
tracked=subprocess.check_output(['git','ls-files','implementation/uni','acro-library-20261027-b/scripts/balanced_replay.py'],cwd=ROOT,text=True).splitlines()
lock={name:hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT)).hexdigest() for name in tracked}
(HERE/'runtime-source-lock.json').write_text(json.dumps(lock,indent=2)+'\n')
names=['cloud_engineering.py','sports_runtime.py','train.py','runtime-source-lock.json']
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w') as tar:
    for name in names:
        data=(HERE/name).read_bytes()
        info=tarfile.TarInfo('experiments/sports-video/'+name)
        info.size=len(data); info.mtime=0; info.mode=0o600
        tar.addfile(info,io.BytesIO(data))
raw=gzip.compress(buf.getvalue(),mtime=0)
payload=base64.b64encode(raw).decode()
digest=hashlib.sha256(raw).hexdigest()
label='cuda-engineering-1110'+attempt
command="""set -euo pipefail
umask 077
timeout --signal=TERM --kill-after=30s 1800s bash -s <<'SPORTS_ENGINEERING'
set -euo pipefail
trial=/mnt/afs/260010076/SuperMimic/sports-video-20261110/engineering-ATTEMPT
mkdir -p /mnt/afs/260010076/SuperMimic/sports-video-20261110
mkdir "$trial"
export SPORTS_TRIAL="$trial"
python3 - <<'UNPACK'
import base64,gzip,hashlib,io,os,pathlib,tarfile
raw=base64.b64decode('PAYLOAD')
assert hashlib.sha256(raw).hexdigest()=='DIGEST'
root=pathlib.Path(os.environ['SPORTS_TRIAL'])/'source'
root.mkdir()
with tarfile.open(fileobj=io.BytesIO(gzip.decompress(raw))) as tar:
    for member in tar.getmembers():
        assert member.isfile() and (root/member.name).resolve().is_relative_to(root)
    tar.extractall(root,filter='data')
UNPACK
export MUJOCO_GL=disable WANDB_MODE=offline HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "$trial/source"
exec /mnt/afs/260010076/SuperMimic/lafan1-20261024/uni/.venv/bin/python -u experiments/sports-video/cloud_engineering.py --label LABEL
SPORTS_ENGINEERING
""".replace('PAYLOAD',payload).replace('DIGEST',digest).replace('LABEL',label).replace('ATTEMPT',attempt)
(OUT/f'engineering-{attempt}.command.sh').write_text(command,encoding='utf-8',newline='\n')
(OUT/f'engineering-{attempt}-payload.tar.gz').write_bytes(raw)
(OUT/f'engineering-{attempt}-source.json').write_text(json.dumps({'payload_sha256':digest,'command_sha256':hashlib.sha256(command.encode()).hexdigest(),'files':{n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in names},'max_wall_seconds':1800,'GPUs':1,'sports_result':False},indent=2)+'\n')
print(json.dumps({'command_chars':len(command),'payload_bytes':len(raw),'locked_files':len(lock),'sha256':digest}))
