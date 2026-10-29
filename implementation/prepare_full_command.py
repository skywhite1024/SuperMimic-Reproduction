"""Build the finite formal run command after actual smoke acceptance."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
base = (ROOT / "run_gpu_v2.py").read_text()
assert hashlib.sha256((ROOT / "run_gpu_v2.py").read_bytes()).hexdigest() == "03cd56f2e3ec810a8682094089c8f4d7ba8478bbcb413e58e1b14fa22bdd4bae"
old = '''                    print("SUPERMIMIC_PROGRESS "+json.dumps({"stage":name,"seconds":time.monotonic()-start,
                          "resources":row,"log_tail":log_tail(folder/"stdout.txt")[-1500:]}),flush=True)'''
new = '''                    fields=row["nvidia_smi"].split(",")
                    print("SUPERMIMIC_PROGRESS "+json.dumps({"stage":name,"seconds":round(time.monotonic()-start,1),
                          "host_bytes":row["cgroup_memory_current"],"gpu":row["nvidia_smi"]}),flush=True)
                    for line in log_tail(folder/"stdout.txt")[-1800:].splitlines():
                        for index in range(0,max(1,len(line)),350):
                            print("SUPERMIMIC_CHILD_TAIL "+line[index:index+350],flush=True)'''
assert base.count(old) == 1
runtime = base.replace(old, new)
(ROOT / "run_gpu_v3.py").write_text(runtime, encoding="utf-8", newline="\n")
runtime_hash = hashlib.sha256((ROOT / "run_gpu_v3.py").read_bytes()).hexdigest()
command = f'''set -e
cd /mnt/afs/USER_ID/SuperMimic/lafan1-20261024
export MUJOCO_GL=disable WANDB_MODE=offline HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4 MAX_JOBS=4
unset PYOPENGL_PLATFORM
python3 - <<'PY'
import hashlib,json,pathlib
p=pathlib.Path('run_gpu_v2.py')
assert hashlib.sha256(p.read_bytes()).hexdigest()=='03cd56f2e3ec810a8682094089c8f4d7ba8478bbcb413e58e1b14fa22bdd4bae'
assert json.loads(pathlib.Path('evidence/acceptance.json').read_text())['passed']
s=p.read_text()
old={old!r}
new={new!r}
assert s.count(old)==1
s=s.replace(old,new)
p=pathlib.Path('run_gpu_v3.py')
assert not p.exists() or p.read_text()==s
p.write_text(s)
assert hashlib.sha256(p.read_bytes()).hexdigest()=='{runtime_hash}'
print('SUPERMIMIC_FORMAL_RUNTIME_VERIFIED',flush=True)
PY
export DEBIAN_FRONTEND=noninteractive
timeout --signal=TERM --kill-after=10s 300s bash -c 'set -euo pipefail; apt-get -o Acquire::Retries=1 -o Acquire::http::Timeout=30 update; apt-get -o Acquire::Retries=1 -o Acquire::http::Timeout=30 install -y --no-install-recommends libosmesa6=25.1.7-1ubuntu2~24.04.2'
cd uni
timeout --signal=TERM --kill-after=30s 19800s .venv/bin/python -u ../run_gpu_v3.py --mode full
'''
(ROOT / "gpu-full.command.txt").write_text(command, encoding="utf-8", newline="\n")
receipt = {"runtime_sha256":runtime_hash,"command_sha256":hashlib.sha256(command.encode()).hexdigest(),
           "changes_from_v2":["short progress log lines to avoid CMS 500-character truncation"],
           "network_mode":"HF_HUB_OFFLINE=1; prevalidated local assets", "gpus":1,"vcpus":4,"memory_gib":64,
           "replicas":1,"max_restarts":0,"max_program_seconds":20140,"formal_iterations":12000,
           "upstream_algorithm_modified":False,"status":"prepared; not submitted"}
(ROOT / "evidence/gpu-full-command.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt))
