"""Make an auditable one-off ACP command without credentials or permanent services."""
from __future__ import annotations
import base64
import hashlib
import json
from pathlib import Path
import zlib

ROOT = Path(__file__).resolve().parent
DEST = "/mnt/afs/USER_ID/SuperMimic/lafan1-20261024"
files = {name: (ROOT / name).read_text(encoding="utf-8") for name in ("prepare_assets.py", "cloud_prepare.sh")}
assets=json.loads((ROOT / "evidence/asset-downloads.json").read_text())
asset_digest=hashlib.sha256(json.dumps({x['remote']:(x['bytes'],x['sha256']) for x in assets['files']},sort_keys=True).encode()).hexdigest()
files["cloud_prepare.sh"]=files["cloud_prepare.sh"].replace('__EXPECTED_RAW_ASSET_DIGEST__',asset_digest)
payload = json.dumps(files, separators=(",", ":")).encode()
blob = base64.b64encode(zlib.compress(payload, 9)).decode()
sha = hashlib.sha256(payload).hexdigest()
bootstrap = f'''import base64,hashlib,json,pathlib,zlib
p=zlib.decompress(base64.b64decode({blob!r}))
assert hashlib.sha256(p).hexdigest()=={sha!r}
r=pathlib.Path({DEST!r})
assert r.resolve().is_relative_to(pathlib.Path('/mnt/afs/USER_ID'))
for n,t in json.loads(p).items():
 d=r/n
 assert d.resolve().is_relative_to(r.resolve())
 d.parent.mkdir(parents=True,exist_ok=True)
 if d.exists() and d.read_text()!=t: raise RuntimeError('Refuse to overwrite changed file: '+n)
 d.write_text(t)
print('SUPERMIMIC_PAYLOAD_VERIFIED',flush=True)
'''
command = "set -e\npython3 - <<'PY'\n" + bootstrap + "PY\n"
command += f"timeout --signal=TERM --kill-after=30s 2700s bash {DEST}/cloud_prepare.sh\n"
(ROOT / "cpu-prepare.command.txt").write_text(command, encoding="utf-8", newline="\n")
(ROOT / "evidence/cpu-command.json").write_text(json.dumps({"payload_sha256":sha,
 "command_sha256":hashlib.sha256(command.encode()).hexdigest(),"characters":len(command),
 "root":DEST,"gpus":0,"vcpus":4,"memory_gib":16,"replicas":1,
 "maximum_seconds":2730,"max_restarts":0,"node_ssh":False},indent=2)+"\n")
print(json.dumps({"characters":len(command),"command_sha256":hashlib.sha256(command.encode()).hexdigest()}))
