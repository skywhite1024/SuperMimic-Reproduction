"""Build an allowlisted Linux staging archive. Never includes credentials or venvs."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"staging"
OUT.mkdir(exist_ok=True)
with urllib.request.urlopen("https://pypi.org/pypi/uv/0.12.2/json",timeout=30) as response:
    metadata=json.load(response)
wheel=next(x for x in metadata["urls"] if x["filename"]=="uv-0.12.2-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl")
wheel_path=OUT/wheel["filename"]
if not wheel_path.exists():
    with urllib.request.urlopen(wheel["url"],timeout=60) as response, wheel_path.open("wb") as stream:
        while chunk:=response.read(1024*1024): stream.write(chunk)
assert hashlib.sha256(wheel_path.read_bytes()).hexdigest()==wheel["digests"]["sha256"]
with zipfile.ZipFile(wheel_path) as archive:
    name=next(n for n in archive.namelist() if n.endswith("/uv") and not n.endswith(".dist-info/uv"))
    (OUT/"uv").write_bytes(archive.read(name))
print("LINUX_UV_VERIFIED",flush=True)
source=ROOT/"uni"
tracked=subprocess.check_output(["git","-C",str(source),"ls-files"],text=True).splitlines()
included=[n for n in tracked if n.startswith(("src/","tools/","tests/","LICENSES/")) or n in ("pyproject.toml","uv.lock","README.md","LICENSE","NOTICE")]
source_hashes={n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in included}
(ROOT/"evidence/source-files.json").write_text(json.dumps({"commit":"faf7818e8abac3ec6abb8f00e6e904d3577f800b","files":source_hashes},indent=2)+"\n")
assets=json.loads((ROOT/"evidence/asset-downloads.json").read_text())
paths={"uni/"+n:source/n for n in included}
paths.update({"uni/"+x["path"]:source/x["path"] for x in assets["files"]})
for name in ("requirements-upstream.lock.txt","train_audited.py","evaluate_audited.py","run_gpu.py","evidence/asset-downloads.json","evidence/source-files.json"):
    paths[name]=ROOT/name
paths["bin/uv"]=OUT/"uv"
archive_path=OUT/"supermimic-lafan1-stage.tar.gz"
with tarfile.open(archive_path,"w:gz",compresslevel=3) as archive:
    for name,path in sorted(paths.items()):
        assert path.is_file() and not path.is_symlink()
        archive.add(path,arcname=name,recursive=False)
receipt={"archive":str(archive_path),"bytes":archive_path.stat().st_size,
         "sha256":hashlib.sha256(archive_path.read_bytes()).hexdigest(),"files":len(paths),
         "uv_wheel":wheel["filename"],"uv_wheel_sha256":wheel["digests"]["sha256"],
         "contains_credentials":False,"source_commit":"faf7818e8abac3ec6abb8f00e6e904d3577f800b"}
(ROOT/"evidence/staging.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt),flush=True)
