"""Fetch an immutable, anonymous G1/LAFAN1 snapshot and validate it with upstream tools.

Only the pure robot constants are preloaded to avoid installing a GPU training
stack on the Windows machine used for asset preparation. Upstream generation
and validation functions run unchanged under MuJoCo 3.11.0.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import runpy
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "uni"
EVIDENCE = ROOT / "evidence"
DATASET = "lvhaidong/LAFAN1_Retargeting_Dataset"
REVISION = "ce1572906efe6157840e8474d5a0d7aa87481e74"
G1 = REPO / "src/unisupermimic/assets/robots/g1"
MOTIONS = REPO / "src/unisupermimic/assets/motions/g1/lafan1_supermimic"


def fetch(remote: str, target: Path) -> dict:
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{remote}"
    target.parent.mkdir(parents=True, exist_ok=True)
    # Validate reused files against the previous receipt, rather than just size.
    prior = EVIDENCE / "asset-downloads.json"
    known = {x["remote"]: x for x in json.loads(prior.read_text())["files"]} if prior.exists() else {}
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    if not (target.exists() and remote in known and digest(target) == known[remote]["sha256"]):
        temporary = target.with_name(target.name + ".download")
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=90) as response, temporary.open("wb") as out:
                    while chunk := response.read(1024 * 1024):
                        out.write(chunk)
                os.replace(temporary, target)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2)
    record = {"remote": remote, "path": target.relative_to(REPO).as_posix(),
              "bytes": target.stat().st_size, "sha256": digest(target)}
    print(json.dumps({"asset": remote, "bytes": record["bytes"]}), flush=True)
    return record


def main() -> None:
    EVIDENCE.mkdir(exist_ok=True)
    url = f"https://huggingface.co/api/datasets/{DATASET}/revision/{REVISION}"
    with urllib.request.urlopen(url, timeout=60) as response:
        info = json.load(response)
    assert info["sha"] == REVISION
    files = {x["rfilename"] for x in info["siblings"]}
    clips = sorted(x for x in files if re.fullmatch(r"g1/[^/]+\.csv", x))
    assert len(clips) == 40, len(clips)
    urdf = G1 / "assets/supermimic/g1_29dof_rev_1_0.urdf"
    records = [fetch("robot_description/g1/g1_29dof_rev_1_0.urdf", urdf)]
    meshes = sorted({PurePosixPath(x.attrib["filename"]).name for x in ET.parse(urdf).iter("mesh")})
    requests = [(x, MOTIONS / PurePosixPath(x).name) for x in clips]
    requests += [(f"robot_description/g1/meshes/{x}", G1 / "assets" / x) for x in meshes]
    for remote, _ in requests:
        assert remote in files, remote
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        records += list(pool.map(lambda pair: fetch(*pair), requests))
    receipt = {"dataset": DATASET, "revision": REVISION, "files": sorted(records, key=lambda x: x["remote"])}
    (EVIDENCE / "asset-downloads.json").write_text(json.dumps(receipt, indent=2) + "\n")
    robot_name = "unisupermimic.tasks.motion_tracking.g1.supermimic.robot"
    spec = importlib.util.spec_from_file_location(robot_name, REPO / "src/unisupermimic/tasks/motion_tracking/g1/supermimic/robot.py")
    robot = importlib.util.module_from_spec(spec)
    sys.modules[robot_name] = robot
    spec.loader.exec_module(robot)
    prepare = runpy.run_path(str(REPO / "tools/prepare_supermimic_assets.py"))
    prepare["generate_robot_xml"]()
    prepare["validate_robot"]()
    prepare["validate_scene"]()
    prepare["validate_motions"]()
    print("PINNED_ASSETS_VALIDATED", flush=True)


if __name__ == "__main__":
    main()
