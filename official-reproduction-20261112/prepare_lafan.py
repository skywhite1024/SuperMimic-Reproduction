"""Convert the preserved raw LAFAN40 files with the author's actual converter.

This prepares a deployment/efficiency dataset, not the full BONES benchmark.
No W&B upload, ground calibration, clipping, or reference repair is performed.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from mjlab.scripts.csv_to_npz import JOINT_NAMES, MotionLoader


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    assert manifest["denominator"] == len(manifest["files"]) == 40
    # Verify the whole source library before writing any converted data.
    for item in manifest["files"]:
        path = args.source / Path(item["path"]).name
        assert path.stat().st_size == item["bytes"], path
        assert sha256(path) == item["sha256"], path
    args.output.mkdir(parents=True, exist_ok=False)
    clips = args.output / "npz"
    clips.mkdir()
    records = []
    for item in manifest["files"]:
        source = args.source / Path(item["path"]).name
        motion = MotionLoader(str(source), input_fps=30.0, output_fps=50.0, device="cpu")
        arrays = motion.arrays()
        assert arrays["joint_names"].tolist() == list(JOINT_NAMES)
        n = len(arrays["joint_pos"])
        for name, array in arrays.items():
            if name in {"joint_names", "fps"}:
                continue
            assert len(array) == n and array.dtype == np.float32
            assert np.isfinite(array).all(), (source, name)
        assert np.allclose(np.linalg.norm(arrays["root_quat_w"], axis=1), 1, atol=2e-4)
        target = clips / (source.stem + ".npz")
        np.savez(target, **arrays)
        records.append({"name": source.stem, "raw_sha256": item["sha256"], "npz_sha256": sha256(target), "frames": n})
    subprocess.run([sys.executable, "-m", "mjlab.scripts.pack_motions", "--input", str(clips / "*.npz"), "--output", str(args.output / "packed"), "--workers", "2"], check=True)
    ends = np.load(args.output / "packed/motion_ends.npy")
    assert len(ends) == 40 and ends[-1] == sum(row["frames"] for row in records)
    result = {
        "status": "prepared", "dataset": "raw LAFAN40", "clips": 40,
        "input_fps": 30, "output_fps": 50, "frames": int(ends[-1]),
        "hours": int(ends[-1]) / 50 / 3600,
        "converter": "official mjlab_research dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd",
        "raw_manifest_sha256": sha256(args.manifest), "records": records,
        "packed_files": [{"file": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted((args.output / "packed").iterdir())],
        "is_full_BONES_paper_benchmark": False,
    }
    (args.output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({key: result[key] for key in ("status", "clips", "frames", "hours")}))


if __name__ == "__main__":
    main()
