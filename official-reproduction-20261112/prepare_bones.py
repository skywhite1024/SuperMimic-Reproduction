"""Adapt raw BONES G1 CSVs, then use the pinned author's resampling/velocities.

The BONES viewer at d9963c21bee413e7eb63c1b88ae01cdcb2cf3601 defines
120 FPS, centimetres, degrees and Three.js Euler ZYX (Rz @ Ry @ Rx).
No name/feasibility filtering, angle unwrapping or ground calibration is applied.
This is the raw corpus, not the paper's unpublished exact filtered split.
"""

import argparse
import contextlib
import hashlib
import io
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from mjlab.scripts.csv_to_npz import JOINT_NAMES, MotionLoader
from scipy.spatial.transform import Rotation

HEADER = (
    "Frame",
    "root_translateX",
    "root_translateY",
    "root_translateZ",
    "root_rotateX",
    "root_rotateY",
    "root_rotateZ",
    *(name + "_dof" for name in JOINT_NAMES),
)


def read_bones(path):
    with path.open(encoding="utf-8-sig") as stream:
        header = tuple(value.strip() for value in stream.readline().split(","))
        if len(header) != len(set(header)) or set(header) != set(HEADER):
            raise ValueError(f"Unexpected BONES columns: {path}")
        raw = np.loadtxt(stream, delimiter=",", ndmin=2)
    if raw.shape[1] != len(header) or not np.isfinite(raw).all():
        raise ValueError(f"Invalid BONES values: {path}")
    raw = raw[:, [header.index(name) for name in HEADER]]
    if len(raw) < 6 or not np.array_equal(np.diff(raw[:, 0]), np.ones(len(raw) - 1)):
        raise ValueError(f"Insufficient or nonconsecutive frames: {path}")
    positions = raw[:, 1:4] * 0.01
    # SciPy lowercase xyz is extrinsic XYZ: Rz @ Ry @ Rx, like THREE Euler ZYX.
    quat_xyzw = Rotation.from_euler("xyz", raw[:, 4:7], degrees=True).as_quat()
    joints = np.deg2rad(raw[:, 7:])
    return np.concatenate((positions, quat_xyzw, joints), axis=1)


class BonesMotionLoader(MotionLoader):
    """Only adapt the input representation; inherit all author numerics."""

    def _load_motion(self):
        motion = torch.from_numpy(read_bones(Path(self.motion_file))).to(torch.float32)
        self.motion_base_poss_input = motion[:, :3]
        self.motion_base_rots_input = motion[:, 3:7][:, [3, 0, 1, 2]]
        self.motion_dof_poss_input = motion[:, 7:]
        self.input_frames = len(motion)
        self.duration = (self.input_frames - 1) * self.input_dt


def convert_one(task):
    source_root, output, item = task
    torch.set_num_threads(1)
    source = Path(source_root) / item["path"]
    raw_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if raw_hash != item["sha256"]:
        raise ValueError(f"Raw checksum mismatch: {source}")
    with contextlib.redirect_stdout(io.StringIO()):
        motion = BonesMotionLoader(str(source), 120.0, 50.0, device="cpu")
    arrays = motion.arrays()
    n = motion.output_frames
    for name, array in arrays.items():
        if name not in {"fps", "joint_names"} and (
            len(array) != n or array.dtype != np.float32 or not np.isfinite(array).all()
        ):
            raise ValueError(f"Invalid converted field {name}: {source}")
    if not np.allclose(np.linalg.norm(arrays["root_quat_w"], axis=1), 1, atol=2e-4):
        raise ValueError(f"Nonunit converted quaternion: {source}")
    target = Path(output) / "npz" / (source.stem + ".npz")
    with target.open("xb") as stream:
        np.savez(stream, **arrays)
    return {
        "status": "converted",
        "path": item["path"],
        "name": source.stem,
        "raw_sha256": raw_hash,
        "input_frames": motion.input_frames,
        "frames": n,
        "npz_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    }


def convert_record(task):
    try:
        return convert_one(task)
    except Exception as error:  # noqa: BLE001 - retain each invalid clip in the ledger
        return {
            "status": "failed",
            "path": task[2]["path"],
            "error": str(error),
            "error_type": type(error).__name__,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-seconds", type=int, default=7200)
    args = parser.parse_args()
    records = [
        json.loads(line) for line in (args.source / "raw-csv-manifest.jsonl").open()
    ]
    records.sort(key=lambda row: row["path"])
    if len({Path(row["path"]).stem for row in records}) != len(records):
        raise ValueError("Duplicate motion names")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("limit must be positive")
        records = records[: args.limit]
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "npz").mkdir()
    started = time.monotonic()
    count, failed, frames, raw_frames = 0, 0, 0, 0
    with (
        (args.output / "conversion-manifest.jsonl").open(
            "x", encoding="utf-8"
        ) as journal,
        ProcessPoolExecutor(max_workers=args.workers) as pool,
    ):
        for offset in range(0, len(records), 256):
            if time.monotonic() - started > args.max_seconds:
                break
            tasks = (
                (str(args.source), str(args.output), item)
                for item in records[offset : offset + 256]
            )
            for row in pool.map(convert_record, tasks, chunksize=8):
                journal.write(json.dumps(row) + "\n")
                if row["status"] == "failed":
                    failed += 1
                else:
                    count += 1
                    frames += row["frames"]
                    raw_frames += row["input_frames"]
            journal.flush()
            print(
                json.dumps(
                    {
                        "clips": count,
                        "failed": failed,
                        "of": len(records),
                        "frames": frames,
                        "seconds": round(time.monotonic() - started, 1),
                    }
                ),
                flush=True,
            )
    result = {
        "status": "converted" if count == len(records) else "partial",
        "clips": count,
        "failed": failed,
        "unprocessed": len(records) - count - failed,
        "input_frames": raw_frames,
        "frames": frames,
        "input_fps": 120,
        "output_fps": 50,
        "name_filter_applied": False,
        "feasibility_filter_applied": False,
        "is_full_BONES_paper_benchmark": False,
        "author_resampling_and_velocity_code_unchanged": True,
        "seconds": round(time.monotonic() - started, 1),
    }
    (args.output / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
