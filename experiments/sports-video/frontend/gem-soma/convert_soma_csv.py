"""Convert official soma-retargeter CSV to SuperMimic CSV, without pose repairs.

Input: named columns, cm, extrinsic XYZ Euler degrees, joint degrees.
Output: no header, root xyz metres, quaternion xyzw, 29 joints radians.
The required input FPS is external evidence, not guessed from a headerless CSV.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT_HEADER = ["Frame", "root_translateX", "root_translateY", "root_translateZ",
               "root_rotateX", "root_rotateY", "root_rotateZ"]


def runtime_joints(xml: Path):
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(xml))
    if model.nq != 36 or model.nv != 35:
        raise ValueError(f"Expected G1 nq=36,nv=35, got {model.nq},{model.nv}")
    entries = sorted((int(model.jnt_qposadr[j]), model.joint(j).name, j)
                     for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE)
    if len(entries) != 29 or [x[0] for x in entries] != list(range(7, 36)):
        raise ValueError("Runtime XML must contain one free root followed by 29 hinge joints")
    return model, [x[1] for x in entries], [x[2] for x in entries]


def convert(input_path: Path, output_path: Path, runtime_xml: Path, input_fps: float,
            output_fps: float = 30.0):
    if not np.isfinite(input_fps) or input_fps <= 0 or output_fps != 30.0:
        raise ValueError("Input FPS must be positive and output FPS must be 30")
    with input_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        values = np.asarray(list(reader), dtype=np.float64)
    if header[:7] != ROOT_HEADER or len(set(header)) != len(header):
        raise ValueError("Unexpected or duplicate official CSV header")
    model, names, ids = runtime_joints(runtime_xml)
    if len(header) != 36 or set(header[7:]) != set(names):
        raise ValueError("CSV joint names do not exactly match runtime 29 DOF")
    if values.ndim != 2 or values.shape[1] != 36 or len(values) < 1 or not np.isfinite(values).all():
        raise ValueError("CSV must contain finite frames with 36 numeric columns")
    if not np.array_equal(values[:, 0], np.arange(len(values))):
        raise ValueError("Frame indices must be contiguous starting at zero")
    order = [header.index(name) for name in names]
    q = np.column_stack((values[:, 1:4] * 0.01,
                         Rotation.from_euler("xyz", values[:, 4:7], degrees=True).as_quat(),
                         np.deg2rad(values[:, order])))
    source_times = np.arange(len(q)) / input_fps
    out_times = np.arange(int(np.floor(source_times[-1] * output_fps + 1e-8)) + 1) / output_fps
    if len(q) > 1 and not (input_fps == output_fps):
        output = np.empty((len(out_times), 36), dtype=np.float64)
        for j in [0, 1, 2, *range(7, 36)]:
            output[:, j] = np.interp(out_times, source_times, q[:, j])
        output[:, 3:7] = Slerp(source_times, Rotation.from_quat(q[:, 3:7]))(out_times).as_quat()
        q = output
    # Quaternion sign continuity changes representation only, not the orientation.
    for i in range(1, len(q)):
        if q[i, 3:7] @ q[i - 1, 3:7] < 0:
            q[i, 3:7] *= -1
    violations = []
    for col, (name, joint_id) in enumerate(zip(names, ids), start=7):
        if model.jnt_limited[joint_id]:
            lo, hi = model.jnt_range[joint_id]
            excess = np.maximum(lo - q[:, col], q[:, col] - hi)
            if np.max(excess) > 1e-5:
                violations.append({"joint": name, "frames": int(np.count_nonzero(excess > 1e-5)),
                                   "max_excess_rad": float(np.max(excess))})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, q, delimiter=",", fmt="%.10g")
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    receipt = {"schema": "soma-csv-to-supermimic/1", "input": str(input_path.resolve()),
               "input_sha256": digest(input_path), "output": str(output_path.resolve()),
               "output_sha256": digest(output_path), "runtime_xml": str(runtime_xml.resolve()),
               "runtime_xml_sha256": digest(runtime_xml), "input_fps": input_fps,
               "fps": output_fps, "input_frames": len(values), "frames": len(q), "columns": 36,
               "last_source_sample_seconds": float(source_times[-1]),
               "last_output_sample_seconds": float(out_times[-1]),
               "root_units": "metres", "root_quaternion": "xyzw", "joint_units": "radians",
               "joint_names": names, "source_column_permutation": order,
               "resampling": "none" if input_fps == output_fps else "linear xyz/joints and quaternion SLERP",
               "pose_repairs": [], "runtime_joint_limit_violations": violations,
               "runtime_limit_eligible": not violations,
               "validation_scope": "format-and-range only; no dynamics or sports success"}
    output_path.with_suffix(output_path.suffix + ".json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    return q, receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-xml", type=Path, required=True)
    parser.add_argument("--input-fps", type=float, required=True)
    args = parser.parse_args()
    _, receipt = convert(args.input, args.output, args.runtime_xml, args.input_fps)
    print(json.dumps(receipt, ensure_ascii=False))
