"""Hold the canonical LAFAN walk1_subject1 first frame without changing its pose."""

import argparse
import json
from pathlib import Path

import numpy as np

from push_recovery_contract import CLIP_FRAMES, sha256, write_json

POSE = "lafan_walk1_first_frame"
CLIP = "walk1_subject1"
SHAPES = {"root_pos_w": 3, "root_quat_w": 4, "joint_pos": 29, "joint_vel": 29,
          "root_lin_vel_w": 3, "root_ang_vel_w": 3}


def prepare(canonical, xml, output):
    import mujoco

    names = json.loads((canonical / "motion_names.json").read_bytes())
    if names.count(CLIP) != 1:
        raise ValueError("Expected exactly one canonical walk1_subject1 clip")
    ends = np.load(canonical / "motion_ends.npy", allow_pickle=False)
    index = names.index(CLIP)
    frame = int(ends[index - 1]) if index else 0
    arrays = {}
    for name, dim in SHAPES.items():
        filename = canonical / f"{name}.npy"
        source = np.load(filename, allow_pickle=False, mmap_mode="r")
        if source.dtype != np.float32 or source.shape != (ends[-1], dim):
            raise ValueError(f"Unexpected shape or dtype: {filename}")
        arrays[name] = source[frame] if name in {"root_pos_w", "root_quat_w", "joint_pos"} else np.zeros(dim, np.float32)
    spec = mujoco.MjSpec.from_file(str(xml))
    spec.worldbody.add_geom(name="reference_floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, .1])
    model = spec.compile()
    joints = json.loads((canonical / "joint_names.json").read_bytes())
    ids = [model.joint(name).id for name in joints]
    if not np.array_equal(model.jnt_qposadr[ids], np.arange(7, 36)):
        raise ValueError("Unexpected canonical joint order")
    hard = model.jnt_range[ids]
    middle = hard.mean(axis=1)
    radius = .45 * (hard[:, 1] - hard[:, 0])
    if np.any(np.abs(arrays["joint_pos"] - middle) > radius + 1e-7):
        raise ValueError("Canonical frame is outside the native soft limits")
    data = mujoco.MjData(model)
    data.qpos[:] = np.concatenate([arrays[name] for name in ("root_pos_w", "root_quat_w", "joint_pos")])
    mujoco.mj_kinematics(model, data)
    arms = {}
    for side, sign in (("left", 1), ("right", -1)):
        delta = data.body(f"{side}_wrist_yaw_link").xpos - data.body(f"{side}_shoulder_roll_link").xpos
        if sign * delta[1] < .35 or np.linalg.norm(delta[[0, 2]]) > .06:
            raise ValueError("The canonical frame does not have two laterally extended arms")
        arms[side] = delta.tolist()
    floor = model.geom("reference_floor").id
    feet = [model.geom(f"{side}_foot{i}_collision").id
            for side in ("left", "right") for i in range(1, 8)]
    pack = output / "packed"
    pack.mkdir(parents=True, exist_ok=False)
    for key, value in arrays.items():
        np.save(pack / f"{key}.npy", np.tile(value, (CLIP_FRAMES, 1)))
    np.save(pack / "motion_ends.npy", np.array([CLIP_FRAMES], dtype=np.int64))
    write_json(pack / "joint_names.json", joints)
    write_json(pack / "motion_names.json", [POSE])
    report = {"canonical_clip": CLIP, "canonical_clip_frame": 0, "canonical_packed_frame": frame,
              "fps": 50, "frames": CLIP_FRAMES, "pose_unchanged": True, "reference_velocities_zero": True,
              "root_height_m": float(data.qpos[2]), "g1_xml_sha256": sha256(xml),
              "shoulder_to_wrist_m": arms,
              "foot_floor_distances_m": [mujoco.mj_geomDistance(model, data, floor, g, 10., None) for g in feet],
              "source_pack_sha256": {path.name: sha256(path) for path in sorted(canonical.iterdir()) if path.is_file()},
              "static_pack_sha256": {path.name: sha256(path) for path in sorted(pack.iterdir())}}
    write_json(output / "reference.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("canonical", "xml", "output"):
        parser.add_argument("--" + argument, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.canonical, args.xml, args.output)), flush=True)
