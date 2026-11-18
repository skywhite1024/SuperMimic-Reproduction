"""Reference construction and scoring for the static G1 push experiment."""

import hashlib
import json
from pathlib import Path

import numpy as np

DT = 0.02
PUSH_STEP = 100
STEPS = 400
CLIP_FRAMES = 600
POSES = ("t_pose", "home_stance")
THRESHOLDS = {
    "planar_speed_m_s": 0.10,
    "angular_speed_rad_s": 0.20,
    "pelvis_tilt_deg": 10.0,
    "joint_rms_deg": 10.0,
}
METRICS = (*THRESHOLDS, "pelvis_z_m", "travel_from_push_m")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


def pulse_grid():
    labels = ["zero"]
    vectors = [[0.0, 0.0, 0.0]]
    for label, direction in (("+x", (1, 0)), ("-x", (-1, 0)),
                             ("+y", (0, 1)), ("-y", (0, -1))):
        for speed in np.arange(0.5, 4.01, 0.5):
            labels.append(f"{label}:{speed:.1f}")
            vectors.append([speed * direction[0], speed * direction[1], 0.0])
    return labels, np.asarray(vectors, dtype=np.float32)


def prepare_reference(xml, joints, pack):
    import mujoco

    spec = mujoco.MjSpec.from_file(str(xml))
    spec.worldbody.add_geom(name="push_reference_floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=[2, 2, 0.1])
    model = spec.compile()
    joint_ids = [model.joint(name).id for name in joints]
    addresses = model.jnt_qposadr[joint_ids]
    if (model.nq != 36 or model.nv != 35 or len(joints) != 29
            or not np.array_equal(addresses, np.arange(7, 36))):
        raise RuntimeError("Unexpected G1 free joint or native joint order")
    hard = model.jnt_range[joint_ids]
    midpoint = hard.mean(axis=1)
    halfspan = 0.45 * np.diff(hard, axis=1)[:, 0]
    floor = model.geom("push_reference_floor").id
    feet = [model.geom(f"{side}_foot{i}_collision").id
            for side in ("left", "right") for i in range(1, 8)]

    # In this robot's MJCF, elbow = 0 points the forearm forward with the
    # shoulder rolled out. The lateral reach occurs near elbow = pi / 2.
    t_pose = {"left_shoulder_roll_joint": np.pi / 2.0,
              "right_shoulder_roll_joint": -np.pi / 2.0,
              "left_elbow_joint": np.pi / 2.0,
              "right_elbow_joint": np.pi / 2.0}
    home = {}
    for side, sign in (("left", 1), ("right", -1)):
        home.update({f"{side}_{joint}_joint": value for joint, value in (
            ("hip_pitch", -.1), ("knee", .3), ("ankle_pitch", -.2),
            ("shoulder_pitch", .2), ("shoulder_roll", sign * .2), ("elbow", 1.28),
        )})

    fields = {name: [] for name in ("root_pos_w", "root_quat_w", "root_lin_vel_w",
                                    "root_ang_vel_w", "joint_pos", "joint_vel")}
    poses = {}
    for label, joint_targets in zip(POSES, (t_pose, home), strict=True):
        q = np.zeros(29)
        for name, value in joint_targets.items():
            q[joints.index(name)] = value
        clipped = q.clip(midpoint - halfspan, midpoint + halfspan)
        data = mujoco.MjData(model)
        data.qpos[:7] = [0, 0, 1, 1, 0, 0, 0]
        data.qpos[addresses] = clipped
        def distances():
            return np.asarray([
                mujoco.mj_geomDistance(model, data, floor, foot, 10.0, None) for foot in feet
            ])
        mujoco.mj_forward(model, data)
        data.qpos[2] += -0.004 - distances().min()
        mujoco.mj_forward(model, data)
        clearance = distances()
        if abs(clearance.min() + .004) > 1e-8:
            raise RuntimeError("Reference height does not reproduce the foot clearance")
        arm_vectors = {}
        for side, sign in (("left", 1), ("right", -1)):
            shoulder = data.body(f"{side}_shoulder_roll_link").xpos
            wrist = data.body(f"{side}_wrist_yaw_link").xpos
            arm_vectors[side] = (wrist - shoulder).tolist()
            lateral = wrist - shoulder
            if label == "t_pose" and (sign * lateral[1] < .35 or
                                      np.linalg.norm(lateral[[0, 2]]) > .03):
                raise RuntimeError("FK does not show both T-pose arms extended laterally")
        row = {
            "root_pos_w": data.qpos[:3],
            "root_quat_w": data.qpos[3:7],
            "root_lin_vel_w": np.zeros(3),
            "root_ang_vel_w": np.zeros(3),
            "joint_pos": clipped,
            "joint_vel": np.zeros(29),
        }
        poses[label] = {
            "root_height_m": float(data.qpos[2]),
            "clipped_joints": {joints[i]: float(clipped[i]) for i in np.flatnonzero(clipped != q)},
            "foot_clearance_m": clearance.tolist(),
            "shoulder_to_wrist_m": arm_vectors,
        }
        for name, value in row.items():
            fields[name].append(np.tile(value.astype(np.float32), (CLIP_FRAMES, 1)))

    pack.mkdir(parents=True, exist_ok=False)
    for name, arrays in fields.items():
        np.save(pack / f"{name}.npy", np.concatenate(arrays))
    np.save(pack / "motion_ends.npy", CLIP_FRAMES * np.arange(1, len(POSES) + 1, dtype=np.int64))
    write_json(pack / "joint_names.json", joints)
    write_json(pack / "motion_names.json", list(POSES))
    return {
        "g1_xml_sha256": sha256(xml),
        "fps": round(1 / DT),
        "frames_per_pose": CLIP_FRAMES,
        "soft_limit_factor": 0.9,
        "poses": poses,
        "files": {path.name: sha256(path) for path in sorted(pack.iterdir())},
    }


def derive_metrics(states, reference):
    """Each row is a physical state at the indicated sample time, before censoring."""
    position = states[..., :3].astype(np.float64)
    rotation = states[..., 3:7].astype(np.float64)
    tilt_cos = 1.0 - 2.0 * np.square(rotation[..., 1:3]).sum(axis=-1)
    tilt = np.rad2deg(np.arccos(np.clip(tilt_cos, -1.0, 1.0)))
    joint_difference = states[..., 7:36].astype(np.float64) - reference[7:36]
    joint_rms = np.rad2deg(np.sqrt(np.square(joint_difference).mean(axis=-1)))
    linear = np.linalg.norm(states[..., 36:38].astype(np.float64), axis=-1)
    angular = np.linalg.norm(states[..., 39:42].astype(np.float64), axis=-1)
    travel = np.linalg.norm(position[..., :2] - position[:, PUSH_STEP:PUSH_STEP+1, :2], axis=-1)
    travel[:, :PUSH_STEP] = 0.0
    series = np.stack((linear, angular, tilt, joint_rms, position[..., 2], travel), axis=-1)
    return series.astype(np.float32)


def score_trials(labels, vectors, metrics, steps, applied):
    usable = (metrics[..., :len(THRESHOLDS)] <= list(THRESHOLDS.values())).all(axis=-1)
    records = []
    for i, label in enumerate(labels):
        last = int(steps[i])
        # The last sample can already be fallen even at the complete horizon.
        row = metrics[i, last]
        height_error = abs(float(row[4]) - float(metrics[i, 0, 4]))
        fallen = bool(height_error > .25 or row[2] > np.rad2deg(np.arccos(.20)))
        if last < STEPS and not fallen:
            raise RuntimeError("Short trial is missing its terminal fall state")
        settled = None
        if not fallen and last == STEPS:
            candidates = np.flatnonzero(~usable[i, PUSH_STEP + 1:])
            # PUSH_STEP is the pre-impulse sample. Post-impulse scoring starts at +dt.
            first = (int(candidates[-1]) + 2) if len(candidates) else 1
            if STEPS - PUSH_STEP - first >= round(.5 / DT):
                settled = round(first * DT, 8)
        records.append({
            "case": label, "delta_velocity_m_s": vectors[i].tolist(),
            "observed_steps": last, "impulse_applied": bool(applied[i]), "fell": fallen,
            "fall_time_from_start_s": round(last * DT, 8) if fallen else None,
            "fall_time_relative_to_push_s": round((last - PUSH_STEP) * DT, 8) if fallen else None,
            "prepush_stable_final_1s": bool(last >= PUSH_STEP and usable[i, PUSH_STEP-50:PUSH_STEP+1].all()),
            "final_1s_stable": bool(not fallen and usable[i, STEPS-50:].all()),
            "recovery_seconds": settled,
            "final_travel_m": float(row[5]) if last >= PUSH_STEP else None,
        })
    return records
