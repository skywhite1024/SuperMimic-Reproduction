"""Apply an explicit Appendix E proxy to the existing 50 Hz G1 pack."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from push_recovery_contract import sha256, write_json

FPS = 50
PAPER_COUNTS = {"raw": 142220, "name": 129667, "feasibility": 129514}
EXTRA_TERMS = ("jump_off", "jump_on_50cm", "look_over_fence")
STANDING = {"pelvis_axis_z_gt": .9, "pelvis_z_gt_m": .65, "root_speed_lt_m_s": .25}
RULES = ("elevated_support", "ground_penetration", "floating_feet")
_WORKER = None


def init_worker(xml, packed):
    """Each process has independent MuJoCo scratch data and read-only array views."""
    global _WORKER
    import mujoco

    spec = mujoco.MjSpec.from_file(str(xml))
    spec.worldbody.add_geom(name="filter_floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=[0, 0, .05])
    model = spec.compile()
    names = json.loads((packed / "joint_names.json").read_bytes())
    adr = np.array([model.joint(name).qposadr[0] for name in names])
    if model.nq != 36 or model.nv != 35 or not np.array_equal(adr, np.arange(7, 36)):
        raise ValueError("Unexpected G1 joint addresses")
    floor = model.geom("filter_floor").id
    feet = tuple(model.geom(f"{side}_foot{i}_collision").id
                 for side in ("left", "right") for i in range(1, 8))
    pelvis = model.body("pelvis").id
    data = mujoco.MjData(model)
    ends, pos, quat, joints, linear = (
        np.load(packed / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        for name in ("motion_ends", "root_pos_w", "root_quat_w", "joint_pos", "root_lin_vel_w")
    )
    if any(x.shape != (ends[-1], dim) or x.dtype != np.float32
           for x, dim in ((pos, 3), (quat, 4), (joints, 29), (linear, 3))):
        raise ValueError("Unexpected packed reference array shape or dtype")
    _WORKER = (mujoco, model, data, floor, feet, pelvis, ends, pos, quat, joints, linear)


def filter_block(indices):
    mujoco, model, data, floor, feet, pelvis, ends, pos, quat, joints, linear = _WORKER
    flags = np.zeros((len(indices), len(RULES)), dtype=np.bool_)
    kin = mujoco.mj_kinematics
    distance = mujoco.mj_geomDistance
    qpos = data.qpos
    heights = data.xpos[1:, 2]  # Excludes world body.
    for row, i in enumerate(indices):
        start = int(ends[i - 1]) if i else 0
        end = int(ends[i])
        positions, rotations, joint_values = pos[start:end], quat[start:end], joints[start:end]
        speed_below_gate = (np.square(linear[start:end].astype(np.float64)).sum(axis=1)
                            < STANDING["root_speed_lt_m_s"] ** 2)
        could_stand = speed_below_gate & (positions[:, 2] > STANDING["pelvis_z_gt_m"])
        severe = False
        soft_count = elev_run = float_run = max_elev = max_float = 0
        for t in range(end - start):
            qpos[:3] = positions[t]
            qpos[3:7] = rotations[t]
            qpos[7:] = joint_values[t]
            kin(model, data)
            z = float(heights.min())
            severe |= z < -.12
            soft_count += z < -.08
            elev_run = elev_run + 1 if z > .20 else 0
            max_elev = max(max_elev, elev_run)
            if (could_stand[t] and data.xmat[pelvis, 8] > STANDING["pelvis_axis_z_gt"]
                    and all(distance(model, data, geom, floor, 1.0, None) > .02 for geom in feet)):
                float_run += 1
                max_float = max(max_float, float_run)
            else:
                float_run = 0
        flags[row] = (max_elev >= 30, severe or soft_count * 5 > (end - start) * 4, max_float >= 25)
    return indices, flags


def filter_pack(xml, packed, name_rules, output, workers, limit=None):
    start = time.monotonic()
    names_file = packed / "motion_names.json"
    ends_file = packed / "motion_ends.npy"
    names = json.loads(names_file.read_bytes())
    audit = json.loads(name_rules.read_bytes())
    ends = np.load(ends_file, allow_pickle=False)
    if (len(set(names)) != len(names) or len(names) != len(ends) or ends[0] <= 0
            or not np.issubdtype(ends.dtype, np.integer) or not np.all(np.diff(ends) > 0)
            or sha256(names_file) != audit["names_sha256"]):
        raise ValueError("Expected the verified BONES pack and original name order")
    lengths = np.diff(ends, prepend=0)
    terms = tuple(key.lower() for key in audit["keyword_matches_overlapping"]) + EXTRA_TERMS
    passed = np.asarray([i for i, name in enumerate(names)
                         if not any(term in name.lower() for term in terms)], dtype=np.int64)
    inspect = passed if limit is None else passed[:limit]
    blocks = [inspect[i:i + 512] for i in range(0, len(inspect), 512)]
    print(f"FILTER_START raw={len(names)} name_pass={len(passed)} geometry_scan={len(inspect)} "
          f"frames={int(lengths[inspect].sum())}", flush=True)
    output.mkdir(parents=True, exist_ok=False)
    all_flags = np.zeros((len(names), len(RULES)), dtype=np.bool_)
    seen = np.zeros(len(names), dtype=np.bool_)
    if workers == 1:
        init_worker(xml, packed)
        results = map(filter_block, blocks)
        accumulate(results, all_flags, seen)
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=init_worker,
                                 initargs=(xml, packed)) as pool:
            accumulate(pool.map(filter_block, blocks), all_flags, seen)
    if not np.array_equal(np.flatnonzero(seen), inspect):
        raise RuntimeError("Missing or extra geometry observations")
    feasible = seen & ~all_flags.any(axis=1)
    bits = np.packbits(feasible, bitorder="little").tobytes()
    (output / "feasible-mask.bin").write_bytes(bits)
    np.save(output / "geometry-flags.npy", all_flags)
    indices = np.flatnonzero(feasible)
    receipt = {
        "status": "full_proxy_filter_complete" if limit is None else "partial_geometry_scan",
        "paper_exact_membership": False, "paper_counts": PAPER_COUNTS,
        "g1_xml_sha256": sha256(xml), "input_names_sha256": sha256(names_file),
        "input_motion_ends_sha256": sha256(ends_file),
        "name_rule_input_sha256": sha256(name_rules),
        "community_reference": "https://github.com/Camtrixxx/SuperMimic/tree/a6ae3e2fb5231b0f1b99eb8d70c4cb1e2b8ba6c5",
        "name_terms": terms, "standing_thresholds": STANDING,
        "geometry_thresholds": {"elevated_z_gt_m": .20, "elevated_frames_at_least": 30,
                                "penetration_z_lt_m": -.12, "soft_penetration_z_lt_m": -.08,
                                "soft_penetration_fraction_gt": .8, "foot_clearance_gt_m": .02,
                                "floating_frames_at_least": 25},
        "geometry_measurement": "MuJoCo kinematics and per-capsule mj_geomDistance to plane",
        "raw_clips": len(names), "name_pass_clips": len(passed), "geometry_scanned_clips": len(inspect),
        "geometry_scanned_frames": int(lengths[inspect].sum()),
        "rule_hits_may_overlap": dict(zip(RULES, all_flags.sum(0).tolist(), strict=True)),
        "geometry_excluded": int(all_flags.any(1).sum()),
        "feasible_clips": len(indices), "feasible_frames": int(lengths[indices].sum()),
        "feasible_hours": int(lengths[indices].sum()) / FPS / 3600,
        "feasible_mask_sha256": hashlib.sha256(bits).hexdigest(),
        "geometry_flags_sha256": sha256(output / "geometry-flags.npy"),
        "elapsed_seconds": time.monotonic() - start,
    }
    write_json(output / "filter-result.json", receipt)
    print("FILTER_RESULT " + json.dumps(receipt), flush=True)


def accumulate(results, all_flags, seen):
    for number, (indices, flags) in enumerate(results, 1):
        if seen[indices].any():
            raise RuntimeError("Duplicate scan indices")
        all_flags[indices] = flags
        seen[indices] = True
        if number == 1 or number % 20 == 0:
            print(f"FILTER_PROGRESS blocks={number} completed={seen.sum()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--packed", type=Path, required=True)
    parser.add_argument("--name-rules", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.workers <= 0 or args.limit is not None and args.limit <= 0:
        parser.error("workers and optional limit must be positive")
    filter_pack(args.xml, args.packed, args.name_rules, args.output, args.workers, args.limit)


if __name__ == "__main__":
    main()
