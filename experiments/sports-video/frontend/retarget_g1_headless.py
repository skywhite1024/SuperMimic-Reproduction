"""Retarget every frame of a verified 30 Hz GVHMR result with pinned GMR.

Public source only is provided here; this runner requires the user's licensed
SMPL-X file. No GUI, dropped first frame, automatic floor clamp, or smoothing.
Output is robot_adapted and is not a measured metric human court trajectory.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

PIN = "25f56db79ac411258b377b6d4ccba0c8ff3372ac"
GVHMR_PIN = "ee960bb6e2ea2d381aa97f08e9b71ef320b624b1"
BODY_MODEL_KEY = "inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_body_model(body_file, receipt):
    """Bind FK reconstruction to the exact body model used for inference."""
    if (receipt.get("schema_version") != "gvhmr-fixed-identity-run/1.0"
            or receipt.get("source_commit") != GVHMR_PIN):
        raise ValueError("Expected a receipt from the pinned GVHMR adapter")
    expected = receipt.get("model_sha256", {}).get(BODY_MODEL_KEY)
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("Inference receipt does not identify its SMPL-X model")
    actual = sha(body_file)
    if actual != expected:
        raise ValueError("SMPL-X model differs from the model used for GVHMR inference; reuse the exact NPZ")
    return actual


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True, help="Explicit pinned official GMR checkout")
    p.add_argument("--gvhmr-result", type=Path, required=True)
    p.add_argument("--inference-receipt", type=Path, required=True)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--body-model-root", type=Path, required=True, help="Parent of the exact smplx/SMPLX_NEUTRAL.npz used by GVHMR")
    p.add_argument("--body-model-ext", choices=["npz"], required=True)
    p.add_argument("--mapping-audit", type=Path, required=True, help="Compiled G1 mapping audit for the exact source and runtime XML hashes")
    p.add_argument("--runtime-xml", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    source, result, receipt_path, prepared, body_root, audit_path, runtime_xml, out = [x.resolve() for x in
        (a.source_root, a.gvhmr_result, a.inference_receipt, a.prepared, a.body_model_root, a.mapping_audit, a.runtime_xml, a.output_dir)]
    if out.exists():
        raise ValueError("Output directory exists; select a fresh output")
    if subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip() != PIN:
        raise ValueError("GMR source differs from the pinned commit")
    subprocess.run(["git", "-C", str(source), "diff", "--exit-code", "HEAD", "--"], check=True, capture_output=True)
    m = json.loads((prepared/"manifest.json").read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    source_xml = source/"assets/unitree_g1/g1_mocap_29dof.xml"
    ik_file = source/"general_motion_retargeting/ik_configs/smplx_to_g1.json"
    body_file = body_root/f"smplx/SMPLX_NEUTRAL.{a.body_model_ext}"
    if not body_file.is_file():
        raise FileNotFoundError(f"User-supplied licensed body model required: {body_file}")
    body_model_sha256 = verify_body_model(body_file, receipt)
    if (sha(source_xml) != audit["source_xml_sha256"] or sha(runtime_xml) != audit["runtime_xml_sha256"]
            or not audit["mapping"]["verified"]):
        raise ValueError("G1 XML hashes differ from the compiled mapping audit; rerun the audit")
    if (receipt["status"] != "gvhmr_inference_complete_not_robot_evaluated"
            or sha(result) != receipt["output_sha256"]
            or sha(prepared/"manifest.json") != receipt["prepared_manifest_sha256"]):
        raise ValueError("GVHMR result does not belong to this prepared clip and time grid")
    if m["target_fps"] != 30 or receipt["frame_count"] != m["frame_count"]:
        raise ValueError("Input does not follow the audited 30 Hz route")
    if sha(prepared/m["frame_map"]) != m["frame_map_sha256"]:
        raise ValueError("Source PTS mapping hash mismatch")
    frame_map = json.loads((prepared/m["frame_map"]).read_text(encoding="utf-8"))
    times = np.asarray([f["time_seconds"] for f in frame_map])
    if len(times) != m["frame_count"] or not np.allclose(times, np.arange(len(times))/30, rtol=0, atol=1e-7):
        raise ValueError("Invalid prepared reference timestamps")
    os.chdir(source)
    sys.path.insert(0, str(source))
    import smplx
    from qpsolvers import available_solvers
    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    from general_motion_retargeting.utils.smpl import load_gvhmr_pred_file, get_gvhmr_data_offline_fast
    if "daqp" not in available_solvers:
        raise RuntimeError("Pinned GMR defaults to daqp; install its explicit solver dependency before running")
    # Upstream leaves ext implicit and its README discusses a site-package edit.
    # Scope the explicit chosen file extension to this one loading call instead.
    original_create = smplx.create
    def create_with_explicit_ext(*args, **kwargs):
        kwargs["ext"] = a.body_model_ext
        return original_create(*args, **kwargs)
    out.mkdir(parents=True)
    qpos_list = []
    frame_index = -1
    try:
        smplx.create = create_with_explicit_ext
        try:
            data, body, posed, human_height = load_gvhmr_pred_file(str(result), body_root)
        finally:
            smplx.create = original_create
        frames, aligned_fps = get_gvhmr_data_offline_fast(data, body, posed, tgt_fps=30)
        if len(frames) != len(times) or float(aligned_fps) != 30:
            raise ValueError("Upstream SMPL conversion changed frame count or frame rate")
        retarget = GMR(src_human="smplx", tgt_robot="unitree_g1", actual_human_height=float(human_height),
                       solver="daqp", verbose=False, use_velocity_limit=False)
        if sha(retarget.xml_file) != audit["source_xml_sha256"]:
            raise ValueError("Instantiated GMR model differs from the verified source XML")
        for frame_index, human_frame in enumerate(frames):
            q = np.asarray(retarget.retarget(human_frame, offset_to_ground=False), dtype=np.float64).copy()
            if q.shape != (36,) or not np.isfinite(q).all() or abs(np.linalg.norm(q[3:7])-1) > 1e-4:
                raise ValueError(f"Invalid retargeted pose at frame {frame_index}")
            qpos_list.append(q)
        qpos = np.asarray(qpos_list)
        mp = audit["mapping"]
        joint = qpos[:, 7:][:, mp["permutation"]]*np.asarray(mp["sign"])+np.asarray(mp["offset_rad"])
        motion = np.concatenate([qpos[:, :3], qpos[:, [4,5,6,3]], joint], axis=1)
        np.savetxt(out/"raw_robot_adapted.csv", motion, delimiter=",", fmt="%.17g")
        np.savez(out/"raw_robot_adapted.npz", root_pos=motion[:,:3], root_rot=motion[:,3:7],
                 dof_pos=motion[:,7:], fps=np.asarray(30), timestamps_s=times)
        limits = np.asarray([r["runtime_range_rad"] for r in audit["joint_rows"]])
        violations = np.argwhere((joint < limits[:,0]-1e-6) | (joint > limits[:,1]+1e-6))
        mapping = dict(mp)
        mapping["evidence"] = f"{audit_path}; SHA256={sha(audit_path)}; recorded XML hashes verified at export"
        root_scale = float(retarget.human_scale_table[retarget.human_root_name])
        contract = {"schema_version": "sports-reference/1.0", "representation": "robot_adapted",
            "coordinate_frame": "z_up_right_handed", "world_frame_id": f"gvhmr_gmr_uncalibrated:{m['clip_id']}",
            "quaternion_order": "xyzw", "position_unit": "m", "joint_unit": "rad", "joint_names": mp["target_joint_names"],
            "time": {"provenance": "resampled_source_pts", "timestamps_s": times.tolist(), "motion_fps": 30,
                     "source_video_sha256": m["source_video_sha256"],
                     "resampling_evidence": f"{prepared/'frame-map.json'}; SHA256={m['frame_map_sha256']}",
                     "prepared_manifest_sha256": sha(prepared/"manifest.json"),
                     "source_start_time_s": frame_map[0]["original_video_time_seconds"],
                     "actual_source_coverage_end_s": frame_map[-1]["original_video_time_seconds"],
                     "crop_duration_s": m["crop_duration_s"]},
            "adaptation": {"root_xy_scale": root_scale, "time_scale": 1.0,
                "human_height_from_upstream_betas_m": float(human_height), "human_scale_table": retarget.human_scale_table,
                "description": "GMR scales pelvis/world trajectory by pelvis factor as well as body proportions; no metric court calibration; no post-export XY, z, orientation or time repair"},
            "joint_mapping": mapping, "joint_limits_rad": limits.tolist(),
            "robot_xml_sha256": audit["runtime_xml_sha256"], "source_robot_xml_sha256": sha(source_xml),
            "ik_config_sha256": sha(ik_file), "motion_sha256": sha(out/"raw_robot_adapted.csv"),
            "source_gvhmr_sha256": sha(result), "body_model_sha256": body_model_sha256,
            "gmr_commit": PIN, "velocity_constraints": "upstream default disabled; audit actual reference velocities separately"}
        (out/"raw_robot_adapted.contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
        report = {"status": "retargeted_not_physical_success", "frames_in": len(times), "frames_out": len(motion),
            "includes_first_frame": True, "runtime_limit_violation_count": len(violations),
            "runtime_limit_violations": [{"frame_index": int(i), "joint_name": mp["target_joint_names"][j],
                "value_rad": float(joint[i,j]), "limits_rad": limits[j].tolist()} for i,j in violations],
            "body_model_ext_explicit": a.body_model_ext, "floor_offset_to_ground": False, "smoothing": False,
            "body_model_matches_inference": True, "body_model_sha256": body_model_sha256,
            "joint_clipping": False, "solver": "daqp", "script_sha256": sha(__file__),
            "versions": {x: importlib.metadata.version(x) for x in ["mujoco", "mink", "smplx", "torch", "numpy", "scipy", "qpsolvers", "daqp"]}}
        (out/"retarget-receipt.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({k: report[k] for k in ["status", "frames_in", "frames_out", "runtime_limit_violation_count"]}, indent=2))
    except BaseException as e:
        if qpos_list:
            np.savez(out/"partial_failed_qpos_wxyz.npz", qpos=np.asarray(qpos_list))
        (out/"failure.json").write_text(json.dumps({"status": "failed", "frame_index": frame_index,
            "completed_frames": len(qpos_list), "error_type": type(e).__name__, "error": str(e)}, indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
