"""Run pinned GEM-X on an existing full 30 Hz fixed-person crop proposal ledger.

Uses public upstream modules, without invoking person detection/track selection,
time normalization, contact IK, or SMPL. Crop proposals are never ground truth.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

GEM_COMMIT = "4aef1452e5df171deff4f293c08e643eb450981e"
GEM_REVISION = "b111f2dcf3ebcf57965fd53499cb645dfa5ba25c"
SOMA_REVISION = "d281db2d01553f7230c56e764cec00fe27ef23f7"
WEIGHTS = {
    "body_s240000.ckpt": (3538753341, "e244d27f2834eb6382c5f6b9cbf2a38681b685514a1b2e86f8f3d2592d614959"),
    "vitpose.pth": (3388483384, "8e3eec71dbd6bb745eba8c6debe3e4b97bcfd1d01f300099f2eedf50b9a80c8e"),
    "motion_native77_e774_s292175.ckpt": (187134331, "c7790f4587e518bcea40cb8281c9be2233d37f6caa668835462542760dcf82aa"),
    "motion_stats_raw.npz": (21320, "b7719b634e0fc1f001005605d117d77448c8d4a110923149df58529c68112ed6"),
}


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True, help="Existing full_30hz directory")
    parser.add_argument("--weights-dir", type=Path, required=True, help="Official GEM files (weights/gem)")
    parser.add_argument("--soma-assets", type=Path, required=True, help="Official SOMA-X v0.3.3 assets")
    parser.add_argument("--gem-root", type=Path, required=True, help="Pinned official GEM-X checkout")
    parser.add_argument("--output", type=Path, required=True, help="New output directory; never overwrites")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--export-joints", action="store_true", help="Also evaluate SOMA FK in meters, without contact IK")
    parser.add_argument("--verify-weight-hashes", action="store_true", help="Rehash large weights (otherwise require verified download receipts)")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    for key in ("prepared", "weights_dir", "soma_assets", "gem_root", "output"):
        setattr(args, key, getattr(args, key).resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    receipt_path = args.output / "receipt.json"
    receipt = {
        "schema": "gem-fixed-identity-run/1.0", "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "preflight", "gem_source_commit_expected": GEM_COMMIT,
        "gem_weights_revision": GEM_REVISION, "soma_assets_revision": SOMA_REVISION,
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "script_sha256": sha256(__file__), "stages": [], "postprocessing": "none",
        "person_reselected": False, "ground_anchors_generated": False,
    }

    def stage(name, **fields):
        receipt["status"] = name
        receipt["stages"].append({"stage": name, "elapsed_s": time.monotonic() - started, **fields})
        receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        print(json.dumps({"stage": name, **fields}), flush=True)

    try:
        if (args.gem_root / ".git").exists():
            commit = subprocess.check_output(["git", "-C", str(args.gem_root), "rev-parse", "HEAD"], text=True).strip()
            if commit != GEM_COMMIT:
                raise ValueError(f"GEM source commit mismatch: {commit}")
            receipt["gem_source_commit_observed"] = commit
        else:
            receipt["gem_source_commit_observed"] = None
            receipt["source_revision_check"] = "archive_without_git; source hashes recorded"
        source_files = ["gem/inference/body_video.py", "gem/inference/motion.py", "gem/common/pose2d/native77.py", "gem/inference/camera_motion.py"]
        receipt["upstream_source_sha256"] = {name: sha256(args.gem_root / name) for name in source_files}
        weight_receipts = {}
        for name, (size, expected) in WEIGHTS.items():
            path = args.weights_dir / name
            if not path.is_file() or path.stat().st_size != size:
                raise FileNotFoundError(f"Missing or wrong-sized official weight: {path}; expected {size} bytes")
            proof_path = path.with_name(path.name + ".download.json")
            if args.verify_weight_hashes:
                actual = sha256(path)
            else:
                proof = json.loads(proof_path.read_text(encoding="utf-8"))
                if proof.get("status") != "verified" or proof.get("revision") != GEM_REVISION or proof.get("size") != size:
                    raise ValueError(f"Weight verification receipt mismatch: {proof_path}")
                actual = proof["sha256"]
            if actual != expected:
                raise ValueError(f"Weight SHA256 mismatch: {path}")
            weight_receipts[name] = {"size": size, "sha256": actual, "verification": "rehash" if args.verify_weight_hashes else "prior_download_receipt_and_current_size"}
        receipt["weights"] = weight_receipts
        manifest_path = args.prepared / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        inputs = {}
        for key in ("output_video", "identity_npz", "frame_map", "bbx_pt"):
            name = manifest[key]
            path = args.prepared / name
            actual = sha256(path)
            if actual != manifest[key + "_sha256"]:
                raise ValueError(f"Prepared input changed: {path}")
            inputs[key] = {"name": name, "sha256": actual}
        if not manifest.get("model_ready_crop_only") or not manifest.get("complete_source_frame_interval_selected"):
            raise ValueError("Prepared video must cover the full selected source interval with usable crop proposals")
        receipt.update(prepared_manifest_sha256=sha256(manifest_path), prepared_inputs=inputs,
                       clip_id=manifest["clip_id"], locked_target_id=manifest["locked_target_id"],
                       frame_map_sha256=manifest["frame_map_sha256"], timing_provenance="unchanged_prepared_source_pts_30hz")

        sys.path.insert(0, str(args.gem_root))
        import numpy as np
        import torch
        from gem.common.detectors.tracks import square_crops
        from gem.common.pose2d import load_native77_detector
        from gem.inference.body_video import body_release, extract_body_features, load_body_video_config, motion_intrinsics
        from gem.inference.camera_motion import camera_motion
        from gem.inference.motion import load_predictor, run_motion
        from gem.inference.serialization import save_predictions
        from gem.inference.video import VideoFrames
        from gem.motion.api import MotionConditionSpec
        from gem.pose.api import load_pose_checkpoint

        config = load_body_video_config(args.weights_dir / "body_video_native77.yaml")
        config["pose"]["batch_size"] = args.batch_size
        config["observations"]["batch_size"] = args.batch_size
        expected = MotionConditionSpec(**config["motion"]["condition_spec"])
        if expected.observation_joint_set != "soma77_native_v1":
            raise ValueError("This entry point requires the pinned native77 release")
        identity = np.load(args.prepared / manifest["identity_npz"], allow_pickle=False)
        boxes = identity["bbx_xyxy"].astype(np.float32)
        frame_times = identity["timestamps_s"].astype(np.float64)
        count = len(frame_times)
        if boxes.shape != (count, 4) or not np.isfinite(boxes).all() or (boxes[:, 2:] <= boxes[:, :2]).any():
            raise ValueError("Invalid fixed-person crop proposals")
        if not identity["crop_bbox_usable"].all():
            raise ValueError("At least one unresolved crop remains; no new filling is performed here")
        if count != manifest["frame_count"] or count < 2 or not np.allclose(frame_times, np.arange(count) / 30, rtol=0, atol=1e-7):
            raise ValueError("Prepared timeline must be complete zero-origin 30 Hz")
        bbx = torch.load(args.prepared / manifest["bbx_pt"], map_location="cpu", weights_only=False)
        boxes_xys = np.asarray(bbx["bbx_xys"], dtype=np.float32)
        if not np.allclose(boxes_xys, square_crops(boxes), rtol=1e-6, atol=1e-4):
            raise ValueError("Frozen crop convention differs from upstream GEM 1.2*max(h,w/.75)")
        stage("inputs_verified", frames=count, original_gate_false=int((~identity["identity_valid"]).sum()),
              interpolated_bbox_proposals=int(identity["bbox_interpolated"].sum()))
        receipt["environment"] = {"torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
                                  "device": args.device, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
        with VideoFrames(args.prepared / manifest["output_video"]) as frames:
            if len(frames) != count or abs(frames.fps - 30) > 1e-6:
                raise ValueError("Video frame count or FPS differs from prepared timeline")
            K = motion_intrinsics(frames.width, frames.height)
            receipt["camera_contract"] = {"method": "background", "intrinsics": K.tolist(),
                "intrinsics_source": "official_diagonal_focal_estimate", "mask_policy": "official_primary_native77_hull",
                "translation_or_court_calibration": "not supplied; raw learned world rollout"}
            stage("native77_started")
            obs = config["observations"]
            detector = load_native77_detector(obs["checkpoint"], release_id=obs["release_id"], device=args.device, batch_size=args.batch_size)
            with torch.inference_mode():
                keypoints = detector.predict(frames, boxes_xys)
            detector_release = detector.release_id
            del detector
            if tuple(keypoints.shape) != (count, 77, 3) or not torch.isfinite(keypoints).all():
                raise ValueError("Native77 output has invalid shape or nonfinite entries")
            save_predictions({"keypoints_native77": keypoints, "frame_times": torch.from_numpy(frame_times)}, args.output / "native77.pt")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            stage("background_camera_started")
            camera = camera_motion(frames, K.numpy(), boxes, mode="background", keypoints=keypoints.numpy())
            save_predictions(camera, args.output / "camera.pt")
            stage("body_features_started")
            release = body_release(config)
            pose = load_pose_checkpoint(config["pose"]["checkpoint"], release, device=args.device)
            features, feature_version = extract_body_features(frames, boxes_xys, K, pose, batch_size=args.batch_size, device=args.device)
            del pose
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        actual = MotionConditionSpec(feature_version, detector_release, "soma77_native_v1")
        if actual != expected or tuple(features.shape) != (count, 1024) or not torch.isfinite(features).all():
            raise ValueError("Native77 / body feature release contract mismatch")
        sequence = {
            "condition_spec": asdict(actual), "length": torch.tensor([count]),
            "f_imgseq": features.unsqueeze(0), "kp2d_native77": keypoints.unsqueeze(0),
            "bbx_xys": torch.from_numpy(boxes_xys).unsqueeze(0),
            "K_fullimg": K.expand(count, 3, 3).unsqueeze(0).clone(),
            "cam_angvel": camera["cam_angvel"].unsqueeze(0),
            "frame_times": torch.from_numpy(frame_times).unsqueeze(0), "sampling_fps": 30.0,
        }
        save_predictions(sequence, args.output / "conditions.pt")
        save_predictions({key: torch.from_numpy(identity[key].copy()) for key in identity.files if key != "source_keypoints_coco17"}, args.output / "input_proposal_ledger.pt")
        stage("motion_started")
        predictor = load_predictor(config["motion"]["checkpoint"], config["motion"], stats_path=config["motion"]["stats"], device=args.device)
        with torch.inference_mode():
            result = run_motion(predictor, sequence, device=args.device)
        for coordinate in ("incam", "global"):
            for key, value in result[f"pred_soma_params_{coordinate}"].items():
                if not torch.isfinite(value).all():
                    raise ValueError(f"Nonfinite prediction: {coordinate}/{key}")
        result["fixed_identity_provenance"] = {
            "prepared_manifest_sha256": receipt["prepared_manifest_sha256"],
            "frame_map_sha256": manifest["frame_map_sha256"], "person_reselected": False,
            "postprocessing": "none", "bbox_is_crop_proposal": True, "contact_ground_truth": False,
        }
        save_predictions(result, args.output / "raw_predictions.pt")
        receipt["output_parameters"] = {coordinate: {key: list(value.shape) for key, value in result[f"pred_soma_params_{coordinate}"].items()} for coordinate in ("incam", "global")}
        del predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        stage("raw_motion_saved")
        if args.export_joints:
            from gem.common.soma import SomaFixedGs1BodyAdapter, SomaRuntimeConfig
            from gem.motion.frontend.joint_sets import get_observation_schema
            body = SomaFixedGs1BodyAdapter(SomaRuntimeConfig(data_root=args.soma_assets, device=args.device,
                lod="xlo", apply_correctives=False, enable_procedural_transforms=True, procedural_reference="runtime"))
            params = result["pred_soma_params_global"]
            values = {key: value.to(args.device) for key, value in params.items()}
            with torch.inference_mode():
                fk = body(**values, pose2rot=True, apply_correctives=False, return_joints_only=True, repose_to_bind_pose=True)
            public_names = [str(name) for name in body.runtime._public_joint_names]
            if len(public_names) != 78 or tuple(fk["transforms"].shape[-3:]) != (78, 4, 4):
                raise ValueError("Unexpected SOMA public FK transform contract")
            if not torch.allclose(fk["joints"], fk["transforms"][..., 1:, :3, 3], atol=1e-6, rtol=0):
                raise ValueError("SOMA public joint positions do not match world transforms")
            save_predictions({"joints": fk["joints"], "joint_names": public_names[1:],
                "observation_schema_names": get_observation_schema("soma77_native_v1").joint_names,
                "transforms": fk["transforms"], "transform_joint_names": public_names,
                "frame_times": sequence["frame_times"], "unit": "meters", "coordinates": "raw_world_rollout_meters",
                "geometry": "SOMA-X0.3.3/reference0.2.0; xlo; correctives=false; no contact IK"}, args.output / "world_joints77.pt")
            stage("world_fk_saved")
        receipt["outputs"] = {path.name: {"size": path.stat().st_size, "sha256": sha256(path)} for path in args.output.glob("*.pt")}
        receipt["elapsed_s"] = time.monotonic() - started
        stage("succeeded")
    except Exception as error:
        receipt["error"] = {"type": type(error).__name__, "message": str(error)}
        receipt["raw_motion_exists"] = (args.output / "raw_predictions.pt").exists()
        stage("failed")
        raise


if __name__ == "__main__":
    main()
