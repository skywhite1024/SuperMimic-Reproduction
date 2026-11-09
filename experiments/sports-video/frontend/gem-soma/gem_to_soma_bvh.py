"""Raw native77 GEM world parameters -> full SOMA BVH; optional official G1 IK.

Uses the official fixed-GS1 geometry with SOMA 0.3.3 / reference 0.2.0.
Does not estimate contacts, smooth poses, reset XY, or correct camera drift.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import warnings
import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_bvh(path, transforms, names, parents, fps=30.0):
    """All joints carry absolute local translations; zero OFFSET avoids double offsets.

    Six channels preserve per-frame identity/scale variation. The official reader
    composes X then Y then Z rotations, so SciPy *intrinsic* XYZ is intentional.
    """
    frames, joints = transforms.shape[:2]
    if transforms.shape[2:] != (4, 4) or joints != 78 or names[0] != "Root":
        raise ValueError("Expected complete 78-joint public SOMA world transforms")
    parents = np.asarray(parents, dtype=int).copy()
    parents[0] = -1
    children = {i: [] for i in range(joints)}
    for i in range(1, joints):
        if parents[i] < 0 or parents[i] >= i:
            raise ValueError("SOMA public hierarchy is not parent-first")
        children[int(parents[i])].append(i)
    local = np.empty_like(transforms)
    local[:, 0] = transforms[:, 0]
    for j in range(1, joints):
        local[:, j] = np.linalg.solve(transforms[:, parents[j]], transforms[:, j])
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        euler = Rotation.from_matrix(local[..., :3, :3].reshape(-1, 3, 3)).as_euler("XYZ", degrees=True).reshape(frames, joints, 3)
    rows, traversal = ["HIERARCHY"], []
    def emit(j, depth):
        pad = "  " * depth
        traversal.append(j)
        rows.extend([f"{pad}{'ROOT' if j == 0 else 'JOINT'} {names[j]}", pad + "{",
                     pad + "  OFFSET 0 0 0", pad + "  CHANNELS 6 Xposition Yposition Zposition Xrotation Yrotation Zrotation"])
        for child in children[j]:
            emit(child, depth + 1)
        if not children[j]:
            rows.extend([pad + "  End Site", pad + "  {", pad + "    OFFSET 0 0 0", pad + "  }"])
        rows.append(pad + "}")
    emit(0, 0)
    rows.extend(["MOTION", f"Frames: {frames}", f"Frame Time: {1 / fps:.12f}"])
    channels = np.concatenate([local[..., :3, 3][:, traversal] * 100.0, euler[:, traversal]], axis=-1).reshape(frames, -1)
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(rows) + "\n")
        np.savetxt(stream, channels, fmt="%.10g")
    return {"bvh_joint_traversal": [names[j] for j in traversal], "bvh_rotation_order": "intrinsic XYZ",
            "bvh_translation_units": "centimetres", "animated_channels_per_joint": 6,
            "euler_gimbal_warning_count": len(seen), "constant_offsets": "zero; complete local translation in each frame"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--conditions", type=Path, required=True, help="Actual GEM conditions.pt containing full timeline")
    p.add_argument("--gem-root", type=Path, required=True)
    p.add_argument("--soma-assets", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--chunk-size", type=int, default=32)
    p.add_argument("--y-offset-m", type=float, default=0.0, help="Explicit whole-sequence vertical translation; default preserves raw world")
    p.add_argument("--retarget-python", type=Path, help="If supplied, run full official BVH->G1 and round-trip verification")
    p.add_argument("--runtime-xml", type=Path)
    p.add_argument("--retargeter-root", type=Path, help="External pinned NVIDIA soma-retargeter checkout")
    p.add_argument("--retargeter-deps", type=Path, help="Optional isolated dependency directory for the retarget interpreter")
    p.add_argument("--input-label", default="actual_GEM_prediction", choices=["actual_GEM_prediction", "synthetic_geometry_probe"])
    args = p.parse_args()
    if args.chunk_size < 1 or not np.isfinite(args.y_offset_m):
        raise ValueError("Invalid chunk size or vertical offset")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    receipt = {"schema": "gem-native77-to-soma-bvh/1", "status": "started", "input_label": args.input_label,
               "predictions_sha256": sha(args.predictions), "conditions_sha256": sha(args.conditions),
               "device": args.device, "geometry": "SOMA-X0.3.3/reference0.2.0/native identity128/GS1/local56+neutral4",
               "explicit_vertical_offset_m": args.y_offset_m, "postprocessing": "none except explicit constant vertical offset",
               "scope": "parameter/FK/serialization; not dynamic tracking or measured court trajectory"}
    try:
        sys.path.insert(0, str(args.gem_root.resolve()))
        import torch
        import soma
        from gem.common.soma import SomaFixedGs1BodyAdapter, SomaRuntimeConfig
        if soma.__version__ != "0.3.3":
            raise ValueError(f"Requires SOMA 0.3.3, got {soma.__version__}")
        raw = torch.load(args.predictions, map_location="cpu", weights_only=False)
        conditions = torch.load(args.conditions, map_location="cpu", weights_only=False)
        params = raw["pred_soma_params_global"]
        root = params["global_orient"]
        if root.ndim != 3 or root.shape[0] != 1 or root.shape[-1] != 3:
            raise ValueError("Expected one complete clip, [1,T,3] root axis-angle")
        length = root.shape[1]
        times = torch.as_tensor(conditions["frame_times"]).reshape(-1).numpy()
        if float(conditions["sampling_fps"]) != 30 or len(times) != length or not np.allclose(times, np.arange(length) / 30, rtol=0, atol=1e-7):
            raise ValueError("Prediction count and actual complete zero-origin 30Hz timeline must match")
        if "length" in conditions and int(torch.as_tensor(conditions["length"]).reshape(-1)[0]) != length:
            raise ValueError("Condition length disagrees with predictions")
        body = SomaFixedGs1BodyAdapter(SomaRuntimeConfig(data_root=args.soma_assets, device=args.device,
            lod="xlo", apply_correctives=False, enable_procedural_transforms=True, procedural_reference="runtime"))
        chunks = []
        for start in range(0, length, args.chunk_size):
            stop = min(start + args.chunk_size, length)
            values = {}
            for name in ("body_pose", "global_orient", "transl", "identity_coeffs", "scale_params"):
                value = params[name]
                if name in ("identity_coeffs", "scale_params") and value.ndim == 2:
                    value = value[:, None].expand(-1, length, -1)
                if value.shape[:2] != (1, length):
                    raise ValueError(f"Unexpected {name} sequence shape {value.shape}")
                values[name] = value[:, start:stop].reshape(stop-start, *value.shape[2:]).to(device=args.device, dtype=torch.float32)
                if not torch.isfinite(values[name]).all():
                    raise ValueError(f"Nonfinite {name}")
            with torch.inference_mode():
                output = body(**values, pose2rot=True, apply_correctives=False, return_joints_only=True, repose_to_bind_pose=True)
            if not torch.allclose(output["joints"], output["transforms"][:, 1:, :3, 3], rtol=0, atol=1e-6):
                raise ValueError("SOMA joints and transforms disagree")
            chunks.append(output["transforms"].detach().cpu().numpy())
        transforms = np.concatenate(chunks)
        names = list(body.runtime.public_joint_names)
        parents = body.runtime.output_joint_parent_ids.detach().cpu().numpy()
        if transforms.shape != (length, 78, 4, 4) or not np.isfinite(transforms).all():
            raise ValueError("Invalid public FK output")
        raw_transforms = transforms.copy()
        transforms[..., 1, 3] += args.y_offset_m
        bvh = args.output / "soma-world-30hz.bvh"
        serialization = write_bvh(bvh, transforms, names, parents)
        if args.y_offset_m != 0.0:
            raw_bvh = args.output / "raw-soma-world-30hz.bvh"
            write_bvh(raw_bvh, raw_transforms, names, parents)
            receipt["raw_bvh"] = {"path": raw_bvh.name, "sha256": sha(raw_bvh)}
        np.savez_compressed(args.output / "expected-world-fk.npz", transforms=transforms,
                            raw_transforms=raw_transforms, joint_names=np.asarray(names), parents=parents, timestamps=times)
        feet = [names.index(n) for n in ("LeftFoot", "LeftToeBase", "LeftToeEnd", "RightFoot", "RightToeBase", "RightToeEnd")]
        y_min = float(raw_transforms[:, feet, 1, 3].min())
        matrix = [[1,0,0],[0,0,-1],[0,1,0]]
        receipt.update(status="bvh_written", frames=length, fps=30.0, all_frames_preserved=True,
                       soma_version=soma.__version__, joint_count=78, controlled_joint_count=77,
                       public_joint_names=names, public_parent_indices=parents.tolist(),
                       world_axes="SOMA +Y up; global yaw/origin are learned rollout gauge",
                       retargeter_source_facing="Mujoco", retargeter_axis_matrix=matrix,
                       axis_mapping="(x,y,z) SOMA metres -> (x,-z,y) Newton metres via Rx(+90 degrees)",
                       raw_first_hips_m=raw_transforms[0,names.index("Hips"),:3,3].tolist(),
                       raw_lowest_foot_joint_y_m=y_min, candidate_joint_based_vertical_offset_m=-y_min,
                       candidate_offset_is_ground_truth=False, bvh_sha256=sha(bvh), **serialization)
        if args.retarget_python:
            if not args.runtime_xml or not args.retargeter_root:
                raise ValueError("--runtime-xml and --retargeter-root required for retargeting")
            retarget_args = ["--retargeter-root", str(args.retargeter_root)]
            if args.retargeter_deps:
                retarget_args += ["--retargeter-deps", str(args.retargeter_deps)]
            subprocess.run([str(args.retarget_python), str(HERE / "verify_soma_bvh.py"), "--bvh", str(bvh),
                            "--expected", str(args.output / "expected-world-fk.npz"), "--output", str(args.output / "bvh-roundtrip.json"), *retarget_args], check=True)
            subprocess.run([str(args.retarget_python), str(HERE / "retarget_bvh_probe.py"), "--input", str(bvh),
                            "--out-dir", str(args.output / "g1"), "--runtime-xml", str(args.runtime_xml),
                            "--max-frames", "0", "--device", "cpu", "--source-facing", "Mujoco", *retarget_args], check=True)
            receipt["retarget_receipt"] = "g1/probe-receipt.json"
        receipt["status"] = "passed"
    except Exception as exc:
        receipt.update(status="failed", exception_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        receipt["elapsed_seconds"] = time.time() - started
        (args.output / "conversion-receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
