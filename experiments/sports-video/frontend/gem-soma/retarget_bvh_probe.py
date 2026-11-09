"""Finite native SOMA BVH -> official G1 retargeter -> SuperMimic format probe.

No model training, no SMPL asset, no dynamic rollout; CPU is the default.
All default parser parameters except three explicit switches remain unchanged.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path



def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--runtime-xml", type=Path, required=True)
    p.add_argument("--max-frames", type=int, default=8,
                   help="30Hz output samples; 0 processes the full source duration")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda:0"])
    p.add_argument("--source-facing", choices=["Mujoco", "Maya"], default="Mujoco")
    p.add_argument("--retargeter-root", type=Path, required=True)
    p.add_argument("--retargeter-deps", type=Path)
    args = p.parse_args()
    sys.path.insert(0, str(args.retargeter_root.resolve()))
    if args.retargeter_deps:
        sys.path.insert(0, str(args.retargeter_deps.resolve()))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    receipt = {"schema": "soma-retarget-probe/1", "status": "started", "device": args.device,
               "input": str(args.input.resolve()), "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
               "max_frames": args.max_frames, "scope": "public BVH kinematic engineering; not sports success"}
    try:
        import numpy as np
        import warp as wp
        import newton
        from soma_retargeter.io.bvh import load_bvh
        from soma_retargeter.io.csv import save_csv, load_csv
        from soma_retargeter.animation.animation_buffer import AnimationBuffer
        from soma_retargeter.pipelines.soma_retargeting_pipeline import SomaRetargetingPipeline
        from soma_retargeter.pipelines.utils import get_retargeter_config
        from soma_retargeter.utils.newton_utils import get_filtered_joint_names
        from soma_retargeter.utils.space_conversion_utils import SpaceConverter, get_facing_direction_type_from_str
        from convert_soma_csv import convert
        wp.init()
        wp.set_device(args.device)
        skeleton, source = load_bvh(str(args.input))
        total = int(np.floor((source.num_frames - 1) / source.sample_rate * 30 + 1e-8)) + 1
        frames = total if args.max_frames == 0 else min(total, args.max_frames)
        if frames < 1:
            raise ValueError("No frames to process")
        local_transforms = np.stack([source.sample(i / 30) for i in range(frames)])
        animation = AnimationBuffer(skeleton, frames, 30.0, local_transforms)
        config = get_retargeter_config("soma", "unitree_g1")
        config.update(enable_post_processing=False, enable_contact_processing=False, smooth_joint_filter_weight=0.0)
        (args.out_dir / "retarget-config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        pipeline = SomaRetargetingPipeline(skeleton, "soma", "unitree_g1", retarget_config=config)
        converter = SpaceConverter(get_facing_direction_type_from_str(args.source_facing))
        pipeline.add_input_motions([animation], [converter.transform(wp.transform_identity())], True)
        buffers = pipeline.execute()
        names = get_filtered_joint_names(pipeline.robot_builder, [newton.JointType.REVOLUTE])
        path = args.out_dir / "official-g1.csv"
        save_csv(str(path), names, buffers[0])
        official_loaded = load_csv(str(path), fps=30.0)
        official_data = np.stack([official_loaded.get_data(i) for i in range(official_loaded.num_frames)])
        converted, conversion = convert(path, args.out_dir / "supermimic-30hz.csv", args.runtime_xml, 30.0)
        reordering = [names.index(n) for n in conversion["joint_names"]]
        expected = np.concatenate([official_data[:, :7], official_data[:, 7:][:, reordering]], axis=1)
        # Quaternion sign is immaterial. Compare rotation matrices independently.
        from scipy.spatial.transform import Rotation
        angle_error = (Rotation.from_quat(expected[:, 3:7]).inv() * Rotation.from_quat(converted[:, 3:7])).magnitude()
        scalar_error = float(np.max(np.abs(expected[:, [0,1,2,*range(7,36)]] - converted[:, [0,1,2,*range(7,36)]])))
        assert scalar_error < 2e-6 and float(angle_error.max()) < 2e-6
        assert len(names) == 29 and converted.shape == (frames, 36) and np.isfinite(converted).all()
        receipt.update(status="passed", frames=frames, fps=30.0, source_frames=source.num_frames,
                       source_fps=source.sample_rate, truncated=frames < total,
                       source_facing=args.source_facing, skeleton_joint_count=skeleton.num_joints,
                       output_joint_order=names, runtime_joint_order=conversion["joint_names"],
                       runtime_joint_permutation=reordering, official_loader_scalar_error=scalar_error,
                       official_loader_quaternion_angle_error_rad=float(angle_error.max()),
                       runtime_limit_eligible=conversion["runtime_limit_eligible"],
                       runtime_limit_violations=conversion["runtime_joint_limit_violations"],
                       config_overrides={"enable_post_processing":False,"enable_contact_processing":False,"smooth_joint_filter_weight":0.0},
                       retained_parser_operations=["human-to-robot scaling", "24-iteration position/orientation IK", "joint-limit objective", "final joint-limit clamp", "15 warmup frames removed"],
                       numpy_version=np.__version__, newton_version=newton.__version__, warp_version=wp.__version__)
    except Exception as exc:
        receipt.update(status="failed", exception_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        receipt["elapsed_seconds"] = time.time() - started
        (args.out_dir / "probe-receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
