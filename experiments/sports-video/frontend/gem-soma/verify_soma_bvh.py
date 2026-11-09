"""Independent readback using official retargeter BVH loader and its FK."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bvh", type=Path, required=True)
    p.add_argument("--expected", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--retargeter-root", type=Path, required=True)
    p.add_argument("--retargeter-deps", type=Path)
    args = p.parse_args()
    sys.path.insert(0, str(args.retargeter_root.resolve()))
    if args.retargeter_deps:
        sys.path.insert(0, str(args.retargeter_deps.resolve()))
    import warp as wp
    from soma_retargeter.io.bvh import load_bvh
    from soma_retargeter.utils.space_conversion_utils import SpaceConverter, FacingDirectionType
    wp.init()
    wp.set_device("cpu")
    expected = np.load(args.expected, allow_pickle=False)
    source = expected["transforms"]
    names = list(expected["joint_names"])
    skel, anim = load_bvh(str(args.bvh))
    assert anim.sample_rate == 30.0 and anim.num_frames == len(source)
    order = [list(skel.joint_names).index(n) for n in names]
    maximum_position, maximum_angle = 0.0, 0.0
    for i in range(anim.num_frames):
        actual = anim.compute_global_transforms(i)[order]
        maximum_position = max(maximum_position, float(np.max(np.linalg.norm(actual[:, :3] - source[i, :, :3, 3], axis=-1))))
        angle = (Rotation.from_matrix(source[i, :, :3, :3]).inv() * Rotation.from_quat(actual[:, 3:7])).magnitude()
        maximum_angle = max(maximum_angle, float(angle.max()))
    converter = SpaceConverter(FacingDirectionType.MUJOCO)
    matrix = np.array([[1,0,0],[0,0,-1],[0,1,0]], dtype=float)
    axes_error = max(float(np.linalg.norm(np.asarray(converter.convert_position(v)) - matrix @ v)) for v in np.eye(3))
    result = {"status": "passed", "frames": anim.num_frames, "fps": anim.sample_rate,
              "max_world_position_error_m": maximum_position, "max_world_rotation_error_rad": maximum_angle,
              "axis_basis_error": axes_error, "axis_matrix": matrix.tolist(), "scope": "full-frame FK serialization; not motion correctness"}
    if maximum_position >= 2e-5 or maximum_angle >= 2e-5 or axes_error >= 2e-6:
        result["status"] = "failed"
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))
    assert result["status"] == "passed", result

if __name__ == "__main__":
    main()
