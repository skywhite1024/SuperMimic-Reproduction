"""Read all v2 trajectory arrays and independently verify saved first states.

Uses the same native clipped-reference contract as the evaluator, with NumPy
and the packed source bytes. No policy replay or motion-quality claim is made.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from audit_saved_trajectories import audit, digest


class InitialStateCheck:
    def __init__(self, motions, protocol, result, accounting, expected_clips):
        motions = Path(motions)
        if (protocol.get("manual_reset_version") != 2
                or result.get("manual_reset_version") != 2
                or result.get("initial_reference_state_verified_for_all_clips") is not True
                or protocol.get("observation_cache_refreshed_after_pose_write") is not True
                or protocol.get("actuator_histories_backfilled_after_pose_write") is not True):
            raise ValueError("A completed manual-reset-v2 result is required")
        self.names = json.loads((motions / "motion_names.json").read_text())
        joints = json.loads((motions / "joint_names.json").read_text())
        if joints != protocol["native_joint_names"] or len(joints) != 29:
            raise ValueError("Packed and native G1 joint orders must match")
        self.ends = np.load(motions / "motion_ends.npy", allow_pickle=False)
        if (len(self.names) != expected_clips or len(set(self.names)) != expected_clips
                or self.ends.shape != (expected_clips,)
                or not np.issubdtype(self.ends.dtype, np.integer)
                or not np.all(np.diff(np.r_[0, self.ends]) > 0)):
            raise ValueError("Invalid complete packed motion membership")
        self.starts = np.r_[0, self.ends[:-1]]
        self.limits = np.asarray(protocol["native_soft_joint_pos_limits"], dtype=np.float32)
        if (self.limits.shape != (29, 2) or not np.isfinite(self.limits).all()
                or not np.all(self.limits[:, 0] <= self.limits[:, 1])):
            raise ValueError("Invalid recorded native joint limits")
        fields = {"root_pos_w": 3, "root_quat_w": 4, "joint_pos": 29,
            "root_lin_vel_w": 3, "root_ang_vel_w": 3, "joint_vel": 29}
        required = {name + ".npy" for name in fields} | {
            "motion_names.json", "joint_names.json", "motion_ends.npy"}
        if set(protocol["motion_files"]) != required:
            raise ValueError("Expected the complete nine-file source pack")
        for name in sorted(required):
            if digest(motions / name) != protocol["motion_files"][name]:
                raise ValueError("Packed source hash differs: " + name)
        self.arrays = {}
        for name, width in fields.items():
            values = np.load(motions / (name + ".npy"), mmap_mode="r", allow_pickle=False)
            if values.shape != (int(self.ends[-1]), width) or values.dtype != np.float32:
                raise ValueError("Unexpected packed array shape/dtype: " + name)
            self.arrays[name] = values
        self.count, self.maximum_error = 0, 0.0
        self.evidence = {"status": "saved_initial_state_audit_incomplete",
            "protocol_sha256": accounting["protocol_sha256"],
            "manual_reset_version": 2, "packed_source_hashes_verified": True,
            "initial_state_tolerance": {"atol": 1e-5, "rtol": 1e-5},
            "saved_initial_states_verified": 0, "maximum_abs_error": 0.0,
            "all_saved_initial_states_verified": False,
            "expected_clips": expected_clips,
            "scope": "All saved71D first rows versus native clipped packed reference; no policy replay, termination replay, deterministic-physics or tracking-quality claim"}

    def __call__(self, row, first_state):
        index = row["index"]
        if (row["name"] != self.names[index]
                or row["reference_frames"] != int(self.ends[index] - self.starts[index])):
            raise ValueError("Saved motion identity differs from packed reference")
        frame = int(self.starts[index])
        pieces = [self.arrays[name][frame] for name in self.arrays]
        pieces[2] = np.clip(pieces[2], self.limits[:, 0], self.limits[:, 1])
        expected = np.concatenate(pieces)
        if (first_state.shape != (71,) or not np.isfinite(expected).all()
                or not np.allclose(first_state, expected, atol=1e-5, rtol=1e-5)):
            raise ValueError(f"Saved first state differs from intended reference at index {index}")
        self.count += 1
        self.maximum_error = max(self.maximum_error, float(np.max(np.abs(first_state - expected))))
        self.evidence.update(saved_initial_states_verified=self.count,
            maximum_abs_error=self.maximum_error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--motions", type=Path, required=True)
    parser.add_argument("--expected-clips", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    read = lambda name: json.loads((args.evaluation / name).read_text())
    result, protocol, accounting = map(read, ("result.json", "protocol.json", "accounting.json"))
    if digest(args.evaluation / "protocol.json") != accounting["protocol_sha256"]:
        raise ValueError("Protocol bytes do not match completed accounting")
    check = InitialStateCheck(args.motions, protocol, result, accounting, args.expected_clips)
    try:
        summary = audit(args.evaluation, args.output, args.expected_clips, initial_state_check=check)
        if check.count != args.expected_clips or summary["array_widths"] != {"full_state": 71, "actions": 29}:
            raise ValueError("Incomplete G1 saved-state verification")
        check.evidence.update(status="complete_saved_initial_state_audit_verified",
            all_saved_initial_states_verified=True,
            trajectory_integrity_summary_sha256=digest(args.output / "summary.json"))
    except Exception as error:
        check.evidence.update(status="saved_initial_state_audit_failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if args.output.is_dir():
            (args.output / "initial-states.json").write_text(
                json.dumps(check.evidence, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("SAVED_INITIAL_STATES_V2_VERIFIED " + json.dumps(check.evidence), flush=True)


if __name__ == "__main__":
    main()
