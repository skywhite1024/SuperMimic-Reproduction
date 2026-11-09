"""Timing/geometry software tests; synthetic trajectories are not motion results."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

import reference_tools as rt
import resample_reference as rs


def fixture(count=26, fps=25):
    times = np.arange(count, dtype=float) / fps
    motion = np.zeros((count, 36))
    motion[:, :3] = times[:, None] * np.array([.8, -.2, .1]) + [0, 0, 1]
    motion[:, 3:7] = Rotation.from_rotvec(times[:, None] * np.array([0, 0, .5])).as_quat()
    motion[:, 7:] = times[:, None] * np.linspace(-.1, .1, 29)
    names = rt.G1_JOINT_NAMES.copy()
    contract = {"schema_version": "sports-reference/1.0", "representation": "robot_adapted",
                "coordinate_frame": "z_up_right_handed", "world_frame_id": "synthetic-test-only",
                "quaternion_order": "xyzw", "position_unit": "m", "joint_unit": "rad", "joint_names": names,
                "time": {"timestamps_s": times.tolist(), "motion_fps": fps, "provenance": "source_pts",
                         "source_video_sha256": "a" * 64},
                "adaptation": {"root_xy_scale": .9, "time_scale": 1., "description": "synthetic metadata propagation"},
                "joint_mapping": {"verified": False, "source_joint_names": names, "target_joint_names": names,
                                  "permutation": list(range(29)), "sign": [1] * 29, "offset_rad": [0.] * 29,
                                  "evidence": "synthetic only; real mapping still unverified"},
                "robot_xml_sha256": "b" * 64, "ik_config_sha256": "c" * 64,
                "joint_limits_rad": [[-3, 3]] * 29, "joint_velocity_limits_rad_s": [2.] * 29}
    return motion, times, contract


class ResamplingContracts(unittest.TestCase):
    def test_25_to_30_preserves_linear_trajectory_and_exact_one_second_coverage(self):
        motion, times, contract = fixture()
        original, metadata = motion.copy(), copy.deepcopy(contract)
        out, sidecar, mapping, report = rs.resample_reference(motion, times, contract)
        target = np.asarray(sidecar["time"]["timestamps_s"])
        self.assertEqual(len(out), 31)
        np.testing.assert_allclose(out[:, :3], target[:, None] * [.8, -.2, .1] + [0, 0, 1], atol=1e-14)
        np.testing.assert_allclose(out[:, 7:], target[:, None] * np.linspace(-.1, .1, 29), atol=1e-14)
        np.testing.assert_allclose(Rotation.from_quat(out[:, 3:7]).as_rotvec()[:, 2], .5 * target, atol=1e-14)
        self.assertEqual(report["coverage"]["crop_duration_s"], 0)
        self.assertTrue(report["coverage"]["source_endpoint_included"])
        self.assertEqual(contract, metadata)
        np.testing.assert_array_equal(motion, original)
        self.assertEqual(sidecar["joint_mapping"], contract["joint_mapping"])
        self.assertFalse(sidecar["joint_mapping"]["verified"])
        self.assertEqual(sidecar["adaptation"], contract["adaptation"])
        self.assertEqual(sidecar["time"]["provenance"], "resampled_source_pts")
        self.assertTrue(report["output_audit"]["legacy_motionlib_default_compatible"])
        for entry in mapping["frames"]:
            interpolated_time = (1 - entry["alpha_right"]) * entry["source_left_pts_s"] + entry["alpha_right"] * entry["source_right_pts_s"]
            self.assertAlmostEqual(interpolated_time, entry["output_pts_s"])

    def test_fractional_tail_is_cropped_not_padded_or_stretched(self):
        motion, times, contract = fixture(count=25)
        out, sidecar, mapping, report = rs.resample_reference(motion, times, contract)
        target = np.array(sidecar["time"]["timestamps_s"])
        self.assertEqual(len(out), 29)
        self.assertAlmostEqual(target[-1], 28 / 30)
        self.assertAlmostEqual(report["coverage"]["crop_duration_s"], .96 - 28 / 30)
        self.assertLess(report["coverage"]["crop_duration_s"], 1 / 30)
        self.assertFalse(report["coverage"]["source_endpoint_included"])
        self.assertFalse(report["coverage"]["padded_or_repeated_endpoint"])
        self.assertFalse(np.array_equal(out[-1, :3], motion[-1, :3]))
        np.testing.assert_allclose(np.diff(target), 1 / 30)

    def test_slerp_takes_short_path_through_180_despite_quaternion_sign_flips(self):
        motion, times, contract = fixture(count=3)
        angles = np.array([170., 180., 190.])
        motion[:, 3:7] = Rotation.from_euler("z", angles, degrees=True).as_quat()
        motion[1, 3:7] *= -1
        out, sidecar, _, _ = rs.resample_reference(motion, times, contract, target_fps=50)
        expected = Rotation.from_euler("z", 170 + 250 * np.array(sidecar["time"]["timestamps_s"]), degrees=True)
        error = (Rotation.from_quat(out[:, 3:7]).inv() * expected).magnitude()
        np.testing.assert_allclose(error, 0, atol=1e-14)
        self.assertTrue(np.all(np.sum(out[:-1, 3:7] * out[1:, 3:7], axis=1) > 0))

    def test_nonmonotonic_large_gap_unknown_units_or_timing_are_rejected(self):
        motion, times, contract = fixture()
        duplicate = times.copy()
        duplicate[3] = duplicate[2]
        with self.assertRaises(rt.ContractError):
            rs.resample_reference(motion, duplicate, contract)
        large = times.copy()
        large[10:] += .3
        altered = copy.deepcopy(contract)
        altered["time"].pop("motion_fps")
        altered["time"]["timestamps_s"] = large.tolist()
        with self.assertRaisesRegex(rt.ContractError, "source gaps"):
            rs.resample_reference(motion, large, altered)
        for field, value in [("joint_unit", "degree"), ("position_unit", "cm")]:
            altered = copy.deepcopy(contract)
            altered[field] = value
            with self.assertRaises(rt.ContractError):
                rs.resample_reference(motion, times, altered)
        for provenance in ["unknown", "export_declared_cfr"]:
            altered = copy.deepcopy(contract)
            altered["time"]["provenance"] = provenance
            with self.assertRaisesRegex(rt.ContractError, "real PTS"):
                rs.resample_reference(motion, times, altered)

    def test_vfr_mapping_uses_real_pts_and_rational_target_rate(self):
        motion, times, contract = fixture(count=7)
        times = np.array([0., .039, .081, .121, .16, .2, .241]) + 10
        contract["time"]["timestamps_s"] = times.tolist()
        del contract["time"]["motion_fps"]
        out, sidecar, mapping, report = rs.resample_reference(motion, times, contract, target_fps="30000/1001")
        self.assertEqual(mapping["target_rate_fraction"], "30000/1001")
        target = np.array(sidecar["time"]["timestamps_s"])
        self.assertLessEqual(target[-1], times[-1])
        self.assertLess(times[-1] - target[-1], 1001 / 30000)
        for row in mapping["frames"]:
            self.assertGreaterEqual(row["output_pts_s"], row["source_left_pts_s"])
            self.assertLessEqual(row["output_pts_s"], row["source_right_pts_s"])

    def test_reject_independent_P1_retiming_and_preserve_prior_history(self):
        motion, times, contract = fixture()
        contract["processing"] = {"operation": "root_xy_only"}
        with self.assertRaisesRegex(rt.ContractError, "BEFORE"):
            rs.resample_reference(motion, times, contract)
        contract["processing"] = {"operation": "GMR_export", "model_version": "synthetic-test"}
        _, sidecar, _, _ = rs.resample_reference(motion, times, contract)
        self.assertEqual(sidecar["processing_history"], [contract["processing"]])
        self.assertEqual(sidecar["processing"]["operation"], "shared_source_pts_resample")

    def test_cli_source_hashes_and_rejected_output_without_motion(self):
        motion, times, contract = fixture(count=25)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savetxt(root / "source.csv", motion, delimiter=",", fmt="%.17g")
            (root / "contract.json").write_text(json.dumps(contract), encoding="utf-8")
            args = [str(root / "source.csv"), "--contract", str(root / "contract.json"), "--output-dir", str(root / "out")]
            self.assertEqual(rs.main(args), 0)
            sidecar = rt.read_json(root / "out/resampled.contract.json")
            self.assertEqual(sidecar["parent_motion_sha256"], rt.sha256_file(root / "source.csv"))
            self.assertEqual(sidecar["motion_sha256"], rt.sha256_file(root / "out/resampled.csv"))
            self.assertEqual(sidecar["time"]["resampling_map_sha256"], rt.sha256_file(root / "out/resampling-map.json"))
            self.assertEqual(sidecar["joint_mapping"], contract["joint_mapping"])
            report = rt.read_json(root / "out/resampling-report.json")
            self.assertFalse(report["motion_quality_validated"])
            contract["time"]["provenance"] = "unknown"
            (root / "contract.json").write_text(json.dumps(contract), encoding="utf-8")
            args[-1] = str(root / "rejected")
            self.assertEqual(rs.main(args), 2)
            self.assertFalse((root / "rejected/resampled.csv").exists())
            self.assertTrue((root / "rejected/resampling-report.json").exists())


if __name__ == "__main__":
    unittest.main()
