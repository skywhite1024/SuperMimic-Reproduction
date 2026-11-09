"""Synthetic contract/geometry regressions, NOT evidence of real motion quality.

Run: python -m unittest discover -s experiments/sports-video -p test_reference_tools.py
"""

import copy
import json
from pathlib import Path
import pickle
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import reference_tools as rt


def fixture():
    times = np.array([0, .08, .18, .3, .4, .52, .61, .71, .8])
    motion = np.zeros((len(times), 36))
    motion[:, 0] = .4 * times
    motion[:, 1] = .2 * times
    motion[:, 2] = 1 + .05 * np.sin(times)
    motion[:, 5] = np.sin(times / 2)
    motion[:, 6] = np.cos(times / 2)
    motion[:, 7:] = times[:, None] * .1
    contract = {
        "schema_version": "sports-reference/1.0", "representation": "robot_adapted",
        "coordinate_frame": "z_up_right_handed", "world_frame_id": "synthetic-court-test-only",
        "position_unit": "m", "joint_unit": "rad", "quaternion_order": "xyzw",
        "joint_names": rt.G1_JOINT_NAMES.copy(),
        "time": {"timestamps_s": times.tolist(), "provenance": "unknown"},
        "adaptation": {"root_xy_scale": 1., "time_scale": 1.},
        "joint_velocity_limits_rad_s": [1.] * 29, "joint_limits_rad": [[-3., 3.]] * 29,
    }
    anchors = {"schema_version": "sports-anchors/1.0", "world_frame_id": contract["world_frame_id"],
               "representation": contract["representation"], "calibrations": {}, "anchors": []}
    for idx, role in [(0, "fit"), (3, "fit"), (8, "fit"), (5, "holdout")]:
        offset = np.array([.1, .04])
        anchors["anchors"].append({"anchor_id": f"synthetic-{idx}", "frame_index": idx, "foot": "left",
            "contact_point": "sole", "ground_contact": True, "confidence": .95,
            "observed_metric_xy": (motion[idx, :2] + offset + [.4, -.2]).tolist(),
            "reference_foot_offset_xy": offset.tolist(), "role": role,
            "source_evidence": "synthetic-unit-test-only; not a real video anchor"})
    return motion, times, contract, anchors


class ReferenceContracts(unittest.TestCase):
    def test_actual_irregular_timing_drives_velocity(self):
        motion, times, contract, _ = fixture()
        report = rt.audit_reference(motion, times, contract)
        self.assertEqual(report["status"], "passed")
        self.assertAlmostEqual(report["joint_speed_max_rad_s"][rt.G1_JOINT_NAMES[0]], .1)
        self.assertFalse(report["timing_verified_by_contract"])
        self.assertFalse(report["motion_quality_validated"])
        motion[3, 7] += 1
        self.assertIn("joint_velocity_limit_exceeded", rt.audit_reference(motion, times, contract)["issues"])

    def test_nonfinite_and_quaternion_are_rejected_without_normalizing(self):
        motion, times, contract, _ = fixture()
        motion[1, 6] *= 2
        self.assertIn("quaternion_norm_error_exceeds_1e-4", rt.audit_reference(motion, times, contract)["issues"])
        motion[1, 2] = np.nan
        self.assertIn("nonfinite_motion", rt.audit_reference(motion, times, contract)["issues"])

    def test_fps_label_is_not_resampling_and_joint_order_is_explicit(self):
        motion, times, contract, _ = fixture()
        contract["time"]["motion_fps"] = 30
        with self.assertRaisesRegex(rt.ContractError, "conflicts"):
            rt.load_timestamps(contract, len(motion))
        del contract["time"]["motion_fps"]
        contract["joint_names"][0], contract["joint_names"][1] = contract["joint_names"][1], contract["joint_names"][0]
        with self.assertRaisesRegex(rt.ContractError, "SDK order"):
            rt.validate_contract(contract)

    def test_ground_only_homography_composes_separate_metric_transform(self):
        anchor = {"ground_contact": True, "contact_point": "sole", "source_evidence": "synthetic geometry",
                  "observed_pixel_xy": [100, 200], "calibration_id": "shot0"}
        calibrations = {"shot0": {"verified": True, "ground_plane_z_m": 0, "line_semantics": "synthetic rectangle",
                                  "evidence": "synthetic-only", "H_image_to_canvas": [[2, 0, 25], [0, 3, 150], [0, 0, 1]],
                                  "A_canvas_to_metric": [[.01, 0, -.25], [0, .02, -3], [0, 0, 1]]}}
        np.testing.assert_allclose(rt.project_ground_anchor(anchor, calibrations), [2, 12])
        calibrations["shot0"]["H_image_to_canvas"] = (1e-5 * np.asarray(calibrations["shot0"]["H_image_to_canvas"])).tolist()
        np.testing.assert_allclose(rt.project_ground_anchor(anchor, calibrations), [2, 12])
        for point in ["pelvis", "ankle", "racket"]:
            with self.assertRaises(rt.ContractError):
                rt.project_ground_anchor({**anchor, "contact_point": point}, calibrations)
        with self.assertRaises(rt.ContractError):
            rt.project_ground_anchor({**anchor, "ground_contact": False}, calibrations)
        del calibrations["shot0"]["A_canvas_to_metric"]
        with self.assertRaises(rt.ContractError):
            rt.project_ground_anchor(anchor, calibrations)

    def test_only_xy_changes_and_holdout_never_enters_fit(self):
        motion, times, contract, anchors = fixture()
        before = motion.copy()
        repaired = rt.repair_root_xy(motion, times, contract, anchors)
        self.assertEqual(repaired.report["status"], "passed")
        np.testing.assert_array_equal(motion, before)
        np.testing.assert_array_equal(repaired.motion[:, 2:], before[:, 2:])
        np.testing.assert_array_equal(times, contract["time"]["timestamps_s"])
        self.assertLess(repaired.report["holdout_error"]["after_rmse_m"], .002)
        anchors["anchors"][-1]["observed_metric_xy"] = [100., -50.]
        repeated = rt.repair_root_xy(motion, times, contract, anchors)
        np.testing.assert_array_equal(repaired.motion, repeated.motion)
        self.assertGreater(repeated.report["holdout_error"]["after_rmse_m"], 90.)
        self.assertEqual(repaired.report["stance_velocity_pairs"], 0)

    def test_no_anchors_and_low_confidence_do_not_produce_a_reference(self):
        motion, times, contract, anchors = fixture()
        for anchor in anchors["anchors"]:
            anchor["confidence"] = .2
        result = rt.repair_root_xy(motion, times, contract, anchors)
        self.assertIsNone(result.motion)
        self.assertEqual(len(result.report["rejected_anchors"]), 4)
        self.assertEqual(result.report["status"], "rejected")

    def test_human_metric_cannot_silently_become_robot_adapted(self):
        motion, times, contract, anchors = fixture()
        anchors["representation"] = "human_metric"
        self.assertIsNone(rt.repair_root_xy(motion, times, contract, anchors).motion)
        anchors["coordinate_transform"] = {"from_world_frame_id": anchors["world_frame_id"],
            "to_world_frame_id": contract["world_frame_id"], "from_representation": "human_metric",
            "to_representation": "robot_adapted", "matrix": np.eye(3).tolist(), "evidence": "explicit synthetic identity mapping"}
        result = rt.repair_root_xy(motion, times, contract, anchors)
        self.assertIsNotNone(result.motion)
        self.assertEqual(len(result.report["transformations"]), 1)

    def test_long_gaps_are_unchanged_and_duplicate_holdout_is_rejected(self):
        motion, times, contract, anchors = fixture()
        anchors["anchors"] = [anchors["anchors"][0], anchors["anchors"][1]]
        duplicate = copy.deepcopy(anchors["anchors"][0])
        duplicate.update(anchor_id="duplicate-as-holdout", role="holdout")
        anchors["anchors"].append(duplicate)
        result = rt.repair_root_xy(motion, times, contract, anchors)
        np.testing.assert_array_equal(result.motion[4:], motion[4:])
        self.assertEqual(result.report["valid_holdout_count"], 0)
        self.assertIn("holdout_overlaps", result.report["rejected_anchors"][0]["reason"])
        self.assertIsNone(rt.repair_root_xy(motion, times, contract, anchors, rt.RepairConfig(max_anchor_gap_s=.1)).motion)

    def test_safe_gmr_npz_and_explicit_pickle_trust(self):
        motion, times, contract, _ = fixture()
        data = {"root_pos": motion[:, :3], "root_rot": motion[:, 3:7], "dof_pos": motion[:, 7:]}
        with tempfile.TemporaryDirectory() as directory:
            npz, pkl = Path(directory) / "gmr.npz", Path(directory) / "gmr.pkl"
            np.savez(npz, **data)
            pkl.write_bytes(pickle.dumps(data))
            np.testing.assert_array_equal(rt.load_motion(npz, contract), motion)
            with self.assertRaisesRegex(rt.ContractError, "trust"):
                rt.load_motion(pkl, contract)
            np.testing.assert_array_equal(rt.load_motion(pkl, contract, allow_trusted_pickle=True), motion)
            np.savez(npz, **data, fps=30)
            with self.assertRaisesRegex(rt.ContractError, "fps conflicts"):
                rt.load_motion(npz, contract)
            np.savez(npz, **data, quaternion_order="wxyz")
            with self.assertRaisesRegex(rt.ContractError, "quaternion_order"):
                rt.load_motion(npz, contract)

    def test_probe_uses_frame_pts_not_nominal_rate(self):
        data = {"streams": [{"avg_frame_rate": "60/1", "time_base": "1/60000"}],
                "frames": [{"pts_time": "3.0"}, {"pts_time": "3.02"}, {"pts_time": "3.05"}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic-not-video.bin"
            path.write_bytes(b"mock_ffprobe_input")
            with patch.object(subprocess, "run", return_value=subprocess.CompletedProcess([], 0, json.dumps(data), "")):
                report = rt.probe_video(path)
            self.assertEqual(report["timestamps_s"], [3., 3.02, 3.05])
            self.assertAlmostEqual(report["effective_fps"], 40.)
            self.assertEqual(report["nominal_avg_fps"], 60.)
            self.assertFalse(report["resampled"])

    def test_cli_export_hashes_roundtrip_and_rejection_ledger(self):
        motion, times, contract, anchors = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savetxt(root / "raw.csv", motion, delimiter=",", fmt="%.17g")
            (root / "contract.json").write_text(json.dumps(contract), encoding="utf-8")
            (root / "anchors.json").write_text(json.dumps(anchors), encoding="utf-8")
            args = ["repair-xy", str(root / "raw.csv"), "--contract", str(root / "contract.json"),
                    "--anchors", str(root / "anchors.json"), "--output-dir", str(root / "out")]
            self.assertEqual(rt.main(args), 0)
            saved = np.loadtxt(root / "out/repaired.csv", delimiter=",")
            np.testing.assert_array_equal(saved[:, 2:], motion[:, 2:])
            report = rt.read_json(root / "out/repair-report.json")
            self.assertEqual(report["output_files"]["motion_sha256"], rt.sha256_file(root / "out/repaired.csv"))
            self.assertFalse(report["motion_quality_validated"])
            self.assertEqual(rt.main(args), 2)
            anchors["anchors"] = []
            (root / "anchors.json").write_text(json.dumps(anchors), encoding="utf-8")
            args[-1] = str(root / "rejected")
            self.assertEqual(rt.main(args), 2)
            self.assertTrue((root / "rejected/repair-report.json").exists())
            self.assertFalse((root / "rejected/repaired.csv").exists())
            raw_hash = rt.sha256_file(root / "raw.csv")
            self.assertEqual(rt.main(["audit", str(root / "raw.csv"), "--contract", str(root / "contract.json"),
                                      "--output", str(root / "raw.csv"), "--overwrite"]), 2)
            self.assertEqual(rt.sha256_file(root / "raw.csv"), raw_hash)


if __name__ == "__main__":
    unittest.main()
