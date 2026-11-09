"""Native physics checks for evaluation denominators, failure retention and budgets."""
import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from test_sports_runtime import tiny_parent, write_standing
from sports_runtime import migrate_checkpoint, sha
from evaluate import aggregate, evaluate_clip, prepare_checkpoint


class SportsEvaluationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.saved = migrate_checkpoint(tiny_parent(), "F0")

    def tearDown(self):
        self.temp.cleanup()

    def evaluate(self, source, name, max_steps=None):
        row = {"path": source, "fps": 30., "group": "new", "sha256": sha(source)}
        return evaluate_clip(row, self.saved, device="cpu", feedback="F0", seed=0,
                             action_mode="raw", output=self.root/name, max_steps=max_steps)

    def test_engineering_prefix_uses_full_denominator_and_is_never_completion(self):
        source = self.root/"standing.csv"
        write_standing(source, 61)
        result = self.evaluate(source, "truncated", max_steps=3)
        self.assertEqual(result["status"], "truncated_engineering_steps")
        self.assertEqual(result["planned_steps"], 100)
        self.assertEqual(result["observed_steps"], 3)
        self.assertFalse(result["completed_without_native_failure"])
        self.assertTrue(result["tracking_errors_censored"])
        self.assertAlmostEqual(result["norm_return_full_reference"], result["reward_sum_until_stop"]/10.)
        trace = np.load(self.root/"truncated/trajectory.npz")
        self.assertEqual(trace["state"].shape, (4, 71))
        self.assertEqual(trace["action_applied"].shape, (3, 29))
        self.assertEqual(aggregate([result])["budget_truncated_or_unrun"], 1)

    def test_native_failure_retains_prefix_and_censors_missing_future(self):
        source = self.root/"untrackable.csv"
        write_standing(source, 61)
        # An explicit infeasible reference provides a reliable native height
        # failure without changing any threshold, state after t0 or controller.
        with source.open(encoding="utf-8") as stream:
            rows = list(csv.reader(stream))
        for row in rows[1:]:
            row[2] = "5.0"
        with source.open("w", newline="") as stream:
            csv.writer(stream).writerows(rows)
        result = self.evaluate(source, "failed")
        self.assertEqual(result["status"], "native_hard_failure")
        self.assertTrue(result["native_hard_failure"])
        self.assertFalse(result["completed_without_native_failure"])
        self.assertLess(result["observed_steps"], result["planned_steps"])
        self.assertEqual(result["first_native_hard_failure_s"], result["observed_steps"]*.02)
        self.assertTrue(result["tracking_errors_censored"])
        trace = np.load(self.root/"failed/trajectory.npz")
        self.assertEqual(len(trace["state"]), result["observed_steps"]+1)
        self.assertTrue(trace["native_hard"][-1])
        self.assertAlmostEqual(result["norm_return_full_reference"], trace["reward"].sum()/10.)
        self.assertEqual(result["post_initialization_resets"], 0)

    def test_short_full_rollout_and_feedback_mismatch_rejection(self):
        source = self.root/"short.csv"
        write_standing(source, 7)
        result = self.evaluate(source, "full")
        self.assertEqual(result["planned_steps"], 10)
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["full_reference_completed"])
        self.assertTrue(result["completed_without_native_failure"])
        self.assertFalse(result["tracking_errors_censored"])
        self.assertIsNone(result["strict_contact_quality_acceptance"])
        with self.assertRaisesRegex(ValueError, "matching sports-video-xy"):
            prepare_checkpoint(self.saved, "F1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
