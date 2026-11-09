"""Offline bookkeeping tests only: no MuJoCo, CUDA, checkpoint or training."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from evaluate_batch import PrefixLedger, safe_frames, compare_prefix


class TestPrefixAccounting(unittest.TestCase):
    def test_failure_freezes_return_and_keeps_failing_reward(self):
        ledger = PrefixLedger([4, 2, 3])
        ledger.advance([1., 2., 3.], [True, False, False])
        ledger.advance([999., 2., 3.], [False, False, False])
        ledger.advance([999., 999., 3.], [False, False, False])
        self.assertEqual(ledger.steps.tolist(), [1, 2, 3])
        self.assertEqual(ledger.returns.tolist(), [1., 4., 9.])
        self.assertEqual(ledger.status.tolist(), ["native_hard_failure", "completed", "completed"])
        self.assertFalse(ledger.alive.any())

    def test_last_step_hard_has_failure_precedence(self):
        ledger = PrefixLedger([1])
        ledger.advance([.05], [True])
        self.assertEqual(ledger.status[0], "native_hard_failure")
        self.assertEqual(ledger.steps[0], ledger.planned[0])

    def test_deadline_does_not_overwrite_terminal_outcomes(self):
        ledger = PrefixLedger([4, 4])
        ledger.advance([.1, .2], [True, False])
        ledger.censor("truncated_engineering_deadline")
        self.assertEqual(ledger.status.tolist(), ["native_hard_failure", "truncated_engineering_deadline"])
        self.assertEqual(ledger.steps.tolist(), [1, 1])

    def test_safe_inactive_clock_never_leaves_own_clip(self):
        starts, ends = np.array([0, 6, 11]), np.array([5, 10, 19])
        frames = safe_frames(starts, ends, 7, np.array([False, False, True]))
        self.assertEqual(frames.tolist(), [4, 9, 18])
        inactive = np.array([True, True, False])
        self.assertTrue(np.all(frames[inactive]+1 <= ends[inactive]))

    def test_nonfinite_dead_rewards_cannot_corrupt_scores(self):
        ledger = PrefixLedger([3, 3])
        ledger.advance([1., 1.], [True, False])
        ledger.advance([np.nan, 1.], [False, False])
        self.assertEqual(ledger.returns.tolist(), [1., 2.])

    def test_prefix_comparison_detects_numerical_and_gate_difference(self):
        data = {"state": np.zeros((6, 71)), "reference": np.zeros((6, 41)),
                "action_raw": np.zeros((5, 29)), "reward": np.zeros(5),
                "native_hard": np.zeros(5, dtype=bool), "native_done": np.zeros(5, dtype=bool)}
        with tempfile.TemporaryDirectory() as folder:
            a, b = Path(folder)/"a.npz", Path(folder)/"b.npz"
            np.savez(a, **data); np.savez(b, **data)
            result = compare_prefix(a, b, 5)
            self.assertTrue(result["passed"])
            self.assertFalse(result["full_termination_equivalence_verified"])
            changed = {**data, "action_raw": np.ones((5, 29))}
            np.savez(b, **changed)
            self.assertFalse(compare_prefix(a, b, 5)["passed"])
            changed = {**data, "native_hard": np.array([False, True, False, False, False])}
            np.savez(b, **changed)
            self.assertFalse(compare_prefix(a, b, 5)["passed"])


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestPrefixAccounting)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    receipt = {"status": "pass" if result.wasSuccessful() else "fail", "tests_run": result.testsRun,
               "failures": len(result.failures), "errors": len(result.errors),
               "scope": "offline mock accounting only", "cuda_executed": False,
               "mujoco_executed": False, "native_batch_equivalence_verified": False}
    Path(__file__).with_name("mock-receipt.json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
    raise SystemExit(0 if result.wasSuccessful() else 1)
