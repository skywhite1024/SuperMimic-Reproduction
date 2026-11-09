"""CPU-only adapter/bookkeeping checks; never claim CUDA parity or simulate physics."""
import json
from pathlib import Path
import unittest
import torch
from diagnose_precision import Repeat40Policy, frozen_differences, short_prefix_gate


class PrecisionMockTest(unittest.TestCase):
    def test_batch40_preserves_input_values_and_original_row(self):
        history, goal = torch.arange(12.).reshape(1, 2, 6), torch.arange(4.).reshape(1, 4)
        calls = []
        def policy(h, g):
            calls.append((h.clone(), g.clone()))
            return torch.arange(h.shape[0]*29.).reshape(h.shape[0], 29)
        result = Repeat40Policy(policy, 20)(history, goal)
        self.assertEqual(result.shape, (1, 29))
        self.assertEqual(float(result[0, 0]), 20*29.)
        self.assertTrue(torch.equal(calls[0][0], history.expand(40, -1, -1)))
        self.assertTrue(torch.equal(calls[0][1], goal.expand(40, -1)))
        with self.assertRaises(ValueError):
            Repeat40Policy(policy, 20)(history.repeat(2, 1, 1), goal.repeat(2, 1))

    def test_frozen_probe_detects_injected_batch_shape_effect(self):
        def policy(h, g):
            return g[:, :1].repeat(1, 29)+(0.125 if h.shape[0] == 1 else 0.)
        h, g = torch.zeros(40, 2, 6), torch.arange(40.).reshape(40, 1)
        rows = frozen_differences(policy, [{"step": 0, "history": h, "goal": g, "J_action": policy(h, g)}], (0, 20, 39), torch)
        self.assertEqual(len(rows), 3)
        for row in rows:
            self.assertEqual(row["single_vs_actual40_max_abs"], .125)
            self.assertEqual(row["repeat40_vs_actual40_max_abs"], 0.)
            self.assertEqual(row["actual40_vs_recorded_J_action_max_abs"], 0.)

    def test_gate_cannot_pass_incomplete_or_unfaithful_reconstruction(self):
        report = {"reconstruction": {"maximum_common_recorded_actions": 100},
                  "bf16_identical_input_comparisons": [{"actual40_vs_recorded_J_action_max_abs": 0.} for _ in range(15)],
                  "rollouts": [{"new_policyrepeat40": {"comparison_with_original_J_batch40": {"passed": True}}} for _ in range(3)]}
        self.assertTrue(short_prefix_gate(report, .005))
        report["bf16_identical_input_comparisons"][0]["actual40_vs_recorded_J_action_max_abs"] = .006
        self.assertFalse(short_prefix_gate(report, .005))
        report["bf16_identical_input_comparisons"][0]["actual40_vs_recorded_J_action_max_abs"] = 0.
        report["rollouts"].pop()
        self.assertFalse(short_prefix_gate(report, .005))


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PrecisionMockTest))
    receipt = {"status": "pass" if result.wasSuccessful() else "fail", "tests": result.testsRun,
               "errors": len(result.errors), "failures": len(result.failures),
               "scope": "CPU mock verifies adapter shape/value preservation and gate bookkeeping only",
               "cuda_executed": False, "physics_executed": False, "native_equivalence_verified": False}
    Path(__file__).with_name("mock-receipt.json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
    raise SystemExit(0 if result.wasSuccessful() else 1)
