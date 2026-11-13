"""Reject invalid numerical evidence before accepting sweep measurements."""

import copy

import pytest

from run_benchmark_sweep import validate_measurement


def fixture_record():
    path = {"seconds_per_repeat": [2., 4., 8.], "steps_per_second": [16000., 8000., 4000.]}
    return {"status": "benchmark_case_completed", "purpose": "experiment", "num_envs": 64,
            "warmup_calls_per_path": 20, "timed_calls_per_repeat": 500, "repeats": 3,
            "paper_speedup_reproduced": False, "policy_quality_evaluated": False,
            "differentiation_speed_or_full_control_speedup_claimed": False,
            "tf32_matmul_allowed": True, "decimation": 4, "control_dt": .02, "physics_dt": .005,
            "simulator_with_policy": {**path, "finite_pre_and_post": True},
            "world_model_modes": [{**copy.deepcopy(path), "inference_mode": False, "autograd_enabled": True},
                                  {**copy.deepcopy(path), "inference_mode": True, "autograd_enabled": False}],
            "simulator_median_sps": 8000., "diagnostic_wm_to_sim_scope_ratios": {"False": 1., "True": 1.}}


def test_recompute_rates_and_ratios_instead_of_trusting_worker_claim():
    record = fixture_record()
    validate_measurement(record, size=64, purpose="experiment")
    record["simulator_with_policy"]["steps_per_second"][0] = 32000.
    with pytest.raises(ValueError, match="recompute"):
        validate_measurement(record, size=64, purpose="experiment")
    record = fixture_record()
    record["diagnostic_wm_to_sim_scope_ratios"]["False"] = 64.
    with pytest.raises(ValueError, match="ratio"):
        validate_measurement(record, size=64, purpose="experiment")


@pytest.mark.parametrize("invalid", [0., float("nan"), float("inf")])
def test_failed_size_cannot_appear_as_zero_or_nonfinite_measurement(invalid):
    record = fixture_record()
    record["world_model_modes"][0]["seconds_per_repeat"][0] = invalid
    with pytest.raises(ValueError, match="Nonpositive/nonfinite"):
        validate_measurement(record, size=64, purpose="experiment")


def test_missing_repeat_and_short_integration_cannot_pass_full_measurement():
    record = fixture_record()
    record["world_model_modes"][1]["seconds_per_repeat"].pop()
    with pytest.raises(ValueError, match="Missing timing repeat"):
        validate_measurement(record, size=64, purpose="experiment")
    record = fixture_record()
    record.update(purpose="integration", warmup_calls_per_path=2, timed_calls_per_repeat=2, repeats=1)
    with pytest.raises(ValueError, match="wrong-purpose"):
        validate_measurement(record, size=64, purpose="experiment")
