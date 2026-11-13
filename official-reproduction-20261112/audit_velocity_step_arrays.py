"""Recompute complete local step-test evidence from every saved trial NPZ.

NumPy/statistics implementation independent of the Torch training/evaluation
kernels. All-fall trials stay undefined. No policy replay or quality claim.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inspect(folder):
    result = json.loads((folder / 'result.json').read_text())
    protocol = json.loads((folder / 'protocol.json').read_text())
    assert result['status'] == 'velocity_step_test_completed'
    assert result['protocol_sha256'] == sha(folder / 'protocol.json')
    assert result['commands'] == protocol['grid_identities'] == 672
    assert result['trials_per_command'] == protocol['trials'] == 4
    assert protocol['fps'] == 50 and protocol['seconds'] == 5
    trace = folder / result['trajectory']['file']
    assert trace.name == result['trajectory']['file']
    assert trace.stat().st_size == result['trajectory']['bytes'] and sha(trace) == result['trajectory']['sha256']
    with np.load(trace, allow_pickle=False) as archive:
        assert set(archive.files) == {'states', 'actions', 'heading_velocity', 'steps', 'fallen'}
        data = {name: archive[name] for name in archive.files}
    steps, fallen = data['steps'], data['fallen']
    assert steps.shape == fallen.shape == (4, 672)
    assert steps.dtype == np.int64 and fallen.dtype == np.bool_
    assert np.all((steps > 0) & (steps <= 250)) and np.all(fallen | (steps == 250))
    for name, shape in [('states', (4, 672, 251, 71)), ('actions', (4, 672, 250, 29)),
                         ('heading_velocity', (4, 672, 250, 3))]:
        values = data[name]
        assert values.shape == shape and values.dtype == np.float32
        valid = np.arange(shape[2])[None, None, :] < steps[..., None] + (name == 'states')
        assert np.isfinite(values[valid]).all() and np.isnan(values[~valid]).all()
    root = np.asarray(protocol['standing_root_state'], dtype=np.float64)
    initial = np.r_[root[:7], protocol['standing_joint_positions'], root[7:], np.zeros(29)]
    assert initial.shape == (71,)
    assert np.allclose(data['states'][:, :, 0], initial, atol=1e-5, rtol=0)
    maximum_projection_error, all_fall, defined, displays = 0.0, 0, [], []
    assert [row['index'] for row in result['records']] == list(range(672))
    for index, row in enumerate(result['records']):
        command = np.asarray(row['command'], dtype=np.float64)
        assert np.linalg.norm(command) > 0
        errors = []
        for trial in range(4):
            count = int(steps[trial, index])
            states = data['states'][trial, index, 1:count + 1].astype(np.float64)
            gravity = 2 * (states[:, 4] ** 2 + states[:, 5] ** 2) - 1
            falls = (states[:, 2] < .35) | (1 + gravity > .7)
            assert bool(falls.any()) == bool(fallen[trial, index])
            if fallen[trial, index]:
                assert int(np.flatnonzero(falls)[0]) + 1 == count
            w, x, y, z = states[:, 3:7].T
            yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y*y + z*z))
            c, s = np.cos(yaw), np.sin(yaw)
            expected = np.stack((c * states[:, 36] + s * states[:, 37],
                -s * states[:, 36] + c * states[:, 37], states[:, 41]), axis=-1)
            measured = data['heading_velocity'][trial, index, :count]
            assert np.allclose(measured, expected, atol=1e-5, rtol=1e-5)
            maximum_projection_error = max(maximum_projection_error, float(np.abs(measured - expected).max()))
            errors.append(None if fallen[trial, index] else float(np.linalg.norm(
                (measured[-100:].astype(np.float64) - command).mean(axis=0)) / np.linalg.norm(command)))
        for actual, recorded in zip(errors, row['trial_errors'], strict=True):
            assert actual is recorded is None or (actual is not None and recorded is not None
                and math.isclose(actual, recorded, abs_tol=1e-12, rel_tol=1e-12))
        valid_errors = [v for v in errors if v is not None]
        median = statistics.median(valid_errors) if valid_errors else None
        assert row['non_falling_trials'] == len(valid_errors)
        assert row['all_trials_fell'] == (not valid_errors)
        recorded_median = row['median_relative_error']
        assert median is recorded_median is None or (median is not None and recorded_median is not None
            and math.isclose(median, recorded_median, abs_tol=1e-12, rel_tol=1e-12))
        display = .9 if median is None else min(median, .9)
        assert math.isclose(display, row['display_relative_error'], abs_tol=1e-12, rel_tol=1e-12)
        displays.append(display)
        if median is None:
            all_fall += 1
        else:
            defined.append(median)
    assert int(fallen.sum()) == result['fallen_trials'] and all_fall == result['all_fall_commands']
    aggregate = statistics.median(defined) if defined else None
    recorded_aggregate = result['median_defined_command_error']
    assert aggregate is recorded_aggregate is None or (aggregate is not None and recorded_aggregate is not None
        and math.isclose(aggregate, recorded_aggregate, abs_tol=1e-12, rel_tol=1e-12))
    assert math.isclose(statistics.median(displays), result['median_display_command_error'], abs_tol=1e-12, rel_tol=1e-12)
    return {'status': 'complete_local_step_arrays_verified', 'mode': result['mode'],
        'auditor_sha256': sha(Path(__file__)),
        'purpose': result['purpose'], 'result_sha256': sha(folder / 'result.json'),
        'protocol_sha256': sha(folder / 'protocol.json'), 'npz_sha256': sha(trace),
        'npz_bytes': trace.stat().st_size, 'commands': 672, 'trials': 2688,
        'initial_states_verified': 2688, 'valid_control_steps': int(steps.sum()),
        'fallen_trials': int(fallen.sum()), 'all_fall_commands': all_fall,
        'maximum_heading_projection_abs_error': maximum_projection_error,
        'all_arrays_crc_dtype_finite_padding_verified': True,
        'first_fall_recomputed_from_saved_states_float64': True,
        'heading_projection_recomputed_float64': True,
        'all_command_errors_recomputed_from_saved_velocities': True,
        'heading_projection_tolerance': {'atol': 1e-5, 'rtol': 1e-5},
        'score_comparison_tolerance': {'atol': 1e-12, 'rtol': 1e-12},
        'policy_replay_executed': False, 'repeatability_verified': False,
        'paper_results_reproduced': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh audit output')
    record = inspect(args.evaluation)
    args.output.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
