"""Describe verified rollout differences without inferring a causal mechanism."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path
import statistics


CLIPS = ['dance2_subject4', 'fallAndGetUp1_subject5', 'fallAndGetUp2_subject3']
FIELDS = ['height_error_m', 'tilt_component_error', 'root_position_error_m',
          'root_orientation_error_rad', 'joint_position_rmse_rad', 'action_rms', 'action_delta_rms']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def window(rows):
    return {key: {'mean': statistics.fmean(r[key] for r in rows),
                  'max': max(r[key] for r in rows)} for key in FIELDS}


def summarize_trace(path, summary):
    with gzip.open(path, 'rt', newline='') as stream:
        rows = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]
    assert len(rows) == summary['steps'] and rows
    assert all(math.isfinite(value) for row in rows for value in row.values())
    assert all(abs(row['time_seconds'] - (index + 1) * .02) < 1e-8 for index, row in enumerate(rows))
    assert all(bool(row['terminated']) == (bool(row['height_gate']) or bool(row['tilt_gate'])) for row in rows)
    last = rows[-1]
    assert bool(last['terminated']) == summary['failed']
    gates = [key for key in ['height_gate', 'tilt_gate'] if last[key] > .5]
    near_threshold = {}
    for field, threshold in [('height_error_m', summary['height_gate_threshold_m']),
                             ('tilt_component_error', summary['tilt_component_gate_threshold'])]:
        indices = [i for i, row in enumerate(rows) if row[field] >= .9 * threshold]
        near_threshold[field] = {'threshold': threshold,
                                 'first_at_90_percent_seconds': rows[indices[0]]['time_seconds'] if indices else None,
                                 'fraction_at_90_percent': len(indices) / len(rows)}
    return {'steps': len(rows), 'seconds': last['time_seconds'], 'failed': summary['failed'],
            'final_gates': gates, 'last_transition': last,
            'whole_clip_observed': window(rows), 'last_second': window(rows[-50:]),
            'last_ten_seconds': window(rows[-500:]), 'threshold_proximity': near_threshold}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    root = args.results.resolve()
    assert read(root.parent / 'local-transfer-verification.json')['status'] == 'archive_and_every_file_verified'
    baseline = read(root / 'original-evaluation.json')
    selection = read(root / 'selection.json')
    best = read(root / 'evaluations' / f'{selection["best"]["iteration"]:06d}.json')
    assert len(best['per_clip']) == len(baseline['per_clip']) == 40
    assert list(best['per_clip']) == list(baseline['per_clip'])
    changes = []
    for name, old in baseline['per_clip'].items():
        new = best['per_clip'][name]
        assert old['full_clip_steps'] == new['full_clip_steps']
        changes.append({'clip': name, 'old_return': old['norm_return'], 'new_return': new['norm_return'],
                        'macro_delta_contribution': (new['norm_return'] - old['norm_return']) / 40,
                        'old_failed': old['failed'], 'new_failed': new['failed'],
                        'old_seconds': old['steps'] * .02, 'new_seconds': new['steps'] * .02,
                        'full_seconds': new['full_clip_steps'] * .02})
    delta = best['norm_return'] - baseline['norm_return']
    assert abs(sum(r['macro_delta_contribution'] for r in changes) - delta) < 1e-5
    traces = {}
    auxiliary=root/'postprocess-20261026-a'
    trace_root=auxiliary if (auxiliary/'evidence/postprocess.json').is_file() else root
    comparisons={}
    for folder in ['parent-012000', 'selected']:
        receipt = read(trace_root / 'diagnostics' / folder / 'summary.json')
        assert receipt.get('independent_trace_contract_checked',receipt['same_all40_rollout_metrics_reproduced'])
        comparisons[folder]=read(trace_root/'diagnostics'/folder/'primary-comparison.json') if trace_root==auxiliary else {'strict_all40_match':True}
        traces[folder] = {name: summarize_trace(trace_root / 'diagnostics' / folder / (name + '.csv.gz'),
                                              receipt['clips'][name]) for name in CLIPS}
    output = {'selected_iteration': selection['best']['iteration'], 'macro_delta': delta,
              'rescued_clips': [r['clip'] for r in changes if r['old_failed'] and not r['new_failed']],
              'newly_failed_clips': [r['clip'] for r in changes if not r['old_failed'] and r['new_failed']],
              'still_failed_clips': [r['clip'] for r in changes if r['old_failed'] and r['new_failed']],
              'all40_changes': sorted(changes, key=lambda r: r['macro_delta_contribution']),
              'traces': traces,'primary_reference_comparison':comparisons,
              'boundary': 'Descriptive evidence from independent numerical rollouts with the frozen policy/MDP/thresholds. Strict primary replay differences remain recorded; these traces do not replace primary acceptance metrics. Threshold branches explain observed termination, not the learning cause. Only the three original failed clips have per-step traces.'}
    destination = root.parent / 'diagnostic-analysis.json'
    destination.write_text(json.dumps(output, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({key: output[key] for key in ['selected_iteration', 'macro_delta', 'rescued_clips', 'newly_failed_clips', 'still_failed_clips']}))


if __name__ == '__main__':
    main()
