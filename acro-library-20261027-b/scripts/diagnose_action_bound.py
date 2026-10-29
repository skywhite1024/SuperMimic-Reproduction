"""Matched complete-tail action-bound comparison; distinguish timing from causation."""
import json
import math
from pathlib import Path
import numpy as np
from runtime import ROOT


def compare(baseline_name, bounded_name, suite):
    left, right = ROOT/'runs'/baseline_name, ROOT/'runs'/bounded_name
    if not (left/'summary.json').exists() or not (right/'summary.json').exists():
        return None
    a, b = [json.loads((folder/'summary.json').read_text(encoding='utf8')) for folder in [left, right]]
    assert a['checkpoint_sha256'] == b['checkpoint_sha256'] and a['device'] == b['device'] == 'cpu'
    assert b['controller_variant'] == 'action-output-bound-20-v1'
    with np.load(left/'trajectory.npz') as archive:
        x = {key: archive[key] for key in ['state', 'reference_qpos', 'action']}
    with np.load(right/'trajectory.npz') as archive:
        y = {key: archive[key] for key in ['state', 'reference_qpos', 'action']}
    assert x['state'].shape == y['state'].shape
    assert np.array_equal(x['reference_qpos'], y['reference_qpos'], equal_nan=True)
    rows = []
    key = 'clips' if suite == 'library' else 'trials'
    for index, (old, new) in enumerate(zip(a[key], b[key])):
        name = old['clip'] if suite == 'library' else f'seed-{old["seed"]}'
        length = old['full_clip_steps'] if suite == 'library' else 6000
        action = x['action'][:length, index]
        bounded_action = y['action'][:length, index]
        assert np.isfinite(action).all() and np.isfinite(bounded_action).all()
        assert float(np.abs(bounded_action).max()) <= 20.
        exceed = np.any(np.abs(action) > 20, axis=-1)
        first = int(np.flatnonzero(exceed)[0]) if exceed.any() else None
        first_fail = old['first_original_hard_failure_s'] if suite == 'library' else old['first_hard_failure_s']
        before = min(length, round(first_fail/.02)) if first_fail is not None else length
        difference = np.max(np.abs(x['state'][:length+1,index]-y['state'][:length+1,index]), axis=-1)
        differing = np.flatnonzero(difference > 1e-5)
        row = {'clip': name, 'group': old.get('group', 'continuous'), 'full_steps': length,
               'baseline_original_hard_gate_passed': old['passed_original_hard_gate'],
               'bounded_original_hard_gate_passed': new['passed_original_hard_gate'],
               'baseline_first_failure_s': first_fail,
               'bounded_first_failure_s': new['first_original_hard_failure_s'] if suite == 'library' else new['first_hard_failure_s'],
               'baseline_peak_abs_action': float(np.abs(action).max()),
               'first_baseline_output_exceeds20_s': None if first is None else first*.02,
               'exceeding_action_elements_before_first_failure': int((np.abs(action[:before]) > 20).sum()),
               'exceeding_action_elements_full_tail': int((np.abs(action) > 20).sum()),
               'overshoot_only_after_first_failure': first_fail is not None and first is not None and first*.02 >= first_fail,
               'first_state_difference_over1e5_s': None if not len(differing) else int(differing[0])*.02,
               'max_state_difference_before_first_clipped_action': float(difference[:length+1 if first is None else first+1].max()),
               'baseline_root_xy_rmse_m': old['root_xy_rmse_m'], 'bounded_root_xy_rmse_m': new['root_xy_rmse_m']}
        rows.append(row)
    result = {'baseline': baseline_name, 'bounded': bounded_name, 'suite': suite,
              'checkpoint_sha256': a['checkpoint_sha256'], 'reference_poses_byte_equal': True, 'matched_cpu_backend': True,
              'rows': rows, 'groups_baseline': a.get('groups'), 'groups_bounded': b.get('groups'),
              'hard_gate_baseline': a.get('passed_original_hard_gate'), 'hard_gate_bounded': b.get('passed_original_hard_gate')}
    for label, folder in [('baseline',left), ('bounded',right)]:
        path = folder / ('combined-assessment.json' if suite == 'library' else 'continuous-assessment.json')
        if path.exists():
            quality = json.loads(path.read_text(encoding='utf8'))
            result[label+'_strict_passed'] = quality['combined_engineering_passed'] if suite == 'library' else quality['continuous_engineering_passed']
    return result


def main():
    pairs = [('unified-stage3-corrected-cpu-all55','stage3-actionbound20-corrected-cpu-all55','library'),
             ('stage3-raw-cpu-new15-control','stage3-actionbound20-raw-cpu-new15','library'),
             ('stage3-frozen-continuous-v1','stage3-actionbound20-frozen-continuous-v1','sequences'),
             ('stage3-corrected-continuous-v1','stage3-actionbound20-corrected-continuous-v1','sequences')]
    comparisons = [result for arguments in pairs if (result := compare(*arguments)) is not None]
    result = {'status': 'all_four_matched_comparisons_complete' if len(comparisons)==4 else 'partial_matched_comparisons',
              'training_updates': 0, 'controller_variant': 'action-output-bound-20-v1',
              'interpretation': 'An upstream collection/imagination/deployment bound inconsistency is established. A threshold exceedance after first physical failure cannot explain that initial failure; deployment-only results do not establish a training-time cause.',
              'comparisons': comparisons}
    (ROOT/'evidence/action-bound-matched-comparison.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'status':result['status'],'comparisons':[{'baseline':item['baseline'],'bounded':item['bounded'],
          'baseline_strict_passed':item.get('baseline_strict_passed'),'bounded_strict_passed':item.get('bounded_strict_passed'),
          'failing_rows':[row for row in item['rows'] if not row['baseline_original_hard_gate_passed']]} for item in comparisons]}),flush=True)


if __name__=='__main__':
    main()
