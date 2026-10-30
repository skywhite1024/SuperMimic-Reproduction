"""Descriptive matched failure/exposure audit; no training or acceptance changes."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    sources = {}

    def read(name):
        path = ROOT / name
        with path.open('rb') as stream:
            sources[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
        return json.loads(path.read_text(encoding='utf8'))

    proof = read('runs/unified-retention-v1-h100-1000/proof.json')
    assert proof['new_updates'] == 1000 and len(proof['retention']['records']) == 800
    coverage = read('evidence/retention-1030-coverage-diagnostic-from-console.json')
    anchors = read('evidence/retention-1030-anchor-diagnostic-from-console.json')
    assert not coverage['canonical_AFS_file_hash_verification_pending']
    assert not anchors['canonical_AFS_file_hash_verification_pending']
    # Historical teacher assessments use the exact weights, but do not establish
    # behavior under the retention trial's deployment bound.
    import torch
    plan = read('retention-trial-v1-plan.json')
    teacher_path = ROOT / plan['new_teacher_local']
    assert hashlib.sha256(teacher_path.read_bytes()).hexdigest() == plan['new_teacher_sha256']
    teacher = torch.load(teacher_path, map_location='cpu', weights_only=True)
    teacher_history = []
    candidates = [teacher_path, ROOT / 'policy/unified-h100-stage2-4000.pt']
    for mode in ['unified-stage2-corrected-cuda-all55', 'unified-stage2-corrected-cpu-all55']:
        summary = read(f'runs/{mode}/summary.json')
        quality = {x['clip']: x for x in read(f'runs/{mode}/combined-assessment.json')['clips']}
        exact = next(path for path in candidates if path.is_file() and
                     hashlib.sha256(path.read_bytes()).hexdigest() == summary['checkpoint_sha256'])
        saved = torch.load(exact, map_location='cpu', weights_only=True)
        assert teacher['pi'].keys() == saved['pi'].keys()
        assert all(torch.equal(teacher['pi'][key], saved['pi'][key]) for key in teacher['pi'])
        teacher_history.append({'run': mode, 'device': summary['device'],
                                'policy_tensors_equal_frozen_new_teacher': True,
                                'controller_variant': summary.get('controller_variant'),
                                'matched_bound20_teacher_qualification_claimed': False,
                                'protected_clips': [{**{key: row.get(key) for key in
                                    ['clip', 'norm_return_original', 'passed_original_hard_gate', 'first_original_hard_failure_s']},
                                    'strict_passed': quality[row['clip']]['combined_engineering_passed']}
                                    for row in summary['clips'] if row['clip'] in plan['protected_new']]})
    del teacher, saved
    pairs = [
        ('cuda', 'raw', 'retention-v1-parent-bound20-raw-cuda-new15', 'retention-v1-raw-cuda-new15'),
        ('cuda', 'calibrated', 'retention-v1-parent-bound20-corrected-cuda-all55', 'retention-v1-corrected-cuda-all55'),
        ('cpu', 'raw', 'stage3-actionbound20-raw-cpu-new15', 'retention-v1-raw-cpu-new15'),
        ('cpu', 'calibrated', 'stage3-actionbound20-corrected-cpu-all55', 'retention-v1-corrected-cpu-all55'),
    ]
    libraries = []
    metrics = ['norm_return_original', 'root_xy_rmse_m', 'root_xy_peak_m', 'body_global_mean_m',
               'body_local_mean_m', 'yaw_rmse_deg', 'root_orientation_geodesic_rmse_deg',
               'first_original_hard_failure_s']
    for device, variant, parent, candidate in pairs:
        entries = []
        for mode in [parent, candidate]:
            summary = read(f'runs/{mode}/summary.json')
            quality = read(f'runs/{mode}/combined-assessment.json')
            tracking = read(f'runs/{mode}/tracking-assessment.json')
            assert summary['device'] == device and summary['post_initialization_resets'] == 0
            entries.append((summary, {x['clip']: x for x in quality['clips']},
                            {x['clip']: x for x in tracking['clips']}))
        old_rows = {x['clip']: x for x in entries[0][0]['clips']}
        clips = []
        for row in entries[1][0]['clips']:
            name = row['clip']
            before = old_rows[name]
            item = {'clip': name, 'group': row['group'],
                    'parent_original_gate': before['passed_original_hard_gate'],
                    'candidate_original_gate': row['passed_original_hard_gate'],
                    'parent_full_denominator_return': before['norm_return_original'],
                    'candidate_full_denominator_return': row['norm_return_original'],
                    'return_delta': row['norm_return_original'] - before['norm_return_original']}
            if row['group'] == 'new':
                item['parent_quality'] = entries[0][1][name]
                item['candidate_quality'] = entries[1][1][name]
                item['parent_tracking'] = {k: entries[0][2][name].get(k) for k in metrics}
                item['candidate_tracking'] = {k: entries[1][2][name].get(k) for k in metrics}
            clips.append(item)
        assert sum(x['group'] == 'new' for x in clips) == 15
        libraries.append({'device': device, 'reference': variant, 'parent': parent, 'candidate': candidate,
                          'new_denominator': 15, 'old_denominator': 40 if variant == 'calibrated' else None,
                          'new_strict_lost': [x['clip'] for x in clips if x['group'] == 'new' and
                                              x['parent_quality']['combined_engineering_passed'] and
                                              not x['candidate_quality']['combined_engineering_passed']],
                          'new_strict_gained': [x['clip'] for x in clips if x['group'] == 'new' and
                                                not x['parent_quality']['combined_engineering_passed'] and
                                                x['candidate_quality']['combined_engineering_passed']],
                          'old_original_gate_lost': [x['clip'] for x in clips if x['group'] == 'lafan' and
                                                    x['parent_original_gate'] and not x['candidate_original_gate']],
                          'old_original_gate_gained': [x['clip'] for x in clips if x['group'] == 'lafan' and
                                                      not x['parent_original_gate'] and x['candidate_original_gate']],
                          'clips': clips})
    sequence_modes = [
        'retention-v1-parent-bound20-raw-continuous-cuda',
        'retention-v1-parent-bound20-corrected-continuous-cuda',
        'retention-v1-raw-continuous-cuda', 'retention-v1-corrected-continuous-cuda',
        'stage3-actionbound20-frozen-continuous-v1', 'stage3-actionbound20-corrected-continuous-v1',
        'retention-v1-raw-continuous-cpu', 'retention-v1-corrected-continuous-cpu',
    ]
    sequences = []
    for mode in sequence_modes:
        summary = read(f'runs/{mode}/summary.json')
        assessment = read(f'runs/{mode}/continuous-assessment.json')
        contacts = read(f'runs/{mode}/continuous-contact-review.json')
        variant = 'corrected' if summary['reference_version'] == 'continuous-corrected-v1' else 'frozen'
        index = read(f'continuous-{variant}-v1/index.json')
        contact_lookup = {x['seed']: x for x in contacts['trials']}
        quality_lookup = {x['seed']: x for x in assessment['trials']}
        trace = ROOT / f'runs/{mode}/trajectory.npz'
        with trace.open('rb') as stream:
            sources[str(trace.relative_to(ROOT)).replace('\\', '/')] = hashlib.file_digest(stream, 'sha256').hexdigest()
        with np.load(trace) as archive:
            xy = archive['xy_error'].copy()
        assert len(summary['trials']) == 3 and summary['post_initialization_resets'] == 0
        trials = []
        for column, trial in enumerate(summary['trials']):
            seed = trial['seed']
            reference = next(x for x in index['sequences'] if x['seed'] == seed)

            def locate(seconds):
                if seconds is None:
                    return None
                frame = int(round(seconds * 50))
                for segment in reference['segments']:
                    if segment['transition_start_frame'] <= frame < segment['source_start_in_sequence_frame']:
                        return {'phase': 'reference_bridge', 'source': segment['source'], 'group': segment['group']}
                    if segment['source_start_in_sequence_frame'] <= frame <= segment['end_frame']:
                        return {'phase': 'source_segment', 'source': segment['source'], 'group': segment['group'],
                                'source_elapsed_s': (frame - segment['source_start_in_sequence_frame']) * .02}
                return {'phase': 'initial_or_terminal_hold'}

            peak_time = int(np.nanargmax(xy[:6001, column])) * .02
            contact = contact_lookup[seed]
            quality = quality_lookup[seed]
            trials.append({'seed': seed, 'seconds': 120, 'post_initialization_resets': trial['post_initialization_resets'],
                           'original_gate_passed': trial['passed_original_hard_gate'],
                           'first_original_failure_s': trial['first_hard_failure_s'],
                           'first_failure_location': locate(trial['first_hard_failure_s']),
                           'root_xy_peak_s': peak_time, 'root_xy_peak_m': float(np.nanmax(xy[:6001, column])),
                           'root_xy_peak_location': locate(peak_time),
                           'strict_passed': quality['continuous_engineering_passed'],
                           'strict_failed_checks': [k for k, v in quality['checks'].items() if not v],
                           'failed_new_segments': [{k: v for k, v in row.items() if k in
                               ['source', 'physical_start_s', 'physical_end_s', 'root_xy_rmse_m', 'root_xy_peak_m', 'checks']}
                               for row in quality['new_full_segments'] if not row['passed']],
                           'transition_windows': trial['transition_windows'],
                           'contact_review_passed': contact['contact_review_passed'],
                           'unexpected_trunk_contact_samples': contact['unexpected_trunk_contact_samples'],
                           'unexpected_trunk_contact_events': contact['unexpected_trunk_contact_events'],
                           'actual_min_after_initial_100ms_m': contact['actual_min_after_initial_100ms_m']})
        sequences.append({'mode': mode, 'device': summary['device'], 'reference_version': summary['reference_version'],
                          'original_gate_passed': summary['passed_original_hard_gate'],
                          'strict_passed': assessment['continuous_engineering_passed'], 'denominator': 3, 'trials': trials})
    result = {'status': 'all_matched_libraries_and_fixed_sequences_descriptively_audited',
              'created_at': datetime.now(timezone.utc).isoformat(), 'libraries': libraries, 'sequences': sequences,
              'exposure': coverage, 'teacher_anchor': anchors,
              'historical_new_teacher_assessments': teacher_history,
              'inferences': [
                  'The bounded candidate fails the frozen adoption rule and is not adopted.',
                  'The protected-clip PI exposure is uneven; the declared rule renormalizes over available clips.',
                  'A zero minimum eligibility value is pooled over WM and PI pool builds and does not identify its time or duration.',
                  'Lower teacher loss on sampled histories does not establish closed-loop tracking or retention.',
                  'Failure timing and geometry are descriptive observations, not a causal identification of training regression.',
                  'No reference, physics, reward, denominator, reset rule or acceptance threshold was changed.'
              ], 'causal_explanation_established': False, 'contact_forces_measured': False,
              'new_training_triggered': False, 'sources': sources}
    target = ROOT / 'evidence/retention-1030-failure-diagnostic.json'
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'library_pairs': len(libraries),
                      'continuous_groups': len(sequences), 'full_trials': 24,
                      'new_strict_losses': [{k: row[k] for k in ['device', 'reference', 'new_strict_lost', 'new_strict_gained', 'old_original_gate_lost', 'old_original_gate_gained']} for row in libraries]}), flush=True)


if __name__ == '__main__':
    main()
