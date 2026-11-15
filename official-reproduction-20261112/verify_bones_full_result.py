"""Locally recompute every raw-BONES motion row from the hash-bound original JSON."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--recovery-log', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, default=Path(__file__).parent / 'bones-warm-tail-1115d')
    args = parser.parse_args()
    read = lambda name: json.loads((args.evidence / name).read_bytes())
    parent = read('result.json')
    accounting = read('evaluation/accounting.json')
    protocol = read('evaluation/protocol.json')
    audit = read('evaluation/audit/summary.json')
    data = args.result.read_bytes()
    original_sha = hashlib.sha256(data).hexdigest()
    assert len(data) == parent['full_result_original']['bytes']
    assert original_sha == parent['full_result_original']['sha256'] == accounting['result_sha256'] == audit['result_sha256']
    result = json.loads(data)
    assert result['status'] == 'full_corpus_evaluation_completed'
    rows = result['records']
    assert len(rows) == result['clips'] == accounting['clips'] == 142220
    assert sorted(row['index'] for row in rows) == list(range(142220))
    maximum = sum(accounting['positive_reward_weights'].values()) * accounting['control_dt']
    assert maximum == .1
    frames = steps = successes = 0
    reward_sum = normalized_sum = 0.
    failures = Counter()
    for row in rows:
        assert 0 < row['steps'] <= row['reference_frames'] and math.isfinite(row['reward_sum'])
        terms = row['termination_terms']
        success = 'motion_end' in terms and not any(not protocol['termination_terms'][term]['time_out'] for term in terms)
        assert success == row['success']
        if success:
            assert row['steps'] == row['reference_frames']
        else:
            failures['+'.join(sorted(terms))] += 1
        frames += row['reference_frames']; steps += row['steps']; successes += success
        reward_sum += row['reward_sum']; normalized_sum += row['reward_sum'] / row['reference_frames'] / maximum
    mean = normalized_sum / len(rows)
    weighted = reward_sum / frames / maximum
    assert frames == accounting['reference_frames'] == 51915358 and steps == audit['verified_steps']
    assert successes == result['successes'] == accounting['successes']
    assert len(rows) - successes == accounting['failures']
    assert dict(failures) == accounting['failure_termination_counts']
    for value, key in [(mean, 'mean_normalized_full_motion_return'), (weighted, 'frame_weighted_normalized_full_motion_return'),
                       (steps / frames, 'tracked_fraction_frames'),
                       (statistics.median(row['steps'] * accounting['control_dt'] for row in rows), 'median_tracked_seconds')]:
        assert math.isclose(value, accounting[key], rel_tol=1e-13)
    validation = dict(status='all142220_original_per_motion_rows_recomputed_locally', original_result_bytes=len(data),
        original_result_sha256=original_sha, recovery_download_gzip_sha256=hashlib.sha256(args.recovery_log.read_bytes()).hexdigest(),
        original_JSON_bytes_and_SHA256_verified=True, all_original_per_motion_rows_loaded_locally=True,
        clips=len(rows), reference_frames=frames, tracked_steps=steps, successes=successes, failures=len(rows)-successes,
        success_fraction=successes/len(rows), mean_normalized_full_motion_return=mean,
        frame_weighted_normalized_full_motion_return=weighted, tracked_fraction_frames=steps/frames,
        failure_termination_counts=dict(failures), all_metrics_match_cloud_accounting=True,
        exact_training_continuation=False, raw_result_kept_out_of_git=True, raw_NPZ_downloaded=False,
        physics_or_policy_replayed=False, author_exact_filtered_membership_verified=False, paper_results_reproduced=False)
    (args.evidence / 'local-per-motion-recomputation.json').write_text(json.dumps(validation, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(validation, indent=2))


if __name__ == '__main__':
    main()
