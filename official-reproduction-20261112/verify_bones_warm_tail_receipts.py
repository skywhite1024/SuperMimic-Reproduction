"""Recover bounded BONES warm-tail originals without treating weights as a full resume.

The large per-motion result, weight files and trajectory arrays remain on AFS.
This verifies the original cloud audit receipts; it does not locally replay physics
or independently recompute the aggregate from the large per-motion result.
"""
import argparse
import gzip
import json
import math
from pathlib import Path
import re

from verify_horizon_grid_receipts import decode_group, sha


TRAINING_NAMES = ('result.json', 'resolved-config.json',
                  'initialization-readback.json', 'checkpoint-times.json')
EVALUATION_NAMES = ('protocol.json', 'accounting.json',
                    'audit/summary.json', 'audit/initial-states.json')


def verify_training(folder, plan, steps, interval):
    read = lambda name: json.loads((folder / name).read_bytes())
    training, cfg, initialization, timing = [read(name) for name in TRAINING_NAMES]
    assert training['status'] == 'pilot_completed'
    assert training['collection_steps'] == cfg['collection_steps'] == steps
    assert training['seed'] == cfg['seed'] == 42
    assert training['num_envs'] == cfg['num_envs'] == 4096
    assert training['corpus'] == cfg['corpus']
    assert training['corpus']['name'] == 'bones-raw'
    assert training['corpus']['clips'] == 142220 and training['corpus']['frames'] == 51915358
    assert not training['corpus']['paper_name_filter_applied']
    assert not training['corpus']['paper_feasibility_filter_applied']
    assert training['goal_assembly_backend'] == cfg['goal_assembly_backend'] == 'eager'
    assert cfg['training_profile'] == 'paper-a5-single-gpu' and not cfg['vendor_files_modified']
    assert cfg['runner']['save_interval'] == interval
    assert cfg['runner']['history_length'] == 10 and cfg['runner']['actor_obs'] == 'proprio'
    assert cfg['runner']['policy']['horizon'] == 32 and cfg['runner']['policy']['gamma'] == .99
    assert training['inherited_collection_steps'] == 296001
    assert training['combined_collection_steps_including_warm_start'] == 296001 + steps
    assert initialization['source_manifest'] == plan['source_snapshot']['path']
    assert initialization['source_manifest_sha256'] == plan['source_snapshot']['sha256']
    sources = {Path(row['path']).name: row for row in plan['source_originals_bound']}
    assert initialization['source_config_sha256'] == sources['resolved-config.json']['sha256']
    assert initialization['inherited_collection_steps'] == initialization['learning_rate_offset'] == 296001
    for key in ('optimizer_state_restored', 'replay_buffer_restored', 'sampler_state_restored',
                'simulator_state_restored', 'RNG_state_restored', 'exact_training_continuation'):
        assert initialization[key] is False
    assert initialization['normalization_buffers_restored']
    assert initialization['all_four_native_state_dictionaries_equal_source']
    assert initialization['all_learned_weights_and_normalization_buffers_read_back']
    for row in initialization['source_files']:
        original = sources[row['file']]
        assert row['bytes'] == original['bytes'] and row['sha256'] == original['sha256']
    assert {row['file'] for row in initialization['source_files']} == {'world_model.pt', 'actor.pt', 'critic.pt'}
    assert all(initialization[key] == value for key, value in cfg['initialization'].items())
    completed = training['initialization']
    assert all(completed[key] == value for key, value in initialization.items())
    assert completed['all_three_learned_networks_changed']
    before, after = completed['initial_learned_parameter_hashes'], completed['final_learned_parameter_hashes']
    assert set(before) == set(after) == {'world_model', 'actor', 'critic'}
    assert all(len(before[key]) == len(after[key]) == 64 and before[key] != after[key] for key in before)
    expected = list(range(1, steps + 1, interval))
    snapshots = training['retained_checkpoint_series']
    assert [row['collection_steps'] for row in snapshots] == expected
    assert [row['collection_steps'] for row in timing['checkpoints']] == expected
    for row in snapshots:
        assert row['status'] == 'model_snapshot_preserved' and not row['full_training_resume_available']
        assert {item['file'] for item in row['files']} == {'world_model.pt', 'actor.pt', 'critic.pt'}
    assert {Path(row['path']).name: (row['bytes'], row['sha256']) for row in training['checkpoints']} == {
        row['file']: (row['bytes'], row['sha256']) for row in snapshots[-1]['files']}
    previous = {key: -1. for key in ('seconds_since_program_entry', 'seconds_since_runner_run_entry')}
    for row in timing['checkpoints']:
        for event in ('callback_entry_after_native_save', 'retained_snapshot_complete', 'offline_artifact_logging_returned'):
            for key in previous:
                value = row[event][key]
                assert math.isfinite(value) and value >= previous[key]
                previous[key] = value
    return dict(collections=steps, combined_collections=296001 + steps,
                real_simulator_steps=steps * 4096, retained_model_sets=len(snapshots),
                initial_learned_parameter_hashes=before, all_three_learned_networks_changed=True,
                source_manifest_sha256=initialization['source_manifest_sha256'],
                exact_training_continuation=False)


def verify_complete(folder, plan, names, log):
    read = lambda name: json.loads((folder / name).read_bytes())
    digest = lambda name: sha((folder / name).read_bytes())
    expected = [label + '/' + name for label in ('integration', 'training') for name in TRAINING_NAMES]
    expected += ['evaluation/' + name for name in EVALUATION_NAMES] + ['result.json']
    assert names == expected and len(names) == 13
    parent = read('result.json')
    assert parent['status'] == 'warm_start_tail_and_full_raw_evaluation_completed'
    assert parent['exact_training_continuation'] is False and parent['full_training_state_resume_available'] is False
    assert parent['source_manifest_sha256'] == plan['source_snapshot']['sha256']
    assert parent['inherited_collection_steps'] == 296001 and parent['new_training_collections'] == 24000
    assert parent['combined_training_collections'] == 320001
    phases = {label: verify_training(folder / label, plan, steps, interval)
              for label, steps, interval in [('integration', 21, 10), ('training', 24000, 23999)]}
    assert phases['integration']['initial_learned_parameter_hashes'] == phases['training']['initial_learned_parameter_hashes']
    assert parent['training_result_sha256'] == digest('training/result.json')
    protocol, accounting, audit, initial = [read('evaluation/' + name) for name in EVALUATION_NAMES]
    assert protocol['training'] == read('training/result.json')
    assert protocol['motion_files'] == plan['raw_motion_hashes']
    assert protocol['manual_reset_version'] == initial['manual_reset_version'] == 2
    assert protocol['clips'] == accounting['clips'] == audit['verified_clips'] == initial['saved_initial_states_verified'] == 142220
    assert protocol['num_envs'] == 1024 and protocol['batch_order'] == 'length' and protocol['seed'] == 42
    assert protocol['all_motions_included'] and not protocol['subset_filter_applied']
    assert not protocol['domain_randomization'] and not protocol['observation_noise']
    assert accounting['status'] == 'complete_corpus_accounting_verified'
    assert accounting['reference_frames'] == 51915358 and accounting['maximum_step_reward'] == .1
    assert accounting['all_indices_names_lengths_verified'] and accounting['all_trajectory_files_present_nonempty']
    assert accounting['successes'] == parent['successes'] == audit['verified_successes']
    assert accounting['failures'] == parent['failures'] == audit['verified_failures']
    assert accounting['successes'] + accounting['failures'] == 142220
    mean = accounting['mean_normalized_full_motion_return']
    assert math.isfinite(mean) and mean == parent['mean_normalized_full_motion_return']
    original = parent['full_result_original']
    assert original['path'] == plan['output'] + '/evaluation/result.json' and original['bytes'] > 0
    assert len(original['sha256']) == 64
    assert original['sha256'] == accounting['result_sha256'] == audit['result_sha256']
    assert accounting['protocol_sha256'] == initial['protocol_sha256'] == digest('evaluation/protocol.json')
    assert audit['accounting_sha256'] == digest('evaluation/accounting.json')
    assert initial['trajectory_integrity_summary_sha256'] == digest('evaluation/audit/summary.json')
    assert audit['all_trajectory_arrays_verified'] and initial['all_saved_initial_states_verified']
    assert initial['packed_source_hashes_verified'] and audit['array_widths'] == dict(full_state=71, actions=29)
    assert parent['full_raw_evaluation_completed'] and parent['all_saved_arrays_and_initial_states_verified']
    for marker in ('BONES_WARM_TAIL_PIPELINE_COMPLETED', 'BONES_WARM_TAIL_JOB_COMPLETED', 'BONES_WARM_TAIL_JOB_EXIT=0'):
        assert marker in log
    return dict(status='warm_tail_training_and_full_raw_cloud_audit_originals_verified',
                training=phases, successes=accounting['successes'], failures=accounting['failures'],
                mean_normalized_full_motion_return=mean, full_result_original=original,
                cloud_verified_trajectory_bytes=audit['verified_bytes'], cloud_verified_steps=audit['verified_steps'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    source = args.log.read_bytes()
    raw = gzip.decompress(source) if args.log.suffix == '.gz' else source
    log = '\n'.join(re.sub(r'^\[[0-9TZ: .+-]+\] ', '', line) for line in raw.decode().splitlines())
    plan = json.loads(args.plan.read_bytes())
    assert 'BONES_WARM_TAIL_BOOTSTRAP_SHA256 ' + plan['command_sha256'] in log
    assert all('BONES_WARM_TAIL_SOURCE_SHA256 ' + name + ' ' + value in log for name, value in plan['source_files'].items())
    assert 'BONES_WARM_SOURCE_296001_ORIGINALS_VERIFIED' in log
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    blocks = re.findall(r'^BONES_WARM_TAIL_RECEIPTS_BEGIN\n(.*?)^BONES_WARM_TAIL_RECEIPTS_END$', log, re.M | re.S)
    assert len(blocks) <= 1
    result = dict(status='original_receipts_not_yet_emitted', original_JSONs=0)
    ledger = []
    if blocks:
        allowed = [label + '/' + name for label in ('integration', 'training') for name in TRAINING_NAMES]
        allowed += ['evaluation/' + name for name in EVALUATION_NAMES] + ['result.json']
        names, ledger = decode_group(blocks[0], plan['output'], output, allowed)
        parent = json.loads((output / 'result.json').read_bytes())
        result.update(status='original_failed_or_incomplete_run_retained', original_JSONs=len(names),
                      reported_status=parent['status'], error=parent.get('error'))
        if parent['status'] == 'warm_start_tail_and_full_raw_evaluation_completed':
            result.update(verify_complete(output, plan, names, log))
    result.update(job_uid=plan['uid'], source_commit=plan['source_commit'],
                  source_files_bound_to_captured_log=True, log_sha256=sha(raw),
                  transport='downloaded offline gzip' if args.log.suffix == '.gz' else 'captured live terminal text',
                  all_recovered_original_bytes_and_SHA256_verified=bool(ledger),
                  large_per_motion_result_downloaded=False, aggregate_recomputed_from_local_per_motion_result=False,
                  trajectory_arrays_or_weights_downloaded=False, exact_training_continuation=False,
                  paper_results_reproduced=False)
    for name, value in [('validation.json', result), ('receipt-ledger.json', dict(receipts=ledger))]:
        (output / name).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8', newline='\n')
    if args.require_complete:
        assert result['status'] == 'warm_tail_training_and_full_raw_cloud_audit_originals_verified', 'Full result not available; recovered originals retained'
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
