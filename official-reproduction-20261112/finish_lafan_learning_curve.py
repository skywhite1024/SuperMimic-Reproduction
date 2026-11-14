"""Complete missing evaluations of an already finished, paused native curve."""
import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

from run_lafan_learning_curve import POINTS, execute, verify_training, write
from select_completed_checkpoint import digest

FILES = ('result.json','protocol.json','accounting.json',
         'audit/summary.json','audit/initial-states.json')


def validated_point(folder, run, step, timing):
    result, protocol, accounting, audit, initial = [json.loads((folder/name).read_bytes()) for name in FILES]
    parent = json.loads((run/'result.json').read_bytes())
    if (result['status'] != 'full_corpus_evaluation_completed'
            or result['clips'] != 40 or len(result['records']) != 40
            or accounting['reference_frames'] != 441140
            or protocol['manual_reset_version'] != 2):
        raise ValueError('A complete canonical40 manual-reset-v2 point is required')
    selection = protocol['checkpoint_selection']
    assert selection['parent_result_sha256'] == digest(run/'result.json')
    assert selection['resolved_config_sha256'] == digest(run/'resolved-config.json')
    assert selection['parent_collection_steps'] == parent['collection_steps']
    assert selection['evaluated_collection_steps'] == step
    snapshot = next(r for r in parent['retained_checkpoint_series'] if r['collection_steps'] == step)
    assert [r['sha256'] for r in selection['files']] == [r['sha256'] for r in snapshot['files']]
    assert audit['result_sha256'] == accounting['result_sha256'] == digest(folder/'result.json')
    assert accounting['protocol_sha256'] == initial['protocol_sha256'] == digest(folder/'protocol.json')
    assert audit['accounting_sha256'] == digest(folder/'accounting.json')
    assert initial['trajectory_integrity_summary_sha256'] == digest(folder/'audit/summary.json')
    assert audit['all_trajectory_arrays_verified'] and audit['verified_clips'] == 40
    assert initial['all_saved_initial_states_verified'] and initial['saved_initial_states_verified'] == 40
    rows = result['records']
    assert sorted(r['index'] for r in rows) == list(range(40))
    assert sum(r['reference_frames'] for r in rows) == 441140
    assert audit['verified_steps'] == sum(r['steps'] for r in rows)
    assert audit['verified_successes'] == accounting['successes'] == result['successes']
    assert accounting['successes'] + accounting['failures'] == 40
    assert accounting['maximum_step_reward'] == .1
    mean = sum(r['reward_sum']/r['reference_frames']/.1 for r in rows)/40
    assert math.isclose(mean,accounting['mean_normalized_full_motion_return'],abs_tol=1e-12,rel_tol=0)
    point = dict(collection_steps=step,status='complete',
        mean_normalized_full_motion_return=mean,successes=accounting['successes'],
        failures=accounting['failures'],reference_frames=441140,
        checkpoint_time=next(r for r in timing['checkpoints'] if r['collection_steps']==step))
    for key,name in zip(('result_sha256','protocol_sha256','accounting_sha256',
                         'trajectory_audit_sha256','initial_states_audit_sha256'),FILES,strict=True):
        point[key] = digest(folder/name)
    return point, protocol['motion_files']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paused-main',type=Path,required=True)
    parser.add_argument('--motions',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    paused, motions, output = [p.resolve() for p in (args.paused_main,args.motions,args.output)]
    run = paused/'training'
    output.mkdir(parents=True,exist_ok=False)
    report = dict(status='recovering_completed_training',training_rerun=False,
        original_training_run=str(run),collection_steps=64001,seed=42,
        points=[],evaluation_steps=POINTS,reused_evaluations=[],new_evaluations=[],
        threshold=.82,paper_results_reproduced=False,
        threshold_convention='First observed sampled checkpoint >=0.82; no interpolation',
        old_partial_evaluations_overwritten=False,
        wall_clock_origin='Original native training runner entry, not evaluation resumption',
        matched_ppo_sim_comparison_completed=False)
    receipts = []
    write(output/'result.json',report)
    try:
        timing = verify_training(run,64001,4000)
        (output/'training').mkdir()
        for name in ('result.json','resolved-config.json','checkpoint-times.json'):
            dest = output/'training'/name
            shutil.copyfile(run/name,dest)
            assert digest(dest) == digest(run/name)
            receipts.append(dest)
        report.update(training_snapshots_and_monotonic_timing_verified=True,retained_snapshots=17)
        corpus_hashes = None
        for step in POINTS:
            original, target = paused/f'eval-{step:09d}', output/f'eval-{step:09d}'
            point = dict(collection_steps=step,status='failed')
            try:
                if all((original/name).is_file() for name in FILES):
                    # Reuse only after binding all original JSONs to this exact native snapshot.
                    point, hashes = validated_point(original,run,step,timing)
                    for name in FILES:
                        dest = target/name
                        dest.parent.mkdir(parents=True,exist_ok=True)
                        shutil.copyfile(original/name,dest)
                        assert digest(dest) == digest(original/name)
                    report['reused_evaluations'].append(step)
                else:
                    execute('evaluate_motion_corpus_v2.py',['--run',run,
                        '--checkpoint-collection-step',step,'--motions',motions,'--output',target,
                        '--expected-clips',40,'--num-envs',40,'--order','index','--seed',42],
                        output/f'eval-{step:09d}.log',900)
                    execute('audit_corpus_result.py',['--result',target/'result.json',
                        '--protocol',target/'protocol.json','--motions',motions,
                        '--expected-clips',40,'--require-trajectories','--output',target/'accounting.json'],
                        output/f'accounting-{step:09d}.log',180)
                    execute('audit_saved_initial_states_v2.py',['--evaluation',target,
                        '--motions',motions,'--expected-clips',40,'--output',target/'audit'],
                        output/f'audit-{step:09d}.log',180)
                    point, hashes = validated_point(target,run,step,timing)
                    report['new_evaluations'].append(step)
                if corpus_hashes is None:
                    corpus_hashes = hashes
                assert hashes == corpus_hashes
                receipts += [target/name for name in FILES]
            except Exception as error:
                point = dict(collection_steps=step,status='failed',error=f'{type(error).__name__}: {error}')
            report['points'].append(point)
            write(output/'result.json',report)
            print('CURVE_POINT_RESULT '+json.dumps(point),flush=True)
        report['status'] = 'complete' if all(r['status']=='complete' for r in report['points']) else 'incomplete'
        report['first_observed_sample_at_or_above_threshold'] = next((r for r in report['points']
            if r['status']=='complete' and r['mean_normalized_full_motion_return']>=.82),None)
    except Exception as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}')
    write(output/'result.json',report)
    receipts.append(output/'result.json')
    print('LAFAN_CURVE_RECEIPTS_BEGIN main',flush=True)
    subprocess.run([sys.executable,str(Path(__file__).parent/'emit_json_receipts.py'),
                    *map(str,receipts)],check=True)
    print('LAFAN_CURVE_RECEIPTS_END main',flush=True)
    if report['status'] != 'complete':
        raise RuntimeError('Curve recovery incomplete; preserved all returned originals')
    print('LAFAN_CURVE_PIPELINE_COMPLETED',flush=True)


if __name__ == '__main__':
    main()
