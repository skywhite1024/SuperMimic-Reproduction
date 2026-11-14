"""Bounded raw-BONES weight warm start, short CUDA check, tail and full evaluation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from run_lafan_learning_curve import execute, write
from select_completed_checkpoint import digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--motions',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    source=json.loads(args.snapshot.read_bytes())
    assert source['collection_steps']==296001
    report=dict(status='starting',kind='explicit_weight_warm_start',
        source_manifest=str(args.snapshot.resolve()),source_manifest_sha256=digest(args.snapshot),
        inherited_collection_steps=296001,new_training_collections=24000,
        combined_training_collections=320001,exact_training_continuation=False,
        original_interrupted_parent_rewritten=False,paper_results_reproduced=False,
        full_training_state_resume_available=False,whole_program_seconds=18000,
        integration_collection_steps=21,main_started=False,full_raw_evaluation_completed=False)
    receipts=[]
    write(output/'result.json',report)
    try:
        for label,steps,interval,limit in [('integration',21,10,1200),('training',24000,23999,9000)]:
            run=output/label
            execute('run_lafan_pilot.py',['--motions',args.motions,'--output',run,
                '--steps',steps,'--checkpoint-interval',interval,'--keep-checkpoint-series',
                '--record-checkpoint-times','--corpus','bones-raw','--num-envs',4096,'--seed',42,
                '--training-profile','paper-a5-single-gpu','--goal-backend','eager',
                '--initial-native-snapshot',args.snapshot,'--learning-rate-offset',296001],
                output/f'{label}.log',limit)
            training=json.loads((run/'result.json').read_bytes())
            initialization=training['initialization']
            assert training['status']=='pilot_completed' and training['collection_steps']==steps
            assert initialization['source_manifest_sha256']==report['source_manifest_sha256']
            assert initialization['all_four_native_state_dictionaries_equal_source']
            assert initialization['all_three_learned_networks_changed']
            assert initialization['optimizer_state_restored'] is False
            assert initialization['replay_buffer_restored'] is False
            assert initialization['exact_training_continuation'] is False
            assert training['combined_collection_steps_including_warm_start']==296001+steps
            receipts += [run/name for name in ('result.json','resolved-config.json',
                                               'initialization-readback.json','checkpoint-times.json')]
            if label=='integration':
                report.update(integration_actual_native_CUDA_passed=True,status='training',main_started=True)
                write(output/'result.json',report)
                print('WARM_START_CUDA_INTEGRATION_PASSED',flush=True)
        run=output/'training';evaluation=output/'evaluation'
        report.update(status='evaluating',training_result_sha256=digest(run/'result.json'))
        write(output/'result.json',report)
        execute('evaluate_motion_corpus_v2.py',['--run',run,'--motions',args.motions,
            '--output',evaluation,'--expected-clips',142220,'--num-envs',1024,
            '--order','length','--seed',42],output/'evaluation.log',7200)
        execute('audit_corpus_result.py',['--result',evaluation/'result.json',
            '--protocol',evaluation/'protocol.json','--motions',args.motions,
            '--expected-clips',142220,'--require-trajectories','--output',evaluation/'accounting.json'],
            output/'accounting.log',600)
        execute('audit_saved_initial_states_v2.py',['--evaluation',evaluation,
            '--motions',args.motions,'--expected-clips',142220,'--output',evaluation/'audit'],
            output/'array-audit.log',3600)
        accounting=json.loads((evaluation/'accounting.json').read_bytes())
        audit=json.loads((evaluation/'audit/summary.json').read_bytes())
        initial=json.loads((evaluation/'audit/initial-states.json').read_bytes())
        assert accounting['clips']==142220 and accounting['reference_frames']==51915358
        assert audit['all_trajectory_arrays_verified'] and audit['verified_clips']==142220
        assert initial['all_saved_initial_states_verified'] and initial['saved_initial_states_verified']==142220
        assert accounting['result_sha256']==audit['result_sha256']==digest(evaluation/'result.json')
        assert initial['trajectory_integrity_summary_sha256']==digest(evaluation/'audit/summary.json')
        receipts += [evaluation/name for name in ('protocol.json','accounting.json',
                                                 'audit/summary.json','audit/initial-states.json')]
        report.update(status='warm_start_tail_and_full_raw_evaluation_completed',
            full_raw_evaluation_completed=True,successes=accounting['successes'],failures=accounting['failures'],
            mean_normalized_full_motion_return=accounting['mean_normalized_full_motion_return'],
            all_saved_arrays_and_initial_states_verified=True,
            full_result_original=dict(path=str(evaluation/'result.json'),
                bytes=(evaluation/'result.json').stat().st_size,sha256=digest(evaluation/'result.json')),
            large_result_and_trajectories_remain_on_AFS=True)
    except Exception as error:
        report.update(status='failed_or_incomplete',error=f'{type(error).__name__}: {error}')
    write(output/'result.json',report)
    # Include every bounded original that was written, including failed training evidence.
    for label in ('integration','training'):
        for name in ('result.json','resolved-config.json','initialization-readback.json','checkpoint-times.json'):
            path=output/label/name
            if path.is_file() and path not in receipts:receipts.append(path)
    for name in ('protocol.json','accounting.json','audit/summary.json','audit/initial-states.json'):
        path=output/'evaluation'/name
        if path.is_file() and path not in receipts:receipts.append(path)
    receipts.append(output/'result.json')
    print('BONES_WARM_TAIL_RECEIPTS_BEGIN',flush=True)
    subprocess.run([sys.executable,str(Path(__file__).parent/'emit_json_receipts.py'),
                    *map(str,receipts)],check=True)
    print('BONES_WARM_TAIL_RECEIPTS_END',flush=True)
    if report['status']!='warm_start_tail_and_full_raw_evaluation_completed':
        raise RuntimeError('Warm-start tail incomplete; existing original evidence retained')
    print('BONES_WARM_TAIL_PIPELINE_COMPLETED',flush=True)


if __name__=='__main__':
    main()
