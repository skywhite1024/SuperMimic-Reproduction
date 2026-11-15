"""Evaluate six pre-stop logged snapshots read-only; never resume training."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from select_completed_checkpoint import digest
from select_interrupted_snapshot import select_interrupted_snapshot


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8', newline='\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    args.output.mkdir(parents=True)
    source = Path(__file__).parent
    evidence = json.loads(args.evidence.read_bytes())
    points = [r['collection_steps'] for r in evidence['logged_snapshots']]
    assert points == [1, 4001, 8001, 12001, 16001, 20001]
    parent = args.parent.resolve()
    run = parent / 'main/training'
    motions = parent / 'motions'
    spec = json.loads((source / 'lafan-locomotion18-spec.json').read_bytes())
    subset = json.loads((parent / 'motion-subset.json').read_bytes())
    assert subset['source_motion_files'] == spec['source_motion_files']
    assert subset['clips'] == 18 and subset['reference_frames'] == 220352
    assert {p.name:digest(p) for p in sorted(motions.iterdir()) if p.is_file()} == subset['packed_motion_files']
    original_meta = [run/'resolved-config.json', run/'checkpoint-times.json', run/'learner-counts.json']
    original_meta += [run/'checkpoint-series'/f'step-{step:09d}'/'manifest.json' for step in points]
    original_meta_hashes = {str(p): digest(p) for p in original_meta}
    for step in points:
        select_interrupted_snapshot(run, step, args.evidence)
    write(args.output/'parent-readback.json', dict(status='all_six_logged_saved_model_sets_rehashed',
        parent_platform_status='stopped_by_user', model_sets=6,
        original_metadata_SHA256=original_meta_hashes, original_stop_evidence_SHA256=digest(args.evidence),
        parent_training_complete=False, training_reexecuted=False, full_training_resume_available=False))
    result = dict(status='evaluating', points=[], all_planned_points=points,
                  training_reexecuted=False, parent_training_complete=False,
                  full_training_resume_available=False, paper_results_reproduced=False)
    originals = [args.evidence, args.output/'parent-readback.json', *original_meta]
    started = time.monotonic()

    def stage(script, arguments, logfile, limit):
        print('INTERRUPTED_EVAL_STAGE_START '+json.dumps(dict(script=script,utc=datetime.now(timezone.utc).isoformat(),timeout_seconds=limit)),flush=True)
        command = ['timeout', '--signal=TERM', '--kill-after=30s', str(limit)+'s',
                   sys.executable, str(source/script), *map(str, arguments)]
        with logfile.open('w', encoding='utf-8') as stream:
            process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        if process.returncode:
            print(logfile.read_text(encoding='utf-8',errors='replace')[-12000:],flush=True)
            raise RuntimeError(script+' exit '+str(process.returncode))

    try:
        for step in points:
            target = args.output/f'eval-{step:09d}'
            row = dict(collection_steps=step, status='incomplete')
            try:
                stage('evaluate_motion_corpus_v2.py', ['--run',run,'--motions',motions,'--output',target,
                    '--expected-clips',18,'--num-envs',18,'--order','index','--seed',42,
                    '--checkpoint-collection-step',step,'--interrupted-run-evidence',args.evidence],
                    args.output/f'eval-{step:09d}.log',900)
                stage('audit_corpus_result.py',['--result',target/'result.json','--protocol',target/'protocol.json',
                    '--motions',motions,'--expected-clips',18,'--require-trajectories','--output',target/'accounting.json'],
                    args.output/f'account-{step:09d}.log',180)
                stage('audit_saved_initial_states_v2.py',['--evaluation',target,'--motions',motions,
                    '--expected-clips',18,'--output',target/'audit'],args.output/f'audit-{step:09d}.log',180)
                accounting = json.loads((target/'accounting.json').read_bytes())
                selection = json.loads((target/'protocol.json').read_bytes())['checkpoint_selection']
                assert selection['evaluated_collection_steps'] == step and selection['parent_training_complete'] is False
                row.update(status='complete',successes=accounting['successes'],failures=accounting['failures'],
                    mean_normalized_full_motion_return=accounting['mean_normalized_full_motion_return'])
            except Exception as error:
                row.update(status='failed',error=type(error).__name__+': '+str(error))
            originals += [target/name for name in ('result.json','protocol.json','accounting.json','audit/summary.json','audit/initial-states.json') if (target/name).is_file()]
            result['points'].append(row)
            write(args.output/'result.json',result)
            print('INTERRUPTED_CONTROL_POINT '+json.dumps(row),flush=True)
    finally:
        result['original_parent_metadata_unchanged'] = all(digest(Path(p))==value for p,value in original_meta_hashes.items())
        result['elapsed_seconds']=time.monotonic()-started
        result['status']='complete' if len(result['points'])==6 and all(p['status']=='complete' for p in result['points']) and result['original_parent_metadata_unchanged'] else 'incomplete'
        write(args.output/'result.json',result)
        originals.append(args.output/'result.json')
        print('INTERRUPTED_CONTROL_ORIGINALS_BEGIN',flush=True)
        subprocess.run([sys.executable,str(source/'emit_json_receipts.py'),'--max-bytes','4000000',*map(str,originals)],check=True,timeout=180)
        print('INTERRUPTED_CONTROL_ORIGINALS_END',flush=True)
    if result['status']!='complete':raise RuntimeError('At least one interrupted-snapshot evaluation is incomplete')
    print('INTERRUPTED_CONTROL_COMPLETED',flush=True)


if __name__ == '__main__':
    main()
