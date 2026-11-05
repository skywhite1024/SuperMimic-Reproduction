"""Finite matched CPU controls; no training, one fresh process per repetition."""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--role', choices=['parent', 'candidate'], required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    args = parser.parse_args()
    process = psutil.Process()
    process.cpu_affinity(process.cpu_affinity()[:2])
    os.environ.update(CUDA_VISIBLE_DEVICES='', MUJOCO_GL='disable',
                      OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
                      WANDB_MODE='offline', HF_HUB_OFFLINE='1', PYTHONUTF8='1')
    receipt = ROOT / f'evidence/specialist8807-1031d-{args.role}-CPU-assessment.json'
    assert not receipt.exists(), 'Preserve the existing fixed repetitions'
    started = time.monotonic()
    result = {'status': 'running', 'role': args.role, 'device': 'cpu',
              'started_at': datetime.now(timezone.utc).isoformat(),
              'CPU_affinity': process.cpu_affinity(), 'maximum_seconds': 900,
              'new_training_updates': 0, 'repetitions': [], 'commands': [],
              'independent99D_specialist_not_unified101D': True,
              'full_goal_complete': False}

    def save(status):
        result.update(status=status, wall_seconds=time.monotonic()-started,
                      updated_at=datetime.now(timezone.utc).isoformat())
        receipt.write_text(json.dumps(result, indent=2)+'\n')

    def run(script, argv, cap):
        index = len(result['commands'])
        log = ROOT / f'evidence/specialist8807-1031d-{args.role}-CPU-{index:02d}.log'
        remaining = 900-(time.monotonic()-started)
        assert remaining > 0
        with log.open('x', encoding='utf8') as stream:
            subprocess.run([sys.executable, '-X', 'utf8', '-u', str(ROOT/'scripts'/script),
                            *argv], cwd=ROOT, env=os.environ, stdout=stream,
                           stderr=subprocess.STDOUT, check=True,
                           timeout=min(cap, remaining))
        result['commands'].append({'script': script, 'arguments': argv,
                                   'log': log.relative_to(ROOT).as_posix()})
        save('running')

    save('running')
    try:
        for variant in ['raw', 'corrected']:
            for repetition in range(3):
                mode = f'specialist8807-1031d-{args.role}-{variant}-cpu-r{repetition}'
                assert not (ROOT/'runs'/mode).exists()
                run('evaluate_specialist8807_1031.py', [
                    '--role', args.role, '--reference-variant', variant,
                    '--checkpoint', str(args.checkpoint.resolve()), '--mode', mode,
                    '--device', 'cpu', '--group', 'new',
                    '--clips', '88_07_stageii_unitree_g1'], 180)
                for script, cap in [('assess_tracking.py', 30),
                                     ('audit_physical_contacts.py', 45),
                                     ('assess_specialist8807_1031.py', 15)]:
                    run(script, ['--run', mode], cap)
                row = json.loads((ROOT/'runs'/mode/'specialist-assessment.json').read_text())
                row.update(repetition=repetition, seed=0)
                result['repetitions'].append(row)
                save('fixed_repetition_complete')
                print(json.dumps({'mode': mode, 'strict_passed': row['strict_passed'],
                                  'completed': len(result['repetitions'])}), flush=True)
        result['counts'] = {variant: sum(x['strict_passed'] for x in result['repetitions']
            if x['reference_variant'] == ('raw' if variant == 'raw' else 'ground-calibration-v1'))
            for variant in ['raw', 'corrected']}
        save('all_six_matched_CPU_fixed_repetitions_complete')
    except BaseException as error:
        result['error'] = repr(error)
        save('failed_partial_evidence_retained')
        raise


if __name__ == '__main__':
    main()
