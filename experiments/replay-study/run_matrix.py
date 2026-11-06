"""Print the immutable 12-condition matrix, or run a finite engineering/pilot block."""
import argparse
import json
import subprocess
import sys
import time
from study_runtime import STUDY, plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--phase', choices=['engineering','pilot','main'], default='engineering')
    p.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    p.add_argument('--parent')
    p.add_argument('--seeds', type=int, nargs='+')
    p.add_argument('--prefix', default='engineering-v1')
    args = p.parse_args(); spec = plan()
    cases = [{'old_fraction': rho, 'seed': seed} for seed in spec['training_seeds'] for rho in spec['old_replay_fractions']]
    if not args.execute:
        print(json.dumps(cases, indent=2)); return
    if args.phase == 'main':
        assert spec['main_budget_frozen'], 'GPU pilot and common budget must be finalized first'
    seeds = args.seeds if args.seeds is not None else (spec['training_seeds'] if args.phase == 'main' else [0])
    assert len(seeds) == len(set(seeds)) and all(seed in spec['training_seeds'] for seed in seeds)
    selected = [case for case in cases if case['seed'] in seeds]
    if args.phase == 'pilot':
        selected = [{'old_fraction': rho, 'seed': spec['pilot']['seed']} for rho in spec['old_replay_fractions']]
    started = time.perf_counter()
    for case in selected:
        label = f"{args.prefix}-rho{round(case['old_fraction']*100):03d}-seed{case['seed']}"
        command = [sys.executable, str(STUDY/'train.py'), '--phase', args.phase, '--device', args.device,
                   '--label', label, '--old-fraction', str(case['old_fraction']), '--seed', str(case['seed'])]
        if args.parent: command += ['--parent', args.parent]
        maximum = 600 if args.phase == 'engineering' else spec['pilot' if args.phase == 'pilot' else 'provisional_main']['max_seconds']
        if args.device == 'cuda' and time.perf_counter()-started+maximum > spec['total_gpu_hour_ceiling']*3600:
            raise TimeoutError('Insufficient remaining aggregate GPU budget for this finite run')
        log_dir = STUDY/'runs/_logs'; log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir/f'{label}.log'
        print('CASE_STARTED '+json.dumps({'label': label, 'condition': case, 'phase': args.phase}), flush=True)
        try:
            with log_path.open('x', encoding='utf-8') as stream:
                subprocess.run(command, check=True, timeout=maximum+120, stdout=stream, stderr=subprocess.STDOUT)
        except BaseException:
            print('\n'.join(log_path.read_text(encoding='utf-8', errors='replace').splitlines()[-16:]), flush=True)
            raise
        for line in log_path.read_text(encoding='utf-8', errors='replace').splitlines():
            if line.startswith('REPLAY_STUDY_VERIFIED '): print(line, flush=True)


if __name__ == '__main__':
    main()
