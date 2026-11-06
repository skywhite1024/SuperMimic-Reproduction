"""One finite single-H100 main matrix, with short progress and complete JSON exports."""
import argparse
import base64
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from study_runtime import ROOT, STUDY, plan, sha


def export_json(label, paths, base):
    """Keep each log record below the console's 500-character Body limit."""
    payload = {'files': {p.relative_to(base).as_posix(): json.loads(p.read_text()) for p in paths},
               'file_sha256': {p.relative_to(base).as_posix(): sha(p) for p in paths}}
    packed = gzip.compress(json.dumps(payload, sort_keys=True).encode(), mtime=0)
    encoded = base64.b64encode(packed).decode()
    chunks = [encoded[i:i+350] for i in range(0, len(encoded), 350)]
    digest = hashlib.sha256(packed).hexdigest()
    print('REPLAY_EXPORT_META '+json.dumps({'label': label, 'chunks': len(chunks),
          'sha256': digest, 'bytes': len(packed)}), flush=True)
    for i, chunk in enumerate(chunks):
        print(f'REPLAY_EXPORT_CHUNK {label} {i:04d} {chunk}', flush=True)
    print(f'REPLAY_EXPORT_DONE {label} {digest}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot-root', type=Path, required=True)
    p.add_argument('--expected-commit', required=True)
    p.add_argument('--label', required=True)
    a = p.parse_args(); pilot = a.pilot_root.resolve(); spec = plan()
    personal = Path('/mnt/afs/260010076/SuperMimic').resolve()
    assert ROOT.resolve().is_relative_to(personal) and pilot.is_relative_to(personal)
    assert spec['main_budget_frozen'] and spec['provisional_main']['steps'] == 2000
    output = STUDY/'runs'/f'main-block-{a.label}'
    assert output.resolve().is_relative_to((STUDY/'runs').resolve()) and not output.exists()
    output.mkdir(parents=True)
    started = time.perf_counter()
    state = {'status': 'preparing', 'source_commit': a.expected_commit,
             'plan_sha256': sha(STUDY/'plan.json'), 'label': a.label, 'stages': [], 'runs': []}
    def save():
        state['wall_seconds'] = time.perf_counter()-started
        (output/'receipt.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
    def run(tag, command, maximum, progress=None):
        state['stage'] = tag; save()
        before = time.perf_counter(); last_iteration = -1
        print('MAIN_STAGE_STARTED '+json.dumps({'stage': tag, 'max_seconds': maximum}), flush=True)
        log = output/f'{tag}.log'
        with log.open('x', encoding='utf-8') as stream:
            child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            try:
                while True:
                    try:
                        code = child.wait(timeout=20)
                        if code: raise subprocess.CalledProcessError(code, command)
                        break
                    except subprocess.TimeoutExpired:
                        if time.perf_counter()-before > maximum:
                            raise TimeoutError(f'Finite stage deadline: {tag}')
                        if progress and progress.exists():
                            rows = progress.read_text().splitlines()
                            if rows:
                                iteration = int(rows[-1].split(',')[0])
                                if iteration-last_iteration >= 100 or last_iteration == -1:
                                    last_iteration = iteration
                                    print('MAIN_TRAIN_PROGRESS '+json.dumps({'run': tag,
                                          'new_updates': iteration-16000,
                                          'elapsed_seconds': round(time.perf_counter()-before, 1)}), flush=True)
                        if tag == 'full55-matrix':
                            receipt = STUDY/'runs'/f'evaluation-{a.label}'/'receipt.json'
                            if receipt.exists():
                                completed = len(json.loads(receipt.read_text())['cases'])
                                if completed != last_iteration:
                                    last_iteration = completed
                                    print('MAIN_EVALUATION_PROGRESS '+json.dumps({'completed': completed, 'total': 13}), flush=True)
            except BaseException:
                child.terminate()
                try: child.wait(timeout=30)
                except subprocess.TimeoutExpired: child.kill(); child.wait(timeout=15)
                print('MAIN_STAGE_LOG_TAIL '+json.dumps(log.read_text(errors='replace').splitlines()[-4:])[:440], flush=True)
                raise
        state['stages'].append({'stage': tag, 'wall_seconds': time.perf_counter()-before}); save()
        print('MAIN_STAGE_COMPLETED '+json.dumps(state['stages'][-1]), flush=True)
    def stage(relative, expected):
        source = pilot/relative; target = ROOT/relative
        assert source.resolve().is_relative_to(pilot) and target.resolve().is_relative_to(ROOT.resolve())
        assert sha(source) == expected, f'Pilot asset mismatch: {relative}'
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            try: os.link(source, target)
            except OSError: shutil.copyfile(source, target)
        assert sha(target) == expected
    try:
        assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == a.expected_commit
        subprocess.run(['git', 'diff', '--exit-code', 'HEAD'], cwd=ROOT, check=True)
        pilot_study = pilot/'experiments/replay-study'
        pilot_receipt = pilot_study/'runs/pilot-block-pilot-c/receipt.json'
        pilot_state = json.loads(pilot_receipt.read_text())
        assert pilot_state['status'] == 'pilot_complete_not_main_results'
        assert pilot_state['source_commit'] == 'cfd02f3516b7ddbf574cdfcc594caa451c0318cb'
        assert sha(pilot_study/'plan.json') == '460c6a5499707cce8f6c66f17f001069d2dba05da392ce51391028d5a82b2361'
        pilot_files = [pilot_study/'plan.json', pilot_study/'inputs.lock.json', pilot_receipt,
            pilot_receipt.with_name('wm-diagnostics.json'), pilot_study/'runs/frozen-probe-cuda/metadata.json',
            pilot/'acro-library-20261027-b/runs/replay-study-pilot-pilot-c-baseline-cuda/summary.json']
        measurements = []
        for rho in spec['old_replay_fractions']:
            directory = pilot_study/'runs'/f'pilot-c-rho{round(rho*100):03d}-seed100'
            proof = json.loads((directory/'proof.json').read_text())
            assert proof['new_updates'] == 100 and proof['device'] == 'cuda' and not proof['scientific_main_result']
            assert proof['normalizer_tensors_unchanged'] and proof['variant']['old_replay_fraction'] == rho
            # A fixed abort gate, never an outcome-dependent change to the 2,000-update budget.
            assert proof['wall_seconds'] < 180 and proof['peak_cuda_reserved_bytes'] < 70*1024**3
            measurements.append({key: proof[key] for key in ['wall_seconds', 'peak_cuda_allocated_bytes', 'peak_cuda_reserved_bytes']})
            pilot_files += [directory/'proof.json', directory/'run.json']
        export_json('pilot-c', pilot_files, pilot)
        state['pilot_timing_memory'] = measurements
        state['budget_gate'] = 'all four 100-update pilots under 180s and 70GiB reserved; main fixed at 2000/200'
        inputs = json.loads((STUDY/'inputs.lock.json').read_text())
        for relative, expected in inputs['motion_sha256'].items(): stage(relative, expected)
        assets = json.loads((ROOT/'implementation/evidence/asset-downloads.json').read_text())
        for row in assets['files']:
            relative = 'implementation/uni/'+row['path']
            if relative not in inputs['motion_sha256']: stage(relative, row['sha256'])
        stage(spec['parent_path'], spec['parent_sha256'])
        probe_metadata = json.loads((pilot_study/'runs/frozen-probe-cuda/metadata.json').read_text())
        assert probe_metadata['backend'] == 'mjwarp' and probe_metadata['window_count'] == 440
        assert probe_metadata['parent_sha256'] == spec['parent_sha256']
        assert probe_metadata['motion_sha256'] == inputs['motion_sha256']
        stage('experiments/replay-study/runs/frozen-probe-cuda/windows.npz', probe_metadata['data_sha256'])
        stage('experiments/replay-study/runs/frozen-probe-cuda/metadata.json', sha(pilot_study/'runs/frozen-probe-cuda/metadata.json'))
        import torch
        assert torch.cuda.device_count() == 1 and 'H100' in torch.cuda.get_device_name(0)
        packages = {}
        for name in ['torch', 'mujoco', 'unilab', 'warp-lang']:
            try: packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError: packages[name] = None
        state.update(status='running', gpu=torch.cuda.get_device_name(0), python=sys.version,
            packages=packages)
        save(); export_json('main-start', [output/'receipt.json', STUDY/'plan.json'], ROOT)
        py = sys.executable; checkpoints = [ROOT/spec['parent_path']]
        for seed in spec['training_seeds']:
            for rho in spec['old_replay_fractions']:
                label = f'{a.label}-rho{round(rho*100):03d}-seed{seed}'
                directory = STUDY/'runs'/label
                run(label, [py, str(STUDY/'train.py'), '--phase', 'main', '--device', 'cuda',
                    '--label', label, '--old-fraction', str(rho), '--seed', str(seed)],
                    spec['provisional_main']['max_seconds']+120, directory/'train_log.csv')
                proof = json.loads((directory/'proof.json').read_text())
                assert proof['scientific_main_result'] and proof['new_updates'] == spec['provisional_main']['steps']
                assert proof['variant']['plan_sha256'] == state['plan_sha256']
                checkpoint = directory/'final-full.pt'; assert sha(checkpoint) == proof['checkpoint_sha256']
                checkpoints.append(checkpoint); state['runs'].append({'label':label, 'checkpoint_sha256':proof['checkpoint_sha256']})
                save(); export_json(label, [directory/'proof.json', directory/'run.json'], ROOT)
        run('full55-matrix', [py, str(STUDY/'evaluate_matrix.py'), '--execute', '--prefix', a.label,
            '--label', a.label], 13*1200+180)
        evaluation = STUDY/'runs'/f'evaluation-{a.label}'/'receipt.json'
        evaluation_state = json.loads(evaluation.read_text())
        export_json('full55', [evaluation, *[ROOT/c['summary'] for c in evaluation_state['cases']]], ROOT)
        run('wm-diagnostics', [py, str(STUDY/'wm_probe.py'), '--probe', str(STUDY/'runs/frozen-probe-cuda/windows.npz'),
            '--checkpoints', *[str(c) for c in checkpoints], '--output', str(output/'wm-diagnostics.json')], 900)
        run('analysis', [py, str(STUDY/'summarize_main.py'), '--receipt', str(evaluation), '--prefix', a.label,
            '--output', str(STUDY/'runs'/f'analysis-{a.label}')], 180)
        state['status'] = 'complete_12_main_runs_13_full55_evaluations'; save()
        export_json('main-final', [output/'receipt.json', output/'wm-diagnostics.json',
            STUDY/'runs'/f'analysis-{a.label}'/'analysis.json'], ROOT)
        print('REPLAY_MAIN_COMPLETE '+json.dumps({'runs':12,'evaluations':13,'wall_seconds':state['wall_seconds']}), flush=True)
    except BaseException as error:
        state.update(status='failed_preserved', error=repr(error)); save()
        export_json('main-failure', [output/'receipt.json'], ROOT)
        raise


if __name__ == '__main__': main()
