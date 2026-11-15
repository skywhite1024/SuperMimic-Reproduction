"""Re-export completed PPO originals and the hash-bound BONES per-motion result.

Reads existing results only. Does not construct a simulator, load model weights,
train, evaluate, or change the original jobs' exit statuses.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path('/mnt/afs/260010076/SuperMimic/official-20261112/runs')
    ppo = root / 'ppo-sim-2048-1115c'
    bones = root / 'bones-warm-tail-1115d'
    output = args.output.resolve()
    assert output.parent == root and output.name == 'completed-result-recovery-1115h'
    output.mkdir(exist_ok=False)
    summary = json.loads((ppo / 'result.json').read_bytes())
    training = json.loads((ppo / 'training/result.json').read_bytes())
    assert summary['status'] == 'complete' and summary['quality_evaluation_completed']
    assert training['status'] == 'real_G1_simulator_PPO_finite_training_completed'
    assert training['ppo_update_calls'] == 2048 and training['simulator_steps'] == 402653184
    expected_training = '7f0dcfb3c9487275794bf216e6d2db3370f20db3eedaa0aaa18461d1f1919aa8'
    assert digest(ppo / 'training/result.json') == summary['training_result_sha256'] == expected_training
    assert [row['ppo_update'] for row in summary['points']] == [128, 256, 512, 1024, 1536, 2048]
    assert all(row['status'] == 'complete' for row in summary['points'])
    files = ('result.json', 'protocol.json', 'accounting.json', 'audit/summary.json', 'audit/initial-states.json')
    names = ['completed-gate-eval/evaluation/' + name for name in files]
    names += [f'eval-{step:06d}/evaluation/' + name for step in (128, 256, 512, 1024, 1536, 2048) for name in files]
    names += ['training/resolved-config.json', 'training/checkpoint-manifest.json', 'training/result.json', 'result.json']
    assert len(names) == 39
    paths = [ppo / name for name in names]
    for path in paths:
        assert path.resolve().is_relative_to(ppo) and path.suffix == '.json'
        assert 0 < path.stat().st_size <= 64_000_000
    raw = bones / 'evaluation/result.json'
    expected_raw = dict(path=str(raw), bytes=52068537,
                        sha256='c9c6cb513153668aa39b78ce08a025e931fe88843038e526399fae252882b7ee')
    parent = json.loads((bones / 'result.json').read_bytes())
    assert parent['full_result_original'] == expected_raw
    assert raw.stat().st_size == expected_raw['bytes'] and digest(raw) == expected_raw['sha256']
    report = dict(status='originals_preflight_passed', original_PPO_job_exit_code=1,
                  original_PPO_failure='Final receipt export exceeded the original 2 MB per-file bound',
                  PPO_training_rerun=False, PPO_evaluation_rerun=False,
                  original_files_modified=False, PPO_original_JSONs=39,
                  PPO_training_result_sha256=expected_training, BONES_full_result_original=expected_raw)
    write = lambda: (output / 'result.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    write()
    emitter = str(Path(__file__).with_name('emit_json_receipts.py'))
    print('PPO_SIM_CURVE_RECEIPTS_BEGIN', flush=True)
    subprocess.run([sys.executable, emitter, '--max-bytes', '64000000', *map(str, paths)], check=True)
    print('PPO_SIM_CURVE_RECEIPTS_END', flush=True)
    print('PPO_SIM_RESULT_REEXPORT_COMPLETED', flush=True)
    print('BONES_FULL_RESULT_ORIGINAL_BEGIN', flush=True)
    subprocess.run([sys.executable, emitter, '--max-bytes', '64000000', str(raw)], check=True)
    print('BONES_FULL_RESULT_ORIGINAL_END', flush=True)
    report['status'] = 'completed_existing_originals_reexported'
    write()
    print('COMPLETED_RESULT_RECOVERY_REPORT_BEGIN', flush=True)
    subprocess.run([sys.executable, emitter, str(output / 'result.json')], check=True)
    print('COMPLETED_RESULT_RECOVERY_REPORT_END', flush=True)
    print('COMPLETED_RESULT_RECOVERY_PIPELINE_COMPLETED', flush=True)


if __name__ == '__main__':
    main()
