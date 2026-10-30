"""Assess and render a verified finite retention candidate using the frozen contracts."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT_CPU = {
    'corrected': 'stage3-actionbound20-corrected-cpu-all55',
    'raw': 'stage3-actionbound20-raw-cpu-new15',
    'raw_sequences': 'stage3-actionbound20-frozen-continuous-v1',
    'corrected_sequences': 'stage3-actionbound20-corrected-continuous-v1',
}
CANDIDATE_CPU = {
    'corrected': 'retention-v1-corrected-cpu-all55',
    'raw': 'retention-v1-raw-cpu-new15',
    'raw_sequences': 'retention-v1-raw-continuous-cpu',
    'corrected_sequences': 'retention-v1-corrected-continuous-cpu',
}


def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf8'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def cuda_modes(parent=False):
    prefix = 'retention-v1-parent-bound20' if parent else 'retention-v1'
    return [prefix + '-corrected-cuda-all55', prefix + '-raw-cuda-new15',
            prefix + '-raw-continuous-cuda', prefix + '-corrected-continuous-cuda']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['assess', 'render'], required=True)
    args = parser.parse_args()
    started = time.monotonic()
    budget = 7200 if args.phase == 'assess' else 14400
    transfer = read('evidence/local-transfer-verification-retention-v1-1000-resume-b.json')
    assert transfer['status'] == 'archive_and_all_manifest_files_verified' and transfer['program_complete']
    cloud = read('evidence/gpu-retention-v1-1000-resume1030b.json')
    assert cloud['status'] == 'finite_candidate_and_all_cuda_assessments_complete'
    resources = read('evidence/retention-resume1030b-final-resource-verification.json')
    assert resources['GPU_job_terminal'] and resources['CPU_CCI_stopped_and_empty'] and resources['DNAT_unbound']
    frozen = read('retention-trial-v1-plan.json')
    for name, digest in frozen['files'].items():
        assert sha(ROOT / name) == digest, name
    checkpoint = ROOT / 'policy/unified-retention-v1-h100-1000.pt'
    assert sha(checkpoint) == cloud['policy_sha256']
    full_sha, policy_sha = cloud['full_checkpoint_sha256'], cloud['policy_sha256']
    invocation = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    receipt = ROOT / f'evidence/retention-1030-{args.phase}-{invocation}.json'
    completed = []

    def record(status, **extra):
        value = {'status': status, 'phase': args.phase, 'updated_at': datetime.now(timezone.utc).isoformat(),
                 'policy_sha256': policy_sha, 'commands_completed': completed, 'full_goal_complete': False, **extra}
        receipt.write_text(json.dumps(value, indent=2) + '\n', encoding='utf8')

    def run(script, argv, artifact=None, render=False):
        if artifact and (ROOT / artifact).exists():
            return
        remaining = budget - (time.monotonic() - started)
        assert remaining > 0, 'Finite local phase budget exhausted'
        environment = os.environ.copy()
        environment.update(PYTHONUTF8='1', MUJOCO_GL='glfw' if render else 'disable',
                           OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', HF_HUB_OFFLINE='1', WANDB_MODE='offline')
        log = ROOT / f'evidence/retention-local-{invocation}-{len(completed):03d}-{script[:-3]}.log'
        command = [sys.executable, '-X', 'utf8', '-u', str(ROOT / 'scripts' / script), *argv]
        print('RETENTION_LOCAL_START ' + json.dumps({'script': script, 'arguments': argv}), flush=True)
        with log.open('x', encoding='utf8') as stream:
            subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT,
                           check=True, timeout=min(2400 if not render else 1800, remaining))
        if artifact:
            assert (ROOT / artifact).exists(), artifact
        completed.append({'script': script, 'arguments': argv, 'log': str(log.relative_to(ROOT))})
        record('running')
        print('RETENTION_LOCAL_END ' + json.dumps({'script': script, 'artifact': artifact}), flush=True)

    def validate(mode, digest, continuous=False):
        summary = read(f'runs/{mode}/summary.json')
        assert summary['checkpoint_sha256'] == digest, mode
        assert summary['post_initialization_resets'] == 0
        if continuous:
            assert len(summary['trials']) == 3
            assert all(row['post_initialization_resets'] == 0 for row in summary['trials'])
        else:
            assert len(summary['clips']) == (55 if 'all55' in mode else 15)
            assert summary['groups']['new']['denominator'] == 15
            if 'all55' in mode:
                assert summary['groups']['lafan']['denominator'] == 40
        return summary

    record('running')
    try:
        for parent in [True, False]:
            for mode in cuda_modes(parent):
                validate(mode, frozen['initial_full_sha256'] if parent else full_sha, 'continuous' in mode)
        if args.phase == 'assess':
            # Existing parent CPU tests use the exported stage3 policy. Confirm exact tensor identity
            # with the full checkpoint reused by CUDA, before reusing those four evaluations.
            import torch
            full = torch.load(ROOT / frozen['initial_full'], map_location='cpu', weights_only=True)
            parent_policy = ROOT / 'policy/unified-h100-stage3-4000.pt'
            parent = torch.load(parent_policy, map_location='cpu', weights_only=True)
            assert full['pi'].keys() == parent['pi'].keys()
            assert all(torch.equal(full['pi'][key], parent['pi'][key]) for key in full['pi'])
            parent_sha = sha(parent_policy)
            for key, mode in PARENT_CPU.items():
                summary = validate(mode, parent_sha, 'sequences' in key)
                assert summary['controller_variant'] == 'action-output-bound-20-v1'
            parent_reuse = {'status': 'exact_parent_policy_tensor_identity_and_all_four_CPU_runs_verified',
                            'parent_full_sha256': frozen['initial_full_sha256'], 'exported_policy_sha256': parent_sha,
                            'modes': PARENT_CPU, 'no_CPU_resimulation': True}
            (ROOT / 'evidence/retention-parent-CPU-reuse.json').write_text(json.dumps(parent_reuse, indent=2) + '\n')
            del full, parent
            for key in ['corrected', 'raw', 'raw_sequences', 'corrected_sequences']:
                continuous = 'sequences' in key
                variant = key.replace('_sequences', '')
                mode = CANDIDATE_CPU[key]
                argv = ['--suite', 'sequences' if continuous else 'library', '--reference-variant', variant,
                        '--checkpoint', str(checkpoint), '--mode', mode, '--device', 'cpu']
                if not continuous:
                    argv += ['--group', 'all' if variant == 'corrected' else 'new']
                run('evaluate_declared_bound.py', argv, f'runs/{mode}/summary.json')
                summary = validate(mode, policy_sha, continuous)
                assert summary['device'] == 'cpu' and summary['controller_variant'] == 'consistent-output-bound-20-retention-v1'
                if continuous:
                    for script, name in [('assess_sequences.py', 'continuous-assessment.json'),
                                         ('audit_sequence_contacts.py', 'continuous-contact-review.json')]:
                        run(script, ['--run', mode], f'runs/{mode}/{name}')
                else:
                    for script, name in [('assess_tracking.py', 'tracking-assessment.json'),
                                         ('audit_physical_contacts.py', 'contact-review.json'),
                                         ('combine_acceptance.py', 'combined-assessment.json')]:
                        run(script, ['--run', mode], f'runs/{mode}/{name}')
            baseline = read('runs/old-16k-all55-v3/summary.json')
            cpu = read(f'runs/{CANDIDATE_CPU["corrected"]}/summary.json')
            gpu = read('runs/retention-v1-corrected-cuda-all55/summary.json')
            limits = read('acceptance-contract.json')['old_lafan_regression']
            assert cpu['device'] == baseline['device'] == 'cpu' and cpu['backend'] == baseline['backend']
            old = cpu['groups']['lafan']
            cpu_old_ok = 40 - old['passed'] <= limits['hard_failed_clips_max'] and baseline['groups']['lafan']['macro_return_original'] - old['macro_return_original'] <= limits['macro_return_drop_from_matched_original_16k_max']
            groups = {}
            for device, mapping in [('cuda', {'corrected': 'retention-v1-corrected-cuda-all55', 'raw': 'retention-v1-raw-cuda-new15',
                                              'raw_sequences': 'retention-v1-raw-continuous-cuda', 'corrected_sequences': 'retention-v1-corrected-continuous-cuda'}),
                                    ('cpu', CANDIDATE_CPU)]:
                for key, mode in mapping.items():
                    quality = read(f'runs/{mode}/' + ('continuous-assessment.json' if 'sequences' in key else 'combined-assessment.json'))
                    groups[device + '_' + key] = quality.get('continuous_engineering_passed', quality.get('combined_engineering_passed'))
            result = {'status': 'all_matched_CPU_CUDA_assessments_complete', 'candidate_classification': cloud['classification'],
                      'new_denominator': 15, 'old_denominator': 40, 'continuous_denominator': 3,
                      'counts': groups, 'CPU_old_regression_passed': cpu_old_ok, 'CPU_old_group': old,
                      'CUDA_old_group': gpu['groups']['lafan'], 'CUDA_old_all40_passed': gpu['groups']['lafan']['passed'] == 40,
                      'all_failed_tails_retained': True, 'training_updates': 1000, 'no_automatic_extension': True,
                      'raw_reference_goal_complete': cpu_old_ok and gpu['groups']['lafan']['passed'] == 40 and
                          all(groups[f'{device}_raw'] == 15 and groups[f'{device}_raw_sequences'] == 3 for device in ['cpu', 'cuda']),
                      'corrected_variant_goal_complete': cpu_old_ok and gpu['groups']['lafan']['passed'] == 40 and
                          all(groups[f'{device}_corrected'] == 15 and groups[f'{device}_corrected_sequences'] == 3 for device in ['cpu', 'cuda']),
                      'video_generation_pending': True, 'paper_equivalent_reproduction_claimed': False}
            (ROOT / 'evidence/retention-1030-physical-assessment.json').write_text(json.dumps(result, indent=2) + '\n')
            record('all_matched_CPU_CUDA_assessments_complete', results=result)
            print('RETENTION_PHYSICAL_ASSESSMENT_COMPLETE ' + json.dumps(result), flush=True)
        else:
            assessment = read('evidence/retention-1030-physical-assessment.json')
            render_modes = cuda_modes(True) + cuda_modes(False) + list(CANDIDATE_CPU.values())
            for mode in render_modes:
                summary = read(f'runs/{mode}/summary.json')
                names = [f'seed-{row["seed"]}' for row in summary['trials']] if 'trials' in summary else [row['clip'] for row in summary['clips'] if row['group'] == 'new']
                for name in names:
                    run('render_library.py', ['--run', mode, '--clips', name], f'videos/{mode}/{name}.json', render=True)
            for mode in [cuda_modes(True)[0], cuda_modes(False)[0], CANDIDATE_CPU['corrected']]:
                for row in read(f'runs/{mode}/summary.json')['clips']:
                    if row['group'] == 'lafan' and not row['passed_original_hard_gate']:
                        run('render_library.py', ['--run', mode, '--clips', row['clip'], '--include-old'],
                            f'videos/{mode}/{row["clip"]}.json', render=True)
            verified = []
            for mode in render_modes:
                for path in (ROOT / 'videos' / mode).glob('*.json'):
                    video = json.loads(path.read_text())
                    assert video['complete_decode_passed'] and (ROOT / video['file']).stat().st_size == video['bytes']
                    assert sha(ROOT / video['file']) == video['sha256']
                    verified.append(str(path.relative_to(ROOT)))
            result = {'status': 'all_requested_full_tail_videos_decoded_and_hashed', 'video_count': len(verified),
                      'receipts': verified, 'all_full_120_second_sequences_retained': True}
            (ROOT / 'evidence/retention-1030-video-verification.json').write_text(json.dumps(result, indent=2) + '\n')
            assessment['video_generation_pending'] = False
            (ROOT / 'evidence/retention-1030-physical-assessment.json').write_text(json.dumps(assessment, indent=2) + '\n')
            record('all_requested_full_tail_videos_decoded_and_hashed', results=result)
            print('RETENTION_VIDEOS_COMPLETE ' + json.dumps(result), flush=True)
    except BaseException as error:
        record('failed_or_interrupted', error=repr(error))
        raise


if __name__ == '__main__':
    main()
