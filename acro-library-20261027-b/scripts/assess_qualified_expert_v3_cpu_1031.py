"""Run the four matched CPU evaluations of the recovered oldw02 candidate; no updates."""
import gc
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    import psutil

    process = psutil.Process()
    process.cpu_affinity(process.cpu_affinity()[:2])
    os.environ.update(CUDA_VISIBLE_DEVICES='', MUJOCO_GL='disable',
                      OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                      WANDB_MODE='offline', HF_HUB_OFFLINE='1', PYTHONUTF8='1')
    receipt = ROOT / 'evidence/qualified-expert-v3-1031-CPU-assessment.json'
    assert not receipt.exists(), 'Preserve the existing assessment before another run'
    started = time.monotonic()
    result = {'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat(),
              'device': 'cpu', 'CPU_affinity': process.cpu_affinity(),
              'new_training_updates': 0, 'maximum_seconds': 1800,
              'candidate_adopted': False, 'full_goal_complete': False, 'commands': []}

    def save(status):
        result.update(status=status, wall_seconds=time.monotonic()-started,
                      updated_at=datetime.now(timezone.utc).isoformat())
        receipt.write_text(json.dumps(result, indent=2)+'\n')

    def run(script, arguments):
        index = len(result['commands'])
        log = ROOT / f'evidence/qualified-expert-v3-1031-CPU-{index:02d}.log'
        remaining = 1800-(time.monotonic()-started)
        assert remaining > 0
        print(json.dumps({'stage': script, 'mode': arguments[-1] if arguments else None,
                          'command': index}), flush=True)
        with log.open('x', encoding='utf8') as stream:
            subprocess.run([sys.executable, '-X', 'utf8', '-u', str(ROOT/'scripts'/script),
                            *arguments], cwd=ROOT, env=os.environ, stdout=stream,
                           stderr=subprocess.STDOUT, check=True, timeout=remaining)
        result['commands'].append({'script': script, 'arguments': arguments,
                                   'log': log.relative_to(ROOT).as_posix()})
        save('running')

    save('running')
    try:
        import torch
        full = torch.load(ROOT/'runs/unified-qualified-expert-v3-h100-1000/final-full.pt',
                          map_location='cpu', weights_only=True)
        checkpoint = ROOT/'policy/qualified-expert-v3-h100-1000.pt'
        exported = torch.load(checkpoint, map_location='cpu', weights_only=True)
        assert full['iteration'] == exported['iteration'] == 31000
        assert exported['pi']['obs_norm.mean'].numel() == 101
        assert set(full['pi']) == set(exported['pi'])
        assert all(torch.equal(value, exported['pi'][name]) for name, value in full['pi'].items())
        result['full_and_exported_policy_tensors_equal'] = True
        result['deployed_student_policies'] = 1
        del full, exported
        gc.collect()
        result['counts'] = {}
        for variant, suite, group in [('raw', 'library', 'new'),
                                      ('corrected', 'library', 'all'),
                                      ('raw', 'sequences', None),
                                      ('corrected', 'sequences', None)]:
            mode = f'qualified-expert-v3-{variant}-{suite}-cpu'
            assert not (ROOT/'runs'/mode).exists()
            argv = ['--role', 'candidate', '--suite', suite, '--reference-variant', variant,
                    '--checkpoint', str(checkpoint), '--device', 'cpu', '--mode', mode]
            if group:
                argv += ['--group', group]
            run('evaluate_qualified_expert_v3.py', argv)
            scripts = (['assess_tracking.py', 'audit_physical_contacts.py', 'combine_acceptance.py']
                       if suite == 'library' else ['assess_sequences.py', 'audit_sequence_contacts.py'])
            for script in scripts:
                run(script, ['--run', mode])
            name = 'combined-assessment.json' if suite == 'library' else 'continuous-assessment.json'
            assessment = json.loads((ROOT/'runs'/mode/name).read_text())
            if suite == 'library':
                result['counts'][variant+'_'+suite] = assessment['combined_engineering_passed']
            else:
                contacts = json.loads((ROOT/'runs'/mode/'continuous-contact-review.json').read_text())
                by_seed = {x['seed']: x for x in contacts['trials']}
                result.setdefault('sequence_tracking_counts', {})[variant] = assessment['continuous_engineering_passed']
                result.setdefault('sequence_contact_counts', {})[variant] = contacts['contact_review_passed']
                result['counts'][variant+'_'+suite] = sum(x['continuous_engineering_passed'] and by_seed[x['seed']]['contact_review_passed'] for x in assessment['trials'])
                result['sequence_counts_require_tracking_and_contact'] = True
            save('running')
            print(json.dumps({'completed': mode, 'count': result['counts'][variant+'_'+suite]}), flush=True)
        summary = json.loads((ROOT/'runs/qualified-expert-v3-corrected-library-cpu/summary.json').read_text())
        result['old_original_gate_passed'] = summary['groups']['lafan']['passed']
        save('all_four_matched_CPU_candidate_assessments_complete')
        print(json.dumps(result), flush=True)
    except BaseException as error:
        result['error'] = repr(error)
        save('failed_partial_evidence_retained')
        raise


if __name__ == '__main__':
    main()
