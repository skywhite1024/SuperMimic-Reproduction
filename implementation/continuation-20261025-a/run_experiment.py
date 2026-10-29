"""A finite single-H100 continuation, staged evaluations, diagnostics and videos."""
from __future__ import annotations
import datetime as dt
import json
import os
from pathlib import Path
import sys
import time
from audit_common import file_hash, write_json

OUT = Path(__file__).resolve().parent
BASE = Path(os.environ['SUPERMIMIC_BASE']).resolve()
sys.path.insert(0, str(BASE))
import run_gpu_v3 as monitor
monitor.ROOT = OUT
monitor.SOURCE = BASE / 'uni'


def main():
    started = time.monotonic()
    deadline = started + 10200
    state = {'started_at': dt.datetime.now(dt.timezone.utc).isoformat(), 'status': 'running',
             'gpus': 1, 'program_budget_seconds': 10200, 'new_experiments_gpu_hours_cap': 6,
             'community_target': {'macro_at_least': 0.82, 'failed': 0}}
    write_json(OUT / 'experiment.json', state)
    try:
        import torch
        from preflight import check
        current = check()
        previous = json.loads((OUT / 'evidence/preflight.json').read_text())
        assert current['state_hashes'] == previous['state_hashes']
        assert current['parent_sha256'] == previous['parent_sha256']
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1 and 'H100' in torch.cuda.get_device_name(0)
        current.update(GPU=torch.cuda.get_device_name(0), cgroup_cpu_max=monitor.read_cgroup('cpu.max'),
                       cgroup_memory_max=monitor.read_cgroup('memory.max'), resources=monitor.resources())
        write_json(OUT / 'evidence/gpu-preflight.json', current)
        print('CONTINUATION_GPU_PREFLIGHT_PASS ' + json.dumps({'parent_iteration': current['iteration'], 'GPU': current['GPU'], 'lr': current['lr_at_first_continuation_iteration']}), flush=True)
        monitor.run('training', [str(OUT / 'train_continuation.py')], deadline, 8200)
        proof = json.loads((OUT / 'evidence/continuation-proof.json').read_text())
        rows = [r for r in proof['evaluations'] if not r['confirmation']]
        assert rows
        best = min(rows, key=lambda r: (r['n_failed'], -r['macro'], -r['iteration']))
        highest = max(rows, key=lambda r: r['macro'])
        final = max(rows, key=lambda r: r['iteration'])
        state.update(best=best, highest_macro=highest, final=final,
                     best_rule='fewest failed clips,then highest macro return,then latest iteration',
                     target_confirmed=proof['finish_reason'] == 'confirmed_community_target_met')
        write_json(OUT / 'selection.json', {k: state[k] for k in ['best', 'highest_macro', 'final', 'best_rule', 'target_confirmed']})
        for name, checkpoint, evaluation in [
            ('parent-012000', BASE/'runs/baseline-seed0/ckpt_012000.pt', BASE/'runs/baseline-evaluation.json'),
            ('selected', Path(best['checkpoint']), Path(best['result']))]:
            monitor.run('diagnostic-' + name, [str(OUT / 'diagnose.py'), '--checkpoint', str(checkpoint),
                '--evaluation', str(evaluation), '--output', str(OUT/'diagnostics'/name)], deadline, 1000)
        os.environ['MUJOCO_GL'] = 'osmesa'
        os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
        original = json.loads((BASE/'runs/baseline-evaluation.json').read_text())
        selected = json.loads(Path(best['result']).read_text())
        for i, clip in enumerate(['dance2_subject4', 'fallAndGetUp1_subject5', 'fallAndGetUp2_subject3']):
            row = selected['per_clip'][clip]
            focus = row['steps']*0.02 if row['failed'] else original['per_clip'][clip]['steps']*0.02 + 5
            start = max(0, focus - 10)
            monitor.run(f'video-{i}', ['-m', 'unisupermimic.tasks.motion_tracking.g1.supermimic.render',
                '--backend', 'mjwarp', '--ckpt', best['checkpoint'], '--clip', clip, '--start', str(start),
                '--seconds', '15', '--global-ghost', '--output', str(OUT/'videos'/(clip+'.mp4'))], deadline, 420)
        state.update(status='complete', final_iteration=proof['after']['iteration'])
        print('CONTINUATION_EXPERIMENT_COMPLETE ' + json.dumps({'target_confirmed': state['target_confirmed'],
            'best_iteration': best['iteration'], 'best_macro': best['macro'], 'best_failed': best['n_failed'],
            'final_iteration': final['iteration'], 'final_macro': final['macro'], 'final_failed': final['n_failed']}), flush=True)
    except BaseException as exc:
        state.update(status='failed_or_interrupted', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        state.update(ended_at=dt.datetime.now(dt.timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-started,
                     resource_release_verified=False)
        write_json(OUT / 'experiment.json', state)


if __name__ == '__main__': main()
