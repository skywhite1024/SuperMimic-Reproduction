"""Add only missing V2 raw/old failure views from retained CUDA trajectories."""
import json, os, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
import psutil

ROOT = Path(__file__).resolve().parents[1]
process = psutil.Process()
process.cpu_affinity(process.cpu_affinity()[:2])
env = os.environ.copy()
env.update(MUJOCO_GL='glfw', PYTHONUTF8='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', WANDB_MODE='offline', HF_HUB_OFFLINE='1')
started = time.monotonic()
receipt = ROOT/'evidence/qualified-expert-v2-1031-supplementary-failure-videos.json'
result = {'status': 'rendering_missing_failure_views', 'started_at': datetime.now(timezone.utc).isoformat(), 'new_training_updates': 0, 'inference_runs': 0, 'source': 'retained V2 CUDA actual/reference trajectories', 'original_21_videos_preserved': True, 'phases': []}
receipt.write_text(json.dumps(result, indent=2)+'\n')
for variant, old in [('raw', False), ('corrected', True)]:
    run = f'qualified-expert-v2-{variant}-library-cuda'
    directory = ROOT/'runs'/run
    if old:
        clips = [x['clip'] for x in json.loads((directory/'summary.json').read_text())['clips'] if x['group']=='lafan' and not x['passed_original_hard_gate']]
    else:
        clips = [x['clip'] for x in json.loads((directory/'combined-assessment.json').read_text())['clips'] if not x['combined_engineering_passed']]
    label = 'qualified-expert-v2-1031-'+('old-failures-cuda' if old else 'raw-failures-cuda')
    argv = [sys.executable, '-X', 'utf8', '-u', str(ROOT/'scripts/render_library.py'), '--run', run, '--clips', *clips, '--video-label', label]
    if old:
        argv += ['--include-old']
    log = ROOT/'evidence'/f'{label}.log'
    with log.open('x', encoding='utf8') as output:
        subprocess.run(argv, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT, check=True, timeout=1800)
    videos = [json.loads((ROOT/'videos'/label/f'{clip}.json').read_text()) for clip in clips]
    assert all(x['complete_decode_passed'] and x['only_retained_actual_physics_states_rendered'] for x in videos)
    result['phases'].append({'run': run, 'label': label, 'clips': clips, 'videos': videos})
    receipt.write_text(json.dumps(result, indent=2)+'\n')
    print('FAILURE_VIEWS_PHASE '+json.dumps({'label': label, 'complete_videos': len(videos)}), flush=True)
result.update(status='missing_failure_views_rendered_and_fully_decoded', wall_seconds=time.monotonic()-started, updated_at=datetime.now(timezone.utc).isoformat())
receipt.write_text(json.dumps(result, indent=2)+'\n')
print('FAILURE_VIEWS_COMPLETE '+json.dumps({'videos': sum(len(x['videos']) for x in result['phases']), 'updates': 0}), flush=True)
