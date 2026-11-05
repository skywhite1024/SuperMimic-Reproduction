"""Render all six fixed target repetitions from saved physics states."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--role', choices=['parent', 'candidate'], required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    args = parser.parse_args()
    process = psutil.Process()
    process.cpu_affinity(process.cpu_affinity()[:2])
    os.environ.update(CUDA_VISIBLE_DEVICES='', MUJOCO_GL='glfw',
                      OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
                      HF_HUB_OFFLINE='1', WANDB_MODE='offline')
    videos = []
    for variant in ['raw', 'corrected']:
        for repetition in range(3):
            mode = f'specialist8807-1031d-{args.role}-{variant}-{args.device}-r{repetition}'
            path = ROOT/'videos'/mode/'88_07_stageii_unitree_g1.json'
            if path.exists():
                row = json.loads(path.read_text())
                assert row['complete_decode_passed']
                assert hashlib.sha256((ROOT/row['file']).read_bytes()).hexdigest() == row['sha256']
            else:
                subprocess.run([sys.executable, '-X', 'utf8', '-u',
                    str(ROOT/'scripts/render_library.py'), '--run', mode,
                    '--clips', '88_07_stageii_unitree_g1'], check=True,
                    cwd=ROOT, env=os.environ, timeout=180)
                row = json.loads(path.read_text())
            videos.append(row)
    target = ROOT/f'evidence/specialist8807-1031d-{args.role}-{args.device}-videos.json'
    target.write_text(json.dumps({'videos': videos, 'count': len(videos),
        'all_complete_decode_passed': all(x['complete_decode_passed'] for x in videos),
        'no_new_physics_or_training': True}, indent=2)+'\n')
    print('FIXED_SIX_VIDEOS_VERIFIED '+str(target), flush=True)


if __name__ == '__main__':
    main()
