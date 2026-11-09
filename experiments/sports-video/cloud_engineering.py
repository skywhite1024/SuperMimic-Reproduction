"""Finite CUDA interface check on existing motions, never a sports result."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
AFS = Path('/mnt/afs/260010076')
LEGACY = AFS / 'SuperMimic'

def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def print_receipt(path):
    """Keep each log record below the platform's 500-character message limit."""
    raw = path.read_bytes()
    encoded = base64.b64encode(raw).decode('ascii')
    chunks = [encoded[i:i+320] for i in range(0, len(encoded), 320)]
    print('RECEIPT_META '+json.dumps({'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}), flush=True)
    for index, chunk in enumerate(chunks):
        print(f'RECEIPT_CHUNK {index:04d}/{len(chunks):04d} {chunk}', flush=True)
    print('RECEIPT_DONE', flush=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--label', required=True)
    a = p.parse_args()
    report = {'status': 'starting', 'engineering_only': True, 'sports_success_claimed': False,
              'label': a.label, 'stages': [], 'wall_start': time.time()}
    report_path = HERE / 'cloud-engineering.json'
    def save():
        report_path.write_text(json.dumps(report, indent=2)+'\n')
    save()
    try:
        candidates = [LEGACY/'course-replay-study-20261106/main-e/source',
                      LEGACY/'course-replay-study-20261106/pilot-c/source']
        cached = next(q for q in candidates if (q/'implementation/uni/src').is_dir())
        locked = json.loads((HERE/'runtime-source-lock.json').read_text())
        for name, expected in locked.items():
            if sha(cached/name) != expected:
                raise ValueError('Cached source differs from main: '+name)
        (ROOT/'implementation').mkdir(exist_ok=True)
        (ROOT/'implementation/uni').symlink_to(cached/'implementation/uni', target_is_directory=True)
        (ROOT/'acro-library-20261027-b').mkdir(exist_ok=True)
        (ROOT/'acro-library-20261027-b/scripts').symlink_to(cached/'acro-library-20261027-b/scripts', target_is_directory=True)
        report['cached_runtime'] = str(cached)
        report['verified_source_files'] = len(locked)
        # Only inspect file names in the user's own AFS. Do not open candidate models here.
        models = []
        scan_started = time.monotonic()
        scanned_dirs = 0
        skipped = {'.git', '.venv', 'site-packages', 'node_modules', '__pycache__', '.cache'}
        for folder, dirs, names in os.walk(AFS):
            dirs[:] = [n for n in dirs if n not in skipped and not n.startswith('.')]
            scanned_dirs += 1
            for name in names:
                if ('smpl' in name.lower()) and name.lower().endswith(('.npz','.pkl')):
                    models.append(str(Path(folder)/name))
            if time.monotonic()-scan_started > 25 or scanned_dirs > 20000:
                report['model_scan_censored'] = True
                break
        report['body_model_candidate_paths'] = models
        report['body_model_scan_dirs'] = scanned_dirs
        print('BODY_MODEL_INVENTORY '+json.dumps({'paths': models, 'censored':report.get('model_scan_censored',False)}), flush=True)
        parent = LEGACY/'lafan1-20261024/experiments/continuation-20261025-a/policy/full_016000.pt'
        expected = '0afcca9af97bd9db1f2dd083875898f962acab0b0a70c7313b221e575b70b05b'
        if sha(parent) != expected:
            raise ValueError('Common full parent mismatch')
        old = cached/'implementation/uni/src/unisupermimic/assets/motions/g1/lafan1_supermimic/walk1_subject1.csv'
        old_candidates = list((cached/'implementation/uni/src/unisupermimic/assets/motions/g1/lafan1_supermimic').glob('walk1*.csv'))
        if not old.exists():
            old = sorted(old_candidates)[0]
        lib = json.loads((cached/'acro-library-20261027-b/library-manifest.json').read_text())
        new_row = next(r for r in lib['files'] if Path(r['path']).name == '88_07_stageii_unitree_g1.csv')
        new = cached/'acro-library-20261027-b'/new_row['path']
        if not new.is_file():
            new = LEGACY/'acro-library-20261027-b'/new_row['path']
        clips = [{'path':str(old),'group':'old','fps':30,'sha256':sha(old)},
                 {'path':str(new),'group':'new','fps':30,'sha256':sha(new)}]
        manifest = HERE/'cloud-engineering-manifest.json'
        manifest.write_text(json.dumps({'reference_variant':'existing-engineering-motions-not-sports','clips':clips},indent=2)+'\n')
        import torch
        if torch.cuda.device_count() != 1:
            raise ValueError('Exactly one visible GPU required')
        report.update(status='running', gpu=torch.cuda.get_device_name(0), parent_sha256=expected,
                      torch_version=str(torch.__version__), motions=clips)
        save()
        for feedback in ['F0','F1']:
            cmd = [sys.executable, '-u', str(HERE/'train.py'), '--manifest', str(manifest),
                   '--parent', str(parent), '--feedback',feedback,'--seed','0','--label',a.label+'-'+feedback.lower(),
                   '--phase','engineering','--device','cuda','--steps','20','--wm-warmup','4',
                   '--num-envs','128','--max-seconds','600']
            started=time.monotonic()
            print('STAGE_START '+feedback,flush=True)
            subprocess.run(cmd, check=True, timeout=690)
            proof=json.loads((HERE/'runs'/(a.label+'-'+feedback.lower())/'proof.json').read_text())
            report['stages'].append({'feedback':feedback,'seconds':time.monotonic()-started,'proof':proof})
            save()
        report['status']='verified_cuda_interface_and_real_updates'
        save()
        print('CLOUD_ENGINEERING_RESULT '+json.dumps(report),flush=True)
        print_receipt(report_path)
    except BaseException as e:
        report.update(status='failed', error=repr(e))
        save()
        print('CLOUD_ENGINEERING_FAILURE '+json.dumps(report),flush=True)
        print_receipt(report_path)
        raise

if __name__ == '__main__':
    main()
