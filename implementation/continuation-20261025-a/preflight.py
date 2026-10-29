"""Finite CPU preflight; no training, dependency installation, or source mutation."""
from __future__ import annotations
import datetime as dt
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import torch
from audit_common import file_hash, state_hash, write_json, lr_at

OUT = Path(__file__).resolve().parent
BASE = Path(os.environ['SUPERMIMIC_BASE']).resolve()


def check():
    frozen = json.loads((OUT / 'frozen.json').read_text())
    for name, expected in frozen['base_files'].items():
        assert file_hash(BASE / name) == expected, name
    source = json.loads((BASE / 'evidence/source-files.json').read_text())
    assets = json.loads((BASE / 'evidence/asset-downloads.json').read_text())
    assert source['commit'] == frozen['source_commit']
    assert assets['revision'] == frozen['dataset_revision']
    for name, expected in source['files'].items():
        assert file_hash(BASE / 'uni' / name) == expected, name
    for item in assets['files']:
        path = BASE / 'uni' / item['path']
        assert path.stat().st_size == item['bytes'] and file_hash(path) == item['sha256'], item['path']
    env = json.loads((BASE / 'evidence/environment.json').read_text())
    actual = {name: importlib.metadata.version(name) for name in env['packages']}
    assert actual == env['packages'], {name: v for name, v in actual.items() if v != env['packages'][name]}
    parent = BASE / 'runs/baseline-seed0/last.pt'
    saved = torch.load(parent, map_location='cpu', weights_only=True)
    assert saved['iteration'] == 12000 and saved['sim_steps'] == 123142144
    assert set(['wm', 'pi', 'critic', 'opt_wm', 'opt_pi', 'opt_v']) <= saved.keys()
    args = json.loads((BASE / 'runs/baseline-seed0/args.json').read_text())
    assert json.loads(json.dumps(saved['args'])) == args and args['iterations'] == 12000
    assert args['lr_decay_iters'] == 250000 and args['lr'] == 2e-4
    assert args['seed'] == 0 and args['horizon'] == 32 and args['amp'] == 'bf16' and args['compile']
    hashes = {key: state_hash(saved[key]) for key in ['wm', 'pi', 'critic', 'opt_wm', 'opt_pi', 'opt_v']}
    steps = {key: sorted({int(state['step']) for state in saved[key]['state'].values()}) for key in ['opt_wm', 'opt_pi', 'opt_v']}
    assert steps == {'opt_wm': [120000], 'opt_pi': [11500], 'opt_v': [230000]}, steps
    result = {'passed': True, 'verified_at': dt.datetime.now(dt.timezone.utc).isoformat(),
              'parent_checkpoint': str(parent), 'parent_sha256': file_hash(parent),
              'iteration': saved['iteration'], 'sim_steps': saved['sim_steps'],
              'checkpoint_args': saved['args'], 'state_hashes': hashes, 'optimizer_steps': steps,
              'lr_at_first_continuation_iteration': lr_at(12000),
              'lr_decay_iters_unchanged': 250000, 'source_files_verified': len(source['files']),
              'asset_files_verified': len(assets['files']), 'package_versions': actual,
              'evaluation_script_sha256': file_hash(BASE / 'evaluate_audited_v2.py'),
              'cuda_build': torch.version.cuda, 'gpu_training_executed': False}
    return result


if __name__ == '__main__':
    result = check()
    write_json(OUT / 'evidence/preflight.json', result)
    print('CONTINUATION_PREFLIGHT_PASS ' + json.dumps({k: result[k] for k in ['passed', 'iteration', 'sim_steps', 'parent_sha256', 'optimizer_steps', 'source_files_verified', 'asset_files_verified']}), flush=True)
