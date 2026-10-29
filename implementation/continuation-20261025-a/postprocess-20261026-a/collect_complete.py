"""Freeze exact continuation artifacts for verified transfer; retains failures."""
from __future__ import annotations
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import tarfile
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from audit_common import file_hash, write_json


def main():
    root = Path(__file__).resolve().parent.parent
    base = Path(os.environ['SUPERMIMIC_BASE']).resolve()
    experiment = json.loads((root/'experiment.json').read_text())
    assert experiment['status'] in ['complete','failed_or_interrupted'], 'Never package an active experiment'
    archive = root/'continuation-results.tar.gz'
    assert not archive.exists(), 'Refusing to overwrite frozen results'
    chosen = set()
    def add(path):
        assert path.is_file() and not path.is_symlink()
        assert path.resolve().is_relative_to(root.resolve())
        chosen.add(path.relative_to(root).as_posix())
    for name in ['experiment.json','selection.json','provenance.json','frozen.json','audit_common.py',
                 'preflight.py','train_continuation.py','run_experiment.py','diagnose.py','collect_continuation.py']:
        if (root/name).is_file(): add(root/name)
    for folder,patterns in [('evidence',['*.json']),('evaluations',['*.json','*.txt']),
             ('diagnostics',['*.json','*.csv.gz']),('videos',['*.mp4']),
             ('runs',['launch.json','completion.json','resources.jsonl','stdout.txt'])]:
        for pattern in patterns:
            for path in (root/folder).rglob(pattern): add(path)
    auxiliary=root/'postprocess-20261026-a'
    auxiliary_state=json.loads((auxiliary/'evidence/postprocess.json').read_text())
    assert auxiliary_state['status']=='complete', 'Auxiliary outputs must be terminal and complete'
    for path in auxiliary.rglob('*'):
        if path.is_file() and path.suffix in ['.py','.json','.jsonl','.txt','.gz','.mp4']: add(path)
    for pattern in ['args.json','run_config.json','train_log.csv','update-state-*.json','events.out.tfevents.*']:
        for path in (root/'policy').glob(pattern): add(path)
    if (root/'selection.json').is_file():
        selection=json.loads((root/'selection.json').read_text())
        iterations={selection[k]['iteration'] for k in ['best','final','highest_macro']}
        for iteration in iterations: add(root/'policy'/f'full_{iteration:06d}.pt')
        for row in json.loads((root/'evaluations/index.json').read_text()):
            p=Path(row['checkpoint'])
            assert file_hash(p)==row['checkpoint_sha256']
            add(p)
    else:
        for name in ['last.pt','emergency-full.pt']:
            if (root/'policy'/name).is_file(): add(root/'policy'/name)
    frozen=json.loads((root/'frozen.json').read_text())
    original=base/'runs/baseline-evaluation.json'
    assert file_hash(original)==frozen['base_files']['runs/baseline-evaluation.json']
    shutil.copyfile(original,root/'original-evaluation.json');add(root/'original-evaluation.json')
    stages={}
    for p in (root/'runs').glob('*/completion.json'):
        row=json.loads(p.read_text())
        stages[p.parent.name]={k:row[k] for k in ['elapsed_seconds','returncode','stop_reason']}
    summary={'created_at':dt.datetime.now(dt.timezone.utc).isoformat(), 'status':experiment['status'],
             'target_confirmed':experiment.get('target_confirmed',False),'original_iteration':12000,
             'auxiliary_status':auxiliary_state['status'], 'primary_pipeline_status_preserved':experiment['status'],
             'strict_replay_failure_preserved':True, 'diagnostic_rollouts_are_independent':True,
             'final_iteration':json.loads((root/'evidence/continuation-proof.json').read_text())['after']['iteration'],'stages':stages,
             'original_evaluation':json.loads(original.read_text()),'selection':experiment.get('best'),
             'controlled_training_change':'continue absolute iterations with original optimizer and LR schedule',
             'resource_release_verified':False,'boundary':'Verify platform terminal state and supervised transfer cleanup separately.'}
    write_json(root/'result-summary.json',summary);add(root/'result-summary.json')
    manifest={name:{'bytes':(root/name).stat().st_size,'sha256':file_hash(root/name)} for name in sorted(chosen)}
    write_json(root/'result-manifest.json',manifest)
    with tarfile.open(archive,'w:gz',compresslevel=3) as tar:
        for name in sorted(chosen|{'result-manifest.json'}):tar.add(root/name,arcname=name,recursive=False)
    receipt={'archive':str(archive),'bytes':archive.stat().st_size,'sha256':file_hash(archive),
             'files':len(manifest)+1,'payload_bytes':sum(row['bytes'] for row in manifest.values())}
    write_json(root/'result-transfer-receipt.json',receipt)
    print('CONTINUATION_RESULTS_FROZEN '+json.dumps(receipt),flush=True)


if __name__=='__main__':main()
