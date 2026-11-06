"""Finite full-55 evaluation of the common parent and all 12 completed main runs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from study_runtime import ROOT, STUDY, plan, references, sha, verify_inputs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prefix', required=True, help='Prefix used by the main training matrix')
    p.add_argument('--label', required=True, help='Fresh evaluation block identifier')
    p.add_argument('--parent', type=Path)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--per-evaluation-seconds', type=int, default=1200)
    a = p.parse_args(); spec = plan()
    assert spec['main_budget_frozen'], 'Freeze the main protocol after the pilot'
    assert 0 < a.per_evaluation_seconds <= 1800
    parent = (a.parent or ROOT/spec['parent_path']).resolve()
    assert sha(parent) == spec['parent_sha256']
    verify_inputs(references()[0])
    cases = [('baseline', parent)]
    for seed in spec['training_seeds']:
        for rho in spec['old_replay_fractions']:
            tag = f'rho{round(rho*100):03d}-seed{seed}'
            run = STUDY/'runs'/f'{a.prefix}-{tag}'
            proof = json.loads((run/'proof.json').read_text(encoding='utf-8'))
            assert proof['scientific_main_result'] and proof['device'] == 'cuda'
            assert proof['variant']['plan_sha256'] == sha(STUDY/'plan.json')
            assert proof['variant']['seed'] == seed and proof['variant']['old_replay_fraction'] == rho
            assert proof['new_updates'] == spec['provisional_main']['steps']
            checkpoint = run/'final-full.pt'
            assert sha(checkpoint) == proof['checkpoint_sha256']
            cases.append((tag, checkpoint))
    if not a.execute:
        print(json.dumps([{'case':tag, 'checkpoint':str(path)} for tag,path in cases],indent=2)); return
    output = STUDY/'runs'/f'evaluation-{a.label}'
    assert output.resolve().is_relative_to((STUDY/'runs').resolve()) and not output.exists()
    output.mkdir(parents=True)
    evaluator = ROOT/'acro-library-20261027-b/scripts/evaluate_library.py'
    receipt = {'device':'cuda', 'backend':'mjwarp', 'plan_sha256':sha(STUDY/'plan.json'),
               'evaluator_sha256':sha(evaluator), 'cases':[], 'status':'running'}
    started = time.perf_counter()
    for tag, checkpoint in cases:
        mode = f'replay-study-{a.label}-{tag}'
        summary = ROOT/'acro-library-20261027-b/runs'/mode/'summary.json'
        assert not summary.parent.exists(), 'Keep every earlier evaluation intact'
        command = [sys.executable,str(evaluator),'--checkpoint',str(checkpoint),
                   '--mode',mode,'--group','all','--device','cuda']
        with (output/f'{tag}.log').open('x',encoding='utf-8') as stream:
            subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True,
                           timeout=a.per_evaluation_seconds)
        result = json.loads(summary.read_text(encoding='utf-8'))
        assert result['checkpoint_sha256'] == sha(checkpoint)
        assert result['backend'] == 'mjwarp' and result['observation_dim'] == 99
        assert result['post_initialization_resets'] == 0 and len(result['clips']) == 55
        assert result['groups']['lafan']['denominator'] == 40 and result['groups']['new']['denominator'] == 15
        receipt['cases'].append({'case':tag,'summary':summary.relative_to(ROOT).as_posix(),
                                 'summary_sha256':sha(summary),'groups':result['groups']})
        receipt['wall_seconds'] = time.perf_counter()-started
        (output/'receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
        print('FULL55_EVALUATED '+json.dumps(receipt['cases'][-1]),flush=True)
    receipt['status'] = 'all_13_full55_evaluations_completed'
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')


if __name__ == '__main__': main()
