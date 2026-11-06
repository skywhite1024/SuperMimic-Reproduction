"""One finite GPU pilot block using existing personal-AFS assets and environment."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from study_runtime import ROOT, STUDY, plan, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--legacy-root', type=Path, required=True)
    p.add_argument('--expected-commit', required=True)
    p.add_argument('--label', required=True)
    a = p.parse_args(); legacy = a.legacy_root.resolve(); spec = plan()
    assert ROOT.resolve().is_relative_to(legacy.parent)
    output = STUDY/'runs'/f'pilot-block-{a.label}'
    assert output.resolve().is_relative_to((STUDY/'runs').resolve()) and not output.exists()
    output.mkdir(parents=True)
    started = time.perf_counter()
    state = {'status':'preparing','source_commit':a.expected_commit,'label':a.label,'stages':[]}
    def save():
        state['wall_seconds'] = time.perf_counter()-started
        (output/'receipt.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
    def run(tag, command, maximum):
        state['stage'] = tag; save()
        before = time.perf_counter()
        print('PILOT_STAGE_STARTED '+tag,flush=True)
        subprocess.run(command,cwd=ROOT,check=True,timeout=maximum)
        state['stages'].append({'stage':tag,'wall_seconds':time.perf_counter()-before}); save()
    def stage(source, target, expected):
        assert source.resolve().is_relative_to(legacy)
        assert target.resolve().is_relative_to(ROOT.resolve())
        assert sha(source) == expected, f'Existing AFS asset mismatch: {source.name}'
        if target.exists():
            assert sha(target) == expected
        else:
            target.parent.mkdir(parents=True,exist_ok=True)
            try: os.link(source,target)
            except OSError: shutil.copyfile(source,target)
    try:
        actual = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        assert actual == a.expected_commit
        source = json.loads((legacy/'lafan1-20261024/evidence/source-files.json').read_text())
        assert source['commit'] == 'faf7818e8abac3ec6abb8f00e6e904d3577f800b'
        old_uni = legacy/'lafan1-20261024/uni'
        inputs = json.loads((STUDY/'inputs.lock.json').read_text())
        for name, expected in inputs['motion_sha256'].items():
            relative = Path(name)
            source = old_uni/relative.relative_to('implementation/uni') if name.startswith('implementation/uni/') else legacy/relative
            stage(source,ROOT/relative,expected)
        assets = json.loads((ROOT/'implementation/evidence/asset-downloads.json').read_text())
        for row in assets['files']:
            stage(old_uni/row['path'],ROOT/'implementation/uni'/row['path'],row['sha256'])
        parent = ROOT/spec['parent_path']
        stage(legacy/'lafan1-20261024/experiments/continuation-20261025-a/policy/full_016000.pt',parent,spec['parent_sha256'])
        import torch
        assert torch.cuda.device_count() == 1
        assert 'H100' in torch.cuda.get_device_name(0), torch.cuda.get_device_name(0)
        state.update(gpu=torch.cuda.get_device_name(0),torch_version=torch.__version__,parent_sha256=sha(parent))
        state['status'] = 'running'; save()
        py = sys.executable
        baseline_mode = f'replay-study-pilot-{a.label}-baseline-cuda'
        run('baseline-full55',[py,str(ROOT/'acro-library-20261027-b/scripts/evaluate_library.py'),
            '--checkpoint',str(parent),'--mode',baseline_mode,'--group','all','--device','cuda'],1200)
        baseline = json.loads((ROOT/'acro-library-20261027-b/runs'/baseline_mode/'summary.json').read_text())
        assert baseline['backend'] == 'mjwarp' and len(baseline['clips']) == 55
        state['baseline'] = baseline['groups']; save()
        run('four-ratio-timing-pilot',[py,str(STUDY/'run_matrix.py'),'--execute','--phase','pilot',
            '--device','cuda','--prefix',a.label],spec['pilot']['max_seconds']*4+180)
        proofs = []
        checkpoints = [parent]
        for rho in spec['old_replay_fractions']:
            directory = STUDY/'runs'/f"{a.label}-rho{round(rho*100):03d}-seed{spec['pilot']['seed']}"
            proof = json.loads((directory/'proof.json').read_text())
            assert not proof['scientific_main_result'] and proof['new_updates'] == spec['pilot']['steps']
            proofs.append({k:proof[k] for k in ['variant','new_updates','policy_updates','wall_seconds',
                'new_sim_transitions','peak_cuda_allocated_bytes','peak_cuda_reserved_bytes','checkpoint_sha256']})
            checkpoints.append(directory/'final-full.pt')
        state['pilot_runs'] = proofs; save()
        run('fixed-probe-collection',[py,str(STUDY/'wm_probe.py'),'--collect','--device','cuda'],300)
        run('world-model-probe',[py,str(STUDY/'wm_probe.py'),'--probe',str(STUDY/'runs/frozen-probe-cuda/windows.npz'),
            '--checkpoints',*[str(path) for path in checkpoints],'--output',str(output/'wm-diagnostics.json')],300)
        state['status'] = 'pilot_complete_not_main_results'; save()
        print('REPLAY_GPU_PILOT_RESULT '+json.dumps(state),flush=True)
    except BaseException as error:
        state.update(status='failed_preserved',error=repr(error)); save()
        print('REPLAY_GPU_PILOT_FAILURE '+json.dumps(state),flush=True)
        raise


if __name__ == '__main__': main()
