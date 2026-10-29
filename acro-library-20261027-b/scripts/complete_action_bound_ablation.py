"""Finite local completion of the frozen same-weight ablation; no cloud actions."""
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def main():
    plan = json.loads((ROOT/'action-bound-consistency-v1-extension-plan.json').read_text(encoding='utf8'))
    for name, digest in plan['files'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    checkpoint = ROOT/'policy/unified-h100-stage3-4000.pt'
    receipt = ROOT/'evidence/action-bound-completion.json'
    assert not receipt.exists(), 'Inspect existing completion process rather than overwrite'
    started, completed = time.monotonic(), []
    def record(status, **extra):
        receipt.write_text(json.dumps({'status':status,'updated_at':datetime.now(timezone.utc).isoformat(),
            'plan_sha256':hashlib.sha256((ROOT/'action-bound-consistency-v1-extension-plan.json').read_bytes()).hexdigest(),
            'completed_commands':completed,**extra},indent=2)+'\n',encoding='utf8')
    def run(script, argv, artifact=None, render=False):
        if artifact and (ROOT/artifact).exists():
            return
        remaining = 3600-(time.monotonic()-started)
        assert remaining>0, 'Finite local ablation completion budget exhausted'
        log = ROOT/'evidence'/f'action-bound-completion-{len(completed):02}-{script.removesuffix(".py")}.log'
        assert not log.exists()
        print('BOUND_LOCAL_START '+json.dumps({'script':script,'arguments':argv}),flush=True)
        environment = os.environ.copy()
        environment.update(PYTHONUTF8='1',MUJOCO_GL='glfw' if render else 'disable')
        with log.open('w',encoding='utf8') as output:
            subprocess.run([sys.executable,'-u',str(ROOT/'scripts'/script),*argv],env=environment,
                           stdout=output,stderr=subprocess.STDOUT,check=True,timeout=min(1200,remaining))
        if artifact:
            assert (ROOT/artifact).exists()
        completed.append({'script':script,'arguments':argv,'log':str(log.relative_to(ROOT))})
        record('running')
        print('BOUND_LOCAL_END '+json.dumps({'script':script,'artifact':artifact}),flush=True)
    try:
        record('running')
        corrected='stage3-actionbound20-corrected-cpu-all55'
        assert json.loads((ROOT/f'runs/{corrected}/summary.json').read_text())['same_checkpoint_weights_unchanged']
        def assess_library(mode):
            for script,artifact in [('assess_tracking.py','tracking-assessment.json'),('audit_physical_contacts.py','contact-review.json'),('combine_acceptance.py','combined-assessment.json')]:
                run(script,['--run',mode],f'runs/{mode}/{artifact}')
        assess_library(corrected)
        run('diagnose_action_bound.py',[])
        rawcontrol='stage3-raw-cpu-new15-control'
        run('evaluate_library.py',['--checkpoint',str(checkpoint),'--mode',rawcontrol,'--group','new','--device','cpu'],f'runs/{rawcontrol}/summary.json')
        assess_library(rawcontrol)
        raw='stage3-actionbound20-raw-cpu-new15'
        run('evaluate_action_bound_extension.py',['--suite','library','--reference-variant','raw','--checkpoint',str(checkpoint),'--mode',raw,'--group','new','--device','cpu'],f'runs/{raw}/summary.json')
        assess_library(raw)
        for variant,mode,baseline in [('raw','stage3-actionbound20-frozen-continuous-v1','stage3-frozen-continuous-v1'),
                                      ('corrected','stage3-actionbound20-corrected-continuous-v1','stage3-corrected-continuous-v1')]:
            run('evaluate_action_bound_extension.py',['--suite','sequences','--reference-variant',variant,'--checkpoint',str(checkpoint),'--mode',mode,'--device','cpu'],f'runs/{mode}/summary.json')
            run('assess_sequences.py',['--run',mode],f'runs/{mode}/continuous-assessment.json')
            for name in [baseline,mode]:
                run('audit_sequence_contacts.py',['--run',name],f'runs/{name}/continuous-contact-review.json')
        run('diagnose_action_bound.py',[])
        record('all_numeric_comparisons_complete_videos_pending')
        for mode in [corrected,raw]:
            data=json.loads((ROOT/f'runs/{mode}/summary.json').read_text())
            for row in data['clips']:
                if row['group']=='new' or not row['passed_original_hard_gate']:
                    name=row['clip'];argv=['--run',mode,'--clips',name]
                    if row['group']=='lafan':argv.append('--include-old')
                    run('render_library.py',argv,f'videos/{mode}/{name}.json',render=True)
        for mode in ['stage3-actionbound20-frozen-continuous-v1','stage3-actionbound20-corrected-continuous-v1']:
            for seed in [0,1,2]:
                run('render_library.py',['--run',mode,'--clips',f'seed-{seed}'],f'videos/{mode}/seed-{seed}.json',render=True)
        verified=[]
        for mode in [corrected,raw,'stage3-actionbound20-frozen-continuous-v1','stage3-actionbound20-corrected-continuous-v1']:
            for path in sorted((ROOT/'videos'/mode).glob('*.json')):
                data=json.loads(path.read_text());target=ROOT/data['file']
                assert data['complete_decode_passed'] and target.stat().st_size==data['bytes'] and hashlib.sha256(target.read_bytes()).hexdigest()==data['sha256']
                verified.append(str(path.relative_to(ROOT)))
        record('all_local_comparisons_and_videos_verified',video_receipts=verified,training_updates=0,cloud_resources_started=0)
        print('BOUND_LOCAL_COMPLETE '+json.dumps({'videos':len(verified),'completed_commands':len(completed)}),flush=True)
    except BaseException as error:
        record('failed_or_interrupted',error=repr(error))
        raise


if __name__=='__main__':
    main()
