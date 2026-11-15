"""Finite Figure5-sized PPO-WM learning and native simulator evaluation on18 clips."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

SCRIPTS=Path(__file__).parent
POINTS=(64,256,1024,2048)
FILES=('result.json','protocol.json','accounting.json','audit/summary.json','audit/initial-states.json')


def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8',newline='\n')
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(script,args,log,limit):
    print('PPO_WM_BATCH_STAGE_START '+json.dumps(dict(script=script,log=str(log),utc=datetime.now(timezone.utc).isoformat(),limit_seconds=limit)),flush=True)
    started=time.monotonic()
    with log.open('w',encoding='utf-8') as stream:
        r=subprocess.run(['timeout','--signal=TERM','--kill-after=30s',str(limit)+'s',sys.executable,str(SCRIPTS/script),*map(str,args)],stdout=stream,stderr=subprocess.STDOUT)
    print('PPO_WM_BATCH_STAGE_EXIT '+json.dumps(dict(script=script,returncode=r.returncode,elapsed_seconds=time.monotonic()-started)),flush=True)
    if r.returncode:
        print(log.read_text(encoding='utf-8',errors='replace')[-12000:],flush=True)
        raise RuntimeError(script+' exited '+str(r.returncode))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-motions',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--batch-size',type=int,required=True)
    parser.add_argument('--rollout-steps',type=int,required=True)
    args=parser.parse_args()
    assert (args.batch_size,args.rollout_steps) in ((4096,32),(8192,24),(8192,32),(16384,32))
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    report=dict(status='starting',batch_size=args.batch_size,rollout_steps=args.rollout_steps,
        target_updates=2048,termination_mode='released_uint8',training_from_scratch=True,
        author_exact_Figure5_membership_confirmed=False,PPO_hyperparameters_author_confirmed=False,
        native_full_first_order_comparison_completed=False,matched_return_or64x_claimed=False,
        paper_results_reproduced=False,points=[])
    write(output/'result.json',report)
    try:
        execute('prepare_lafan_locomotion18.py',['--source',args.source_motions,'--output',output/'motions'],output/'prepare.log',180)
        execute('run_ppo_wm_integration.py',['--motions',output/'motions','--output',output/'training',
            '--termination','released_uint8','--training-purpose','finite-training','--updates',2048,
            '--batch-size',args.batch_size,'--rollout-steps',args.rollout_steps,'--expected-clips',18],output/'training.log',7200)
        report.update(status='evaluating',training_result_sha256=digest(output/'training/result.json'))
        write(output/'result.json',report)
        for update in POINTS:
            evaluation=output/f'eval-{update:06d}';point=dict(ppo_update=update,status='starting')
            try:
                execute('evaluate_motion_corpus_v2.py',['--run',output/'training','--motions',output/'motions',
                    '--output',evaluation,'--expected-clips',18,'--num-envs',18,'--order','index',
                    '--training-algorithm','ppo-wm','--ppo-wm-update',update],output/f'eval-{update:06d}.log',900)
                execute('audit_corpus_result.py',['--result',evaluation/'result.json','--protocol',evaluation/'protocol.json',
                    '--motions',output/'motions','--expected-clips',18,'--require-trajectories','--output',evaluation/'accounting.json'],output/f'accounting-{update:06d}.log',180)
                execute('audit_saved_initial_states_v2.py',['--evaluation',evaluation,'--motions',output/'motions',
                    '--expected-clips',18,'--output',evaluation/'audit'],output/f'arrays-{update:06d}.log',180)
                accounting=json.loads((evaluation/'accounting.json').read_bytes())
                point.update(status='complete',successes=accounting['successes'],failures=accounting['failures'],
                    mean_normalized_full_motion_return=accounting['mean_normalized_full_motion_return'],
                    cumulative_imagined_steps=update*args.batch_size*args.rollout_steps,
                    real_simulator_steps=update*10*4096,
                    receipts={name:digest(evaluation/name) for name in FILES})
            except Exception as error:point.update(status='failed',error=f'{type(error).__name__}: {error}')
            report['points'].append(point);write(output/'result.json',report)
        report['status']='complete' if all(p['status']=='complete' for p in report['points']) else 'incomplete'
    except Exception as error:report.update(status='failed_or_incomplete',error=f'{type(error).__name__}: {error}')
    write(output/'result.json',report)
    receipts=[output/'motion-subset.json']
    receipts += [output/'training'/name for name in ('resolved-config.json','initial-model-state.json','native-checkpoint-manifest.json','result.json')]
    receipts += [output/f'eval-{step:06d}'/name for step in POINTS for name in FILES]
    receipts=[p for p in receipts if p.is_file()]+[output/'result.json']
    print('PPO_WM_BATCH_ORIGINALS_BEGIN',flush=True)
    subprocess.run([sys.executable,str(SCRIPTS/'emit_json_receipts.py'),'--max-bytes','64000000',*map(str,receipts)],check=True)
    print('PPO_WM_BATCH_ORIGINALS_END',flush=True)
    if report['status']!='complete':raise RuntimeError('Finite PPO-WM learning incomplete; every written bounded original retained')
    print('PPO_WM_BATCH_PIPELINE_COMPLETED',flush=True)


if __name__=='__main__':main()
