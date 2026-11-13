"""Reuse the completed PPO gate, then a finite 2048-update real-G1 experiment."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from ppo_sim_checkpoint import digest, select_ppo_sim_checkpoint

SCRIPTS=Path(__file__).resolve().parent
RETAINED=[3,128,256,512,1024,1536,2048]
POINTS=RETAINED[1:]


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def execute(script,arguments,log,seconds):
    print('PPO_SIM_CURVE_STAGE_START '+json.dumps(dict(script=script,log=str(log),
        utc=datetime.now(timezone.utc).isoformat(),timeout_seconds=seconds)),flush=True)
    started=time.monotonic()
    with log.open('wb') as stream:
        result=subprocess.run(['timeout','--signal=TERM','--kill-after=30s',str(seconds)+'s',
            sys.executable,str(SCRIPTS/script),*map(str,arguments)],stdout=stream,stderr=subprocess.STDOUT)
    print('PPO_SIM_CURVE_STAGE_EXIT '+json.dumps(dict(script=script,returncode=result.returncode,
        elapsed_seconds=time.monotonic()-started)),flush=True)
    if result.returncode:
        print(log.read_text(encoding='utf-8',errors='replace')[-10000:],flush=True)
        raise RuntimeError(f'{script} failed {result.returncode}; original log retained at {log}')


def evaluate(run,update,motions,output,receipts):
    output.mkdir()
    evaluation=output/'evaluation'
    paths=[evaluation/name for name in ('result.json','protocol.json','accounting.json',
        'audit/summary.json','audit/initial-states.json')]
    try:
        execute('evaluate_motion_corpus_v2.py',['--run',run,'--training-algorithm','ppo-sim',
            '--ppo-sim-update',update,'--motions',motions,'--output',evaluation,
            '--expected-clips',40,'--num-envs',40,'--order','index','--seed',42],output/'evaluation.log',900)
        execute('audit_corpus_result.py',['--result',evaluation/'result.json',
            '--protocol',evaluation/'protocol.json','--motions',motions,'--expected-clips',40,
            '--require-trajectories','--output',evaluation/'accounting.json'],output/'accounting.log',180)
        execute('audit_saved_initial_states_v2.py',['--evaluation',evaluation,'--motions',motions,
            '--expected-clips',40,'--output',evaluation/'audit'],output/'array-audit.log',180)
        accounting=json.loads((evaluation/'accounting.json').read_bytes())
        protocol=json.loads((evaluation/'protocol.json').read_bytes())
        _,_,_,selection=select_ppo_sim_checkpoint(run,update)
        assert protocol['checkpoint_selection']==selection
        assert protocol['training_algorithm']=='ppo-sim'
        assert accounting['clips']==40 and accounting['reference_frames']==441140
        assert json.loads(paths[3].read_bytes())['status']=='complete_trajectory_integrity_verified'
        assert json.loads(paths[4].read_bytes())['status']=='complete_saved_initial_state_audit_verified'
        point=dict(status='complete',ppo_update=update,collection_steps=update*24,
            real_simulator_steps=update*24*8192,successes=accounting['successes'],failures=accounting['failures'],
            mean_normalized_full_motion_return=accounting['mean_normalized_full_motion_return'],
            checkpoint_selection=selection,all_saved_arrays_and_initial_states_verified=True,
            receipts={p.relative_to(evaluation).as_posix():digest(p) for p in paths})
        print('PPO_SIM_CURVE_POINT '+json.dumps(point),flush=True)
        return point
    finally:
        receipts.extend(path for path in paths if path.is_file())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motions',type=Path,required=True)
    parser.add_argument('--completed-gate',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    motions=args.motions.resolve();gate=args.completed_gate.resolve()
    report=dict(status='starting',algorithm='PPO-Sim',seed=42,num_envs=8192,
        rollout_steps=24,main_updates=2048,main_real_simulator_steps=402653184,
        checkpoint_updates=RETAINED,evaluated_main_updates=POINTS,points=[],
        training_timeout_seconds=5400,whole_program_seconds=16200,
        gate_training_rerun=False,no_world_model_constructed=True,
        imagined_transitions=0,world_model_updates=0,quality_evaluation_completed=False,
        evaluation='same canonical40 manual-reset-v2 deterministic native actor',
        paper_PPO_hyperparameters_and_sampler_confirmed=False,
        full_18h_1_4_8_GPU_comparison_completed=False,paper_results_reproduced=False,
        threshold=.82,threshold_definition='First sampled checkpoint at or above .82; no exact crossing-time claim')
    receipts=[]
    try:
        parent,_,_,selection=select_ppo_sim_checkpoint(gate,3)
        assert parent['ppo_update_calls']==3 and parent['simulator_steps']==589824
        report['completed_gate_selection']=selection
        report['gate_evaluation']=evaluate(gate,3,motions,output/'completed-gate-eval',receipts)
        report['status']='training';write(output/'result.json',report)
        training=output/'training'
        execute('run_ppo_sim_integration.py',['--motions',motions,'--output',training,
            '--training-purpose','finite-training','--updates',2048,
            '--checkpoint-updates',*RETAINED,'--trace-updates',1,*RETAINED],output/'training.log',5400)
        completed,config,_,_=select_ppo_sim_checkpoint(training,2048)
        assert completed['status']=='real_G1_simulator_PPO_finite_training_completed'
        assert completed['simulator_steps']==report['main_real_simulator_steps']
        assert [row['ppo_update'] for row in completed['checkpoint_sets']]==RETAINED
        assert sum('trace' in row for row in completed['updates'])==len(RETAINED)+1
        previous=-1.
        for row in completed['checkpoint_sets']:
            select_ppo_sim_checkpoint(training,row['ppo_update'])
            current=row['retained_snapshot_complete_seconds_since_training_entry']
            assert current>previous>=-1
            previous=current
        report.update(status='evaluating',training_result_sha256=digest(training/'result.json'),
            elapsed_training_seconds=completed['elapsed_training_seconds'])
        write(output/'result.json',report)
        for update in POINTS:
            try:
                point=evaluate(training,update,motions,output/f'eval-{update:06d}',receipts)
                snapshot=next(row for row in completed['checkpoint_sets'] if row['ppo_update']==update)
                point['checkpoint_timing']={key:value for key,value in snapshot.items() if key.startswith('retained_snapshot_complete_seconds')}
            except Exception as error:
                point=dict(status='failed',ppo_update=update,error=f'{type(error).__name__}: {error}')
            report['points'].append(point);write(output/'result.json',report)
        report['quality_evaluation_completed']=all(row['status']=='complete' for row in report['points'])
        report['status']='complete' if report['quality_evaluation_completed'] else 'incomplete'
        report['first_observed_main_sample_at_or_above_threshold']=next((row for row in report['points']
            if row['status']=='complete' and row['mean_normalized_full_motion_return']>=.82),None)
    except Exception as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}')
    finally:
        receipts.extend(path for path in (output/'training'/name for name in
            ('resolved-config.json','checkpoint-manifest.json','result.json')) if path.is_file())
        write(output/'result.json',report);receipts.append(output/'result.json')
        print('PPO_SIM_CURVE_RECEIPTS_BEGIN',flush=True)
        subprocess.run([sys.executable,str(SCRIPTS/'emit_json_receipts.py'),*map(str,receipts)],check=True)
        print('PPO_SIM_CURVE_RECEIPTS_END',flush=True)
    if report['status']!='complete':raise RuntimeError('Finite PPO-Sim experiment incomplete; all failures retained')
    print('PPO_SIM_CURVE_PIPELINE_COMPLETED',flush=True)


if __name__=='__main__':main()
