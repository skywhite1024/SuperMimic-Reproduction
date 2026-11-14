"""Finite H8/16 or H64/128 native critic/no-critic16k endpoints, all failures kept."""
import argparse
from pathlib import Path
import subprocess
import sys

from run_lafan_learning_curve import run_case, write


def export_phase(folder, step, label):
    names=['training/result.json','training/resolved-config.json','training/checkpoint-times.json']
    names += [f'eval-{step:09d}/'+name for name in
              ('result.json','protocol.json','accounting.json','audit/summary.json','audit/initial-states.json')]
    names += ['result.json']
    paths=[folder/name for name in names if (folder/name).is_file()]
    if not paths:return
    print('HORIZON_GRID_ORIGINALS_BEGIN '+label,flush=True)
    subprocess.run([sys.executable,str(Path(__file__).parent/'emit_json_receipts.py'),*map(str,paths)],check=True)
    print('HORIZON_GRID_ORIGINALS_END '+label,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motions',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--horizons',nargs='+',type=int,choices=(8,16,64,128),required=True)
    args=parser.parse_args()
    if len(args.horizons)!=2 or set(args.horizons) not in ({8,16},{64,128}):
        parser.error('Choose exactly the finite pair8/16 or64/128')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',horizons=args.horizons,main_collection_steps=16001,
        num_envs=4096,seed=42,save_interval=4000,fixed_start_protocol='manual-reset-v2 full canonical40',
        evaluation_clips_per_point=40,reference_frames_per_point=441140,
        H32_previously_completed_and_not_repeated=True,whole_program_seconds=32400,
        full_training_episode_return_curves_completed=False,multiple_seeds=False,
        paper_results_reproduced=False,cases=[])
    write(output/'result.json',report)
    for horizon in args.horizons:
        for no_critic in (False,True):
            label=f'h{horizon:03d}-'+('no-critic' if no_critic else 'critic')
            case=output/label;case.mkdir()
            row=dict(case=label,horizon=horizon,critic_enabled=not no_critic,status='starting',phases=[])
            report['cases'].append(row);write(output/'result.json',report)
            for phase,steps,interval,point,limit in [('integration',21,10,11,1800),('main',16001,4000,16001,7200)]:
                folder=case/phase
                row['status']=phase;write(output/'result.json',report)
                try:
                    result=run_case(folder,args.motions.resolve(),steps,interval,[point],limit,900,
                        policy_horizon=horizon,without_critic=no_critic,emit_receipts=False)
                    row['phases'].append(dict(phase=phase,status='complete',point=result['points'][0]))
                except Exception as error:
                    row['status']='failed_or_incomplete'
                    row['phases'].append(dict(phase=phase,status='failed_or_incomplete',error=f'{type(error).__name__}: {error}'))
                    export_phase(folder,point,label+'/'+phase)
                    write(output/'result.json',report)
                    print('HORIZON_GRID_CASE_FAILED '+label+' '+phase,flush=True)
                    break
                export_phase(folder,point,label+'/'+phase)
            else:
                row['status']='complete'
                print('HORIZON_GRID_CASE_COMPLETED '+label,flush=True)
            write(output/'result.json',report)
    report['status']='complete' if all(row['status']=='complete' for row in report['cases']) else 'failed_or_incomplete'
    write(output/'result.json',report)
    print('HORIZON_GRID_SUMMARY_BEGIN',flush=True)
    subprocess.run([sys.executable,str(Path(__file__).parent/'emit_json_receipts.py'),str(output/'result.json')],check=True)
    print('HORIZON_GRID_SUMMARY_END',flush=True)
    if report['status']!='complete':raise RuntimeError('Horizon grid retained incomplete/failed cases')
    print('HORIZON_GRID_PIPELINE_COMPLETED',flush=True)


if __name__=='__main__':main()
