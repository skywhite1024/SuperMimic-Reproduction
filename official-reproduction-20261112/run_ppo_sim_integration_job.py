"""Bound real-simulator PPO integration and emit originals even after failure."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motions',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    scripts=Path(__file__).parent
    print('PPO_SIM_STAGE_START '+json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),
        log=str(output/'training.log'),timeout_seconds=1200)),flush=True)
    started=time.monotonic()
    with (output/'training.log').open('wb') as stream:
        process=subprocess.run(['timeout','--signal=TERM','--kill-after=30s','1200s',
            sys.executable,str(scripts/'run_ppo_sim_integration.py'),
            '--motions',str(args.motions.resolve()),'--output',str(output/'training')],
            stdout=stream,stderr=subprocess.STDOUT)
    report=dict(status='failed',worker_returncode=process.returncode,
        worker_elapsed_seconds=time.monotonic()-started,
        whole_program_seconds=1800,quality_evaluation_completed=False,paper_results_reproduced=False)
    originals=[]
    for name in ['resolved-config.json','checkpoint-manifest.json','result.json']:
        file=output/'training'/name
        if file.is_file():originals.append(file)
    if process.returncode==0:
        try:
            result=json.loads((output/'training/result.json').read_bytes())
            assert result['status']=='real_G1_simulator_PPO_integration_completed'
            assert result['ppo_update_calls']==3 and result['collection_steps']==72
            assert result['simulator_steps']==589824
            assert result['imagined_transitions']==result['world_model_updates']==0
            assert result['return_sampler_distribution_updates']==3
            assert result['all_model_and_trace_readbacks_verified']
            assert len(result['updates'])==len(result['checkpoint_sets'])==3
            assert len(originals)==3
            report.update(status='complete',ppo_update_calls=3,real_simulator_steps=589824,
                imagined_transitions=0,world_model_updates=0,
                worker_result_sha256=hashlib.sha256((output/'training/result.json').read_bytes()).hexdigest())
        except Exception as error:
            report['error']=f'{type(error).__name__}: {error}'
    if report['status']!='complete':
        print((output/'training.log').read_text(encoding='utf-8',errors='replace')[-8000:],flush=True)
    (output/'result.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
    originals.append(output/'result.json')
    subprocess.run([sys.executable,str(scripts/'emit_json_receipts.py'),*map(str,originals)],check=True)
    if report['status']!='complete':raise RuntimeError('PPO-Sim integration failed; original outputs retained')
    print('PPO_SIM_REAL_G1_PIPELINE_COMPLETED',flush=True)


if __name__=='__main__':
    main()
