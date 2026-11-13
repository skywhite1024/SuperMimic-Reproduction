"""Two bounded real-G1 PPO-WM integration arms; preserves failures and originals."""
import argparse
from datetime import datetime, timezone
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
    output=args.output.resolve(); output.mkdir(parents=True,exist_ok=False)
    scripts=Path(__file__).parent
    report=dict(status='running',arms=[],same_initial_native_weights_verified=False,
        matched_paper_training=False,paper_results_reproduced=False,
        quality_evaluation_completed=False,whole_program_limit_seconds=2400,
        per_arm_training_limit_seconds=900,per_arm_forced_kill_grace_seconds=30)
    receipts=[]
    def write():
        (output/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',
            encoding='utf-8',newline='\n')
    write()
    for mode in ('released_uint8','diagnostic_float'):
        stage_log=output/(mode+'.log')
        print('PPO_WM_ARM_START '+json.dumps(dict(mode=mode,utc=datetime.now(timezone.utc).isoformat(),
            stage_log=str(stage_log))),flush=True)
        started=time.monotonic()
        with stage_log.open('wb') as stream:
            process=subprocess.run(['timeout','--signal=TERM','--kill-after=30s','900s',
                sys.executable,str(scripts/'run_ppo_wm_integration.py'),
                '--motions',str(args.motions.resolve()),'--output',str(output/mode),
                '--termination',mode],stdout=stream,stderr=subprocess.STDOUT)
        row=dict(mode=mode,returncode=process.returncode,elapsed_seconds=time.monotonic()-started)
        for name in ('resolved-config.json','initial-model-state.json','result.json'):
            path=output/mode/name
            if path.is_file():
                receipts.append(path)
        if process.returncode:
            row.update(status='failed',log=str(stage_log))
            report['arms'].append(row); report['status']='failed'
            print('PPO_WM_ARM_FAILURE '+json.dumps(row),flush=True)
            print(stage_log.read_text(encoding='utf-8',errors='replace')[-6000:],flush=True)
            break
        try:
            result=json.loads((output/mode/'result.json').read_bytes())
            initial=json.loads((output/mode/'initial-model-state.json').read_bytes())
            assert result['status']=='real_g1_native_network_ppo_wm_integration_completed'
            assert result['real_native_ppo_updates']==3 and result['real_simulator_steps']==122880
            assert result['imagined_transitions']==393216
            assert result['final_actor_and_critic_save_readback_verified']
            assert result['final_world_model_and_context_save_readback_verified']
            assert len(result['snapshots'])==3 and len(result['updates'])==3
            for update in result['updates']:
                trace=Path(update['trace']['path'])
                assert trace.stat().st_size==update['trace']['bytes']
                assert hashlib.sha256(trace.read_bytes()).hexdigest()==update['trace']['sha256']
        except Exception as error:
            row.update(status='validation_failed',error=f'{type(error).__name__}: {error}')
            report['arms'].append(row); report['status']='failed'
            break
        row.update(status='complete',initial_state_hashes=initial['state_hashes'],
            dimensions=initial['dimensions'],result_sha256=hashlib.sha256((output/mode/'result.json').read_bytes()).hexdigest())
        report['arms'].append(row); write()
        print('PPO_WM_ARM_EXIT '+json.dumps(row),flush=True)
    if len(report['arms'])==2 and all(row['status']=='complete' for row in report['arms']):
        same=report['arms'][0]['initial_state_hashes']==report['arms'][1]['initial_state_hashes']
        report.update(status='complete' if same else 'incomplete_initialization_mismatch',
            same_initial_native_weights_verified=same)
    write()
    receipts.append(output/'result.json')
    subprocess.run([sys.executable,str(scripts/'emit_json_receipts.py'),*map(str,receipts)],check=True)
    if report['status']!='complete':
        raise RuntimeError('PPO-WM integration pair incomplete; original failures retained')
    print('PPO_WM_REAL_G1_PAIR_COMPLETED',flush=True)


if __name__=='__main__':
    main()
