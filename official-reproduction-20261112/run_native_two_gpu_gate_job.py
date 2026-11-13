"""Bound actual two-GPU torchrun and preserve exact original rank receipts."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motions',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    training=output/'training';scripts=Path(__file__).resolve().parent
    report=dict(status='starting',worker_timeout_seconds=2400,whole_program_seconds=2700,
        world_size=2,expected_global_real_steps=172032,quality_evaluation_completed=False,
        paper_results_reproduced=False,full_training_state_resume_available=False)
    (output/'result.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
    print('NATIVE_TWO_GPU_STAGE_START '+json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),
        log=str(output/'training.log'),worker_timeout_seconds=2400)),flush=True)
    started=time.monotonic()
    with (output/'training.log').open('wb') as stream:
        process=subprocess.run(['timeout','--signal=TERM','--kill-after=30s','2400s',
            sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1',
            '--nproc-per-node=2','--max-restarts=0',str(scripts/'run_native_two_gpu_gate.py'),
            '--motions',str(args.motions.resolve()),'--output',str(training)],
            stdout=stream,stderr=subprocess.STDOUT)
    report.update(status='failed',worker_returncode=process.returncode,
                  worker_elapsed_seconds=time.monotonic()-started)
    paths=[training/'result.json']
    paths+=[training/f'rank-{rank}'/name for rank in range(2) for name in
            ('resolved-config.json','result.json')]
    try:
        if process.returncode:raise RuntimeError(f'torchrun exited {process.returncode}')
        parent=json.loads(paths[0].read_bytes())
        assert parent['status']=='two_rank_native_training_completed'
        assert parent['world_size']==2 and parent['backend']=='nccl'
        assert parent['global_simulator_steps']==172032
        participants=parent['participants']
        assert [row['rank'] for row in participants]==[0,1]
        assert participants[0]['final_model_state_hashes']==participants[1]['final_model_state_hashes']
        for rank,row in enumerate(participants):
            folder=training/f'rank-{rank}'
            original=json.loads((folder/'result.json').read_bytes())
            assert original==row and row['status']=='two_rank_native_participant_completed'
            assert row['local_simulator_steps']==86016 and row['local_collection_steps']==21
            assert row['actor_real_observation_count']==172032 and row['all_native_learned_parameters_changed']
            assert all(count>0 for count in row['optimizer_steps'].values())
            assert row['resolved_config_sha256']==digest(folder/'resolved-config.json')
            assert len(row['checkpoints'])==3
            for record in row['checkpoints']:
                path=Path(record['path']).resolve()
                assert path.parent==folder and path.name==record['file']
                assert record['native_model_readback_verified']
                assert path.stat().st_size==record['bytes'] and digest(path)==record['sha256']
        report.update(status='two_GPU_native_gate_completed',global_real_simulator_steps=172032,
            rank_model_identity_and_parameter_changes_verified=True,
            original_rank_config_and_results_bound=True,
            native_weights_files_verified=6,parent_result_sha256=digest(paths[0]))
    except Exception as error:
        report['error']=f'{type(error).__name__}: {error}'
        print((output/'training.log').read_text(encoding='utf-8',errors='replace')[-10000:],flush=True)
    (output/'result.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
    paths+=[*sorted(training.glob('failed-rank-*.json')),output/'result.json']
    originals=[path for path in paths if path.is_file()]
    subprocess.run([sys.executable,str(scripts/'emit_json_receipts.py'),*map(str,originals)],check=True)
    if report['status']!='two_GPU_native_gate_completed':raise RuntimeError('Actual two-GPU gate failed; original receipts retained')
    print('NATIVE_TWO_GPU_GATE_PIPELINE_COMPLETED',flush=True)


if __name__=='__main__':main()
