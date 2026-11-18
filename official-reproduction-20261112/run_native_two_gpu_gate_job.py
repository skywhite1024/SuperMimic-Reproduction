"""Bound actual two-GPU torchrun and preserve exact original rank receipts."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from run_native_two_gpu_gate import BONES_CLIPS, BONES_QUALITY_COLLECTIONS


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motions',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--bones-throughput', action='store_true')
    mode.add_argument('--bones-quality', action='store_true')
    args=parser.parse_args()
    if args.bones_quality:
        steps, clips = BONES_QUALITY_COLLECTIONS, BONES_CLIPS
        timeout, program_timeout = 21600, 23700
    else:
        steps, clips = (1001, BONES_CLIPS) if args.bones_throughput else (21, 40)
        timeout, program_timeout = 2400, 2700
    real_steps = 8192 * steps
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    training=output/'training';scripts=Path(__file__).resolve().parent
    report=dict(status='starting',worker_timeout_seconds=timeout,whole_program_seconds=program_timeout,
        world_size=2,expected_global_real_steps=real_steps,quality_evaluation_completed=False,
        paper_results_reproduced=False,full_training_state_resume_available=False)
    (output/'result.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
    print('NATIVE_TWO_GPU_STAGE_START '+json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),
        log=str(output/'training.log'),worker_timeout_seconds=timeout)),flush=True)
    started=time.monotonic()
    command = ['timeout','--signal=TERM','--kill-after=30s',f'{timeout}s',
               sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1',
               '--nproc-per-node=2','--max-restarts=0',str(scripts/'run_native_two_gpu_gate.py'),
               '--motions',str(args.motions.resolve()),'--output',str(training),
               '--steps',str(steps),'--expected-clips',str(clips)]
    with (output/'training.log').open('w', encoding='utf-8') as stream:
        if args.bones_quality:
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  encoding="utf-8", errors="replace") as process:
                for line in process.stdout:
                    stream.write(line)
                    if line.startswith(("NATIVE_TWO_GPU_RANK_UPDATE ", "NATIVE_TWO_GPU_INTERMEDIATE_SNAPSHOT ")):
                        stream.flush()
                        print(line.rstrip(), flush=True)
                returncode = process.wait()
        else:
            returncode = subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT).returncode
    report.update(status='failed',worker_returncode=returncode,
                  worker_elapsed_seconds=time.monotonic()-started)
    paths=[training/'result.json']
    paths+=[training/f'rank-{rank}'/name for rank in range(2) for name in
            ('resolved-config.json','result.json')]
    try:
        if returncode:raise RuntimeError(f'torchrun exited {returncode}')
        parent=json.loads(paths[0].read_bytes())
        assert parent['status']=='two_rank_native_training_completed'
        assert parent['world_size']==2 and parent['backend']=='nccl'
        assert parent['global_simulator_steps']==real_steps
        participants=parent['participants']
        assert [row['rank'] for row in participants]==[0,1]
        assert participants[0]['final_model_state_hashes']==participants[1]['final_model_state_hashes']
        for rank,row in enumerate(participants):
            folder=training/f'rank-{rank}'
            original=json.loads((folder/'result.json').read_bytes())
            assert original==row and row['status']=='two_rank_native_participant_completed'
            assert row['local_simulator_steps']==real_steps // 2 and row['local_collection_steps']==steps
            assert row['actor_real_observation_count']==real_steps and row['all_native_learned_parameters_changed']
            assert row['clips'] == clips
            assert all(count>0 for count in row['optimizer_steps'].values())
            assert row['resolved_config_sha256']==digest(folder/'resolved-config.json')
            assert len(row['checkpoints'])==3
            for record in row['checkpoints']:
                path=Path(record['path']).resolve()
                assert path.parent==folder and path.name==record['file']
                assert record['native_model_readback_verified']
                assert path.stat().st_size==record['bytes'] and digest(path)==record['sha256']
        if args.bones_quality:
            snaps = participants[0]["intermediate_snapshots"]
            assert [row["collection_steps_per_rank"] for row in snaps] == [10001, 20001, 30001]
            for row in snaps:
                step = row["collection_steps_per_rank"]
                directory = training / "rank-0/checkpoint-series" / f"step-{step:09d}"
                assert row == json.loads((directory / "manifest.json").read_bytes())
                for record in row["files"]:
                    path = Path(record["path"]).resolve()
                    assert path.parent == directory and path.name == record["file"]
                    assert path.stat().st_size == record["bytes"] and digest(path) == record["sha256"]
            report["intermediate_rank0_snapshots_verified"] = len(snaps)
        if args.bones_throughput:
            elapsed = max(row['steady_last_800_collections_seconds'] for row in participants)
            assert 0 < elapsed < report['worker_elapsed_seconds']
            report['steady_window'] = dict(first_excluded_collection=201, last_included_collection=1001,
                seconds_slowest_rank=elapsed, global_simulator_transitions=800*8192,
                global_transitions_per_second=800*8192/elapsed)
        report.update(status='two_GPU_native_gate_completed',global_real_simulator_steps=real_steps,
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
