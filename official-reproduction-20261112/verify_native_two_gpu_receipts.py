"""Recover original NCCL gate receipts and verify their cross-rank invariants."""
import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path
import re
import zlib


def digest(data):return hashlib.sha256(data).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download',type=Path,required=True)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    plan=json.loads(args.plan.read_bytes());raw=gzip.decompress(args.download.read_bytes())
    clean='\n'.join(re.sub(r'^\[[0-9TZ: .+-]+\] ','',line) for line in raw.decode().splitlines())
    assert 'NATIVE_TWO_GPU_BOOTSTRAP_SHA256 '+plan['command_sha256'] in clean
    assert 'NATIVE_TWO_GPU_GATE_PIPELINE_COMPLETED' in clean and 'NATIVE_TWO_GPU_JOB_EXIT=0' in clean
    for name,sha in plan['source_files'].items():assert 'NATIVE_TWO_GPU_SOURCE_SHA256 '+name+' '+sha in clean
    metadata={};chunks={}
    for line in clean.splitlines():
        if line.startswith('RECEIPT_META '):
            item=json.loads(line.removeprefix('RECEIPT_META '))
            assert item['i'] not in metadata;metadata[item['i']]=item
        match=re.fullmatch(r'RECEIPT_DATA (\d+) (\d+) (\S+)',line)
        if match:
            i,n=map(int,match.group(1,2));assert n not in chunks.setdefault(i,{})
            chunks[i][n]=match[3]
    assert set(metadata)==set(chunks)==set(range(6))
    names=['training/result.json',*['training/rank-'+str(rank)+'/'+name for rank in range(2)
                                      for name in ('resolved-config.json','result.json')],'result.json']
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    for i,name in enumerate(names):
        item=metadata[i]
        assert item['path']==plan['output']+'/'+name and set(chunks[i])==set(range(item['chunks']))
        data=zlib.decompress(base64.b64decode(''.join(chunks[i][n] for n in range(item['chunks'])),validate=True))
        assert len(data)==item['bytes'] and digest(data)==item['sha256']
        json.loads(data);path=output/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    read=lambda name:json.loads((output/name).read_bytes())
    training=read('training/result.json');parent=read('result.json')
    assert training['status']=='two_rank_native_training_completed'
    assert parent['status']=='two_GPU_native_gate_completed' and parent['worker_returncode']==0
    assert parent['parent_result_sha256']==digest((output/'training/result.json').read_bytes())
    assert parent['native_weights_files_verified']==6 and parent['original_rank_config_and_results_bound']
    assert training['global_simulator_steps']==parent['global_real_simulator_steps']==172032
    ranks=training['participants'];assert [r['rank'] for r in ranks]==[0,1]
    assert {r['local_rank'] for r in ranks}=={0,1}
    assert ranks[0]['initial_real_state_sha256']!=ranks[1]['initial_real_state_sha256']
    shared=('initial_model_state_hashes','initial_learned_parameter_hashes','final_model_state_hashes',
            'final_learned_parameter_hashes','optimizer_steps','actor_real_observation_count',
            'shared_sampler_value_sha256','shared_sampler_count_sha256','shared_sampler_effort_sha256',
            'sim_context_state_sha256')
    assert all(ranks[0][key]==ranks[1][key] for key in shared)
    for rank,row in enumerate(ranks):
        assert row==read(f'training/rank-{rank}/result.json')
        cfg=read(f'training/rank-{rank}/resolved-config.json')
        assert digest((output/f'training/rank-{rank}/resolved-config.json').read_bytes())==row['resolved_config_sha256']
        assert row['status']=='two_rank_native_participant_completed' and row['backend']=='nccl'
        assert row['local_collection_steps']==21 and row['local_simulator_steps']==86016
        assert row['actor_real_observation_count']==172032 and row['all_native_learned_parameters_changed']
        assert all(value!=row['initial_learned_parameter_hashes'][name] for name,value in row['final_learned_parameter_hashes'].items())
        assert row['native_learner_outer_updates']==6
        assert row['optimizer_steps']==dict(world_model_optimizer_steps=60,actor_optimizer_steps=6,critic_optimizer_steps=120)
        assert row['sampler_real_records']==21 and row['sampler_distribution_updates']==1
        assert cfg['environment_seed']==42+rank and cfg['local_num_envs']==4096 and cfg['global_num_envs']==8192
        assert cfg['learner_CUDA_graphs_disabled'] and cfg['runner']['policy']['horizon']==32
        assert {r['file'] for r in row['checkpoints']}=={'world_model.pt','actor.pt','critic.pt'}
        assert all(r['native_model_readback_verified'] and r['bytes']>0 and len(r['sha256'])==64 for r in row['checkpoints'])
    validation=dict(status='actual_two_H100_NCCL_native_gate_originals_verified',job_uid=plan['uid'],
        original_JSON_receipts=6,all_byte_counts_and_SHA256_verified=True,source_files_bound_to_actual_log=True,
        source_commit=plan['source_commit'],H100=2,ranks=2,local_collection_steps=21,global_real_simulator_steps=172032,
        native_learner_outer_updates_per_rank=6,optimizer_steps_per_rank=ranks[0]['optimizer_steps'],
        learned_networks_changed_and_identical_across_ranks=True,independent_real_initial_states=True,
        actor_real_observation_count_both_ranks=172032,sampler_value_count_effort_hashes_identical=True,
        native_weight_files_cloud_hash_and_tensor_readback_verified=6,raw_weights_downloaded=False,
        learner_CUDA_graphs_disabled=True,quality_evaluation_completed=False,
        full_training_state_resume_available=False,paper_performance_equivalence=False,paper_results_reproduced=False)
    for name,value in [('validation.json',validation),('receipt-ledger.json',dict(receipts=list(metadata.values()))),
                       ('download-ledger.json',dict(lines=len(raw.splitlines()),gzip_sha256=digest(args.download.read_bytes()),
                           timestamped_stdout_sha256=digest(raw),original_timestamped_stdout_preserved=True))]:
        (output/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(validation,indent=2))


if __name__=='__main__':main()
