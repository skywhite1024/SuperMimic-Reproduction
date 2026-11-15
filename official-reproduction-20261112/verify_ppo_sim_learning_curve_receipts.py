"""Recover the39 original finite PPO-Sim receipts and verify real counters/full40 points."""
import argparse
import base64
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import zlib

PUBLIC=Path(__file__).parent
POINTS=[128,256,512,1024,1536,2048]
FILES=('result.json','protocol.json','accounting.json','audit/summary.json','audit/initial-states.json')
NAMES=[f'completed-gate-eval/evaluation/{name}' for name in FILES]
NAMES += [f'eval-{step:06d}/evaluation/{name}' for step in POINTS for name in FILES]
NAMES += ['training/resolved-config.json','training/checkpoint-manifest.json','training/result.json','result.json']


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--download',type=Path,required=True)
    parser.add_argument('--run-stem',default='ppo-sim-2048-1115c')
    parser.add_argument('--job-uid',default='bb89a2d9e06042dd9426fc52c0eb8854')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--recovery-plan',type=Path)
    parser.add_argument('--original-failure-log',type=Path)
    parser.add_argument('--selected-messages',action='store_true',
                        help='Input contains selected original timestamped messages rather than the whole job stdout')
    args=parser.parse_args()
    assert bool(args.recovery_plan)==bool(args.original_failure_log)
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    assert re.fullmatch(r'ppo-sim-2048-\d{4}[a-z]+',args.run_stem)
    assert re.fullmatch(r'[a-f0-9]{32}',args.job_uid)
    root=f'/mnt/afs/260010076/SuperMimic/official-20261112/runs/{args.run_stem}/'
    compressed=args.download.read_bytes();raw=gzip.decompress(compressed)
    original=raw.decode('utf-8')
    clean='\n'.join(re.sub(r'^\[[0-9TZ: .+-]+\] ','',line) for line in original.splitlines())+'\n'
    assert clean.count('PPO_SIM_CURVE_RECEIPTS_BEGIN')==clean.count('PPO_SIM_CURVE_RECEIPTS_END')==1
    text=clean.split('PPO_SIM_CURVE_RECEIPTS_BEGIN\n',1)[1].split('PPO_SIM_CURVE_RECEIPTS_END',1)[0]
    (output/'offline-log.txt.gz').write_bytes(compressed)
    metadata,chunks={},{}
    for line in text.splitlines():
        if line.startswith('RECEIPT_META '):
            row=json.loads(line.removeprefix('RECEIPT_META '));assert row['i'] not in metadata
            metadata[row['i']]=row
        match=re.fullmatch(r'RECEIPT_DATA (\d+) (\d+) (\S+)',line)
        if match:
            i,n=map(int,match.group(1,2));assert n not in chunks.setdefault(i,{})
            chunks[i][n]=match[3]
    assert set(metadata)==set(chunks)==set(range(len(metadata)))
    assert len({row['path'] for row in metadata.values()})==len(metadata)
    recovered=[]
    for i,row in metadata.items():
        assert row['path'].startswith(root);name=row['path'].removeprefix(root);assert name in NAMES
        assert set(chunks[i])==set(range(row['chunks']))
        data=zlib.decompress(base64.b64decode(''.join(chunks[i][n] for n in range(row['chunks'])),validate=True))
        assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
        json.loads(data);path=output/name;path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():assert path.read_bytes()==data, 'Previously recovered original changed: '+name
        else:path.write_bytes(data)
        recovered.append(name)
    def read(name):return json.loads((output/name).read_bytes())
    def sha(name):return hashlib.sha256((output/name).read_bytes()).hexdigest()
    def write(name,value):(output/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8',newline='\n')
    write('receipt-ledger.json',dict(status='original_receipts_byte_counts_and_SHA256_verified',
        uid=args.job_uid,receipts=list(metadata.values())))
    write('download-ledger.json',dict(download_file=args.download.name,lines=len(original.splitlines()),
        gzip_bytes=len(compressed),gzip_sha256=hashlib.sha256(compressed).hexdigest(),decompressed_bytes=len(raw),
        decompressed_sha256=hashlib.sha256(raw).hexdigest(),original_timestamped_messages_preserved=True,
        whole_job_stdout=not args.selected_messages,
        transport_scope='selected original messages' if args.selected_messages else 'complete offline job stdout'))
    summary=read('result.json')
    if summary['status']!='complete':
        write('validation.json',dict(status='originals_recovered_finite_experiment_incomplete',
            main_status=summary['status'],recovered=recovered,missing=sorted(set(NAMES)-set(recovered)),
            points=summary.get('points',[]),paper_results_reproduced=False))
        raise RuntimeError('Finite PPO-Sim incomplete; every returned original preserved')
    assert len(metadata)==39 and recovered==NAMES
    plan=json.loads(args.plan.read_bytes())
    assert plan['uid']==args.job_uid
    provenance=clean
    original_exit=0;recovery_uid=None
    if args.original_failure_log:
        failed=args.original_failure_log.read_bytes()
        provenance='\n'.join(re.sub(r'^\[[0-9TZ: .+-]+\] ','',line) for line in gzip.decompress(failed).decode().splitlines())
        assert 'PPO_SIM_CURVE_JOB_EXIT=1' in provenance
        assert 'Only explicitly named JSON receipts up to 2 MB are accepted' in provenance
        assert 'PPO_SIM_RESULT_REEXPORT_COMPLETED' in clean
        assert 'COMPLETED_RESULT_RECOVERY_PIPELINE_COMPLETED' in clean
        assert 'COMPLETED_RESULT_RECOVERY_JOB_EXIT=0' in clean
        recovery=json.loads(args.recovery_plan.read_bytes());recovery_uid=recovery['uid']
        assert 'COMPLETED_RESULT_RECOVERY_BOOTSTRAP_SHA256 '+recovery['command_sha256'] in clean
        assert all('COMPLETED_RESULT_RECOVERY_SOURCE_SHA256 '+name+' '+value in clean for name,value in recovery['source_files'].items())
        (output/'original-failed-job-log.txt.gz').write_bytes(failed)
        original_exit=1
    else:
        assert 'PPO_SIM_CURVE_PIPELINE_COMPLETED' in clean and 'PPO_SIM_CURVE_JOB_EXIT=0' in clean
    assert 'RESUMPTION_BOOTSTRAP_SHA256 '+plan['command_sha256'] in provenance
    assert all('PPO_SIM_CURVE_SOURCE_SHA256 '+name+' '+value in provenance for name,value in plan['source_files'].items())
    cfg,manifest,training=[read(name) for name in NAMES[-4:-1]]
    assert training['status']=='real_G1_simulator_PPO_finite_training_completed'
    assert cfg['training_algorithm']=='PPO-Sim' and cfg['no_world_model_constructed']
    assert cfg['outer_updates']==training['ppo_update_calls']==2048
    assert cfg['rollout_steps']==24 and cfg['num_envs']==8192 and cfg['seed']==42
    assert training['collection_steps']==49152 and training['simulator_steps']==402653184
    assert training['world_model_updates']==cfg['world_model_updates']==0
    assert training['imagined_transitions']==cfg['imagined_transitions']==0
    assert training['source_config_sha256']==sha(NAMES[-4])
    assert training['checkpoint_manifest_sha256']==sha(NAMES[-3]) and training['checkpoint_sets']==manifest
    assert [row['ppo_update'] for row in manifest]==[3,*POINTS]
    assert len(training['updates'])==2048
    previous_clocks={key:-1. for key in ('update_complete_seconds_since_program_entry','update_complete_seconds_since_training_entry')}
    for index,row in enumerate(training['updates'],1):
        assert row['update']==index and row['collection_steps']==index*24 and row['simulator_steps']==index*196608
        assert row['native_actor_normalization_count']==index*196608
        assert row['native_model_parameters_finite'] and all(math.isfinite(value) for value in row['losses'].values())
        assert row['stored_rewards_match_native_RSL_timeout_bootstrap'] and row['stored_dones_match_simulator']
        assert row['world_model_updates']==row['imagined_transitions']==0
        assert row['sampler_distribution_updates']==index*24//cfg['reset_sampler_cfg']['horizon']
        if index>1:
            assert row['actor_before']==training['updates'][index-2]['actor_after']
            assert row['critic_before']==training['updates'][index-2]['critic_after']
        for key in previous_clocks:
            assert math.isfinite(row[key]) and row[key]>=previous_clocks[key];previous_clocks[key]=row[key]
    assert training['initial_model_state_hashes']['actor']==training['updates'][0]['actor_before']
    assert training['final_model_state_hashes']['actor']==training['updates'][-1]['actor_after']
    assert training['initial_model_state_hashes']['critic']==training['updates'][0]['critic_before']
    assert training['final_model_state_hashes']['critic']==training['updates'][-1]['critic_after']
    assert any(row['native_actor_parameters_changed'] for row in training['updates'])
    assert [row['update'] for row in training['updates'] if 'trace' in row]==[1,3,*POINTS]
    assert all(row['trace']['readback_verified'] for row in training['updates'] if 'trace' in row)
    for row in manifest:
        assert row['collection_steps']==row['ppo_update']*24
        assert len(row['files'])==3 and {item['file'] for item in row['files']}=={'actor.pt','critic.pt','ppo.pt'}
        assert all(item['model_state_readback_verified'] for item in row['files'])
    canonical=json.loads((PUBLIC/'manual-reset-v2-lafan40-protocol.json').read_bytes())['motion_files']
    canonical_rows=json.loads((PUBLIC/'manual-reset-v2-lafan40-result.json').read_bytes())['records']
    motions={row['index']:(row['name'],row['reference_frames']) for row in canonical_rows}
    old=PUBLIC/'ppo-sim-integration-1113aj/training'
    old_hashes={name:hashlib.sha256((old/name).read_bytes()).hexdigest() for name in
                ('result.json','resolved-config.json','checkpoint-manifest.json')}
    cases=[(3,'completed-gate-eval/evaluation/',summary['gate_evaluation'],False)]
    assert [row['ppo_update'] for row in summary['points']]==POINTS
    cases += [(update,f'eval-{update:06d}/evaluation/',declared,True)
              for update,declared in zip(POINTS,summary['points'],strict=True)]
    curve=[];gate=None
    for update,folder,declared,is_main in cases:
        result,protocol,accounting,audit,initial=[read(folder+name) for name in FILES]
        assert result['status']=='full_corpus_evaluation_completed'
        assert protocol['manual_reset_version']==result['manual_reset_version']==2
        assert len(result['records'])==result['clips']==protocol['clips']==accounting['clips']==40
        assert protocol['motion_files']==canonical and accounting['reference_frames']==441140
        assert sorted(row['index'] for row in result['records'])==list(range(40))
        assert all((row['name'],row['reference_frames'])==motions[row['index']] for row in result['records'])
        selected=protocol['checkpoint_selection'];assert selected==declared['checkpoint_selection']
        assert protocol['training_algorithm']=='ppo-sim' and selected['no_world_model_loaded']
        assert selected['deterministic_native_actor_mean'] and selected['evaluated_ppo_update']==update
        if is_main:
            assert selected['parent_result_sha256']==sha(NAMES[-2])
            assert selected['resolved_config_sha256']==sha(NAMES[-4]) and selected['manifest_sha256']==sha(NAMES[-3])
            assert selected['parent_ppo_updates']==2048 and selected['parent_collection_steps']==49152
            chosen=next(row for row in manifest if row['ppo_update']==update)
            assert selected['files']==chosen['files']
        else:
            assert selected['parent_result_sha256']==old_hashes['result.json']
            assert selected['resolved_config_sha256']==old_hashes['resolved-config.json']
            assert selected['manifest_sha256']==old_hashes['checkpoint-manifest.json']
            assert selected['parent_ppo_updates']==3
        assert selected['evaluated_collection_steps']==update*24
        assert accounting['result_sha256']==audit['result_sha256']==sha(folder+'result.json')
        assert accounting['protocol_sha256']==initial['protocol_sha256']==sha(folder+'protocol.json')
        assert audit['accounting_sha256']==sha(folder+'accounting.json')
        assert initial['trajectory_integrity_summary_sha256']==sha(folder+'audit/summary.json')
        assert audit['all_trajectory_arrays_verified'] and audit['verified_clips']==40
        assert initial['all_saved_initial_states_verified'] and initial['saved_initial_states_verified']==40
        assert audit['verified_steps']==sum(row['steps'] for row in result['records'])
        assert result['successes']==accounting['successes']==audit['verified_successes']
        assert accounting['successes']+accounting['failures']==40 and accounting['maximum_step_reward']==.1
        for row in result['records']:
            assert 0<row['steps']<=row['reference_frames'] and math.isfinite(row['reward_sum'])
            success='motion_end' in row['termination_terms'] and not any(
                not protocol['termination_terms'][name]['time_out'] for name in row['termination_terms'])
            assert success==row['success']
            if success:assert row['steps']==row['reference_frames']
        mean=sum(row['reward_sum']/row['reference_frames']/.1 for row in result['records'])/40
        assert math.isclose(mean,accounting['mean_normalized_full_motion_return'],rel_tol=1e-13)
        assert declared['status']=='complete' and math.isclose(mean,declared['mean_normalized_full_motion_return'],rel_tol=1e-13)
        for name in FILES:assert declared['receipts'][name]==sha(folder+name)
        point=dict(ppo_update=update,collection_steps=update*24,real_simulator_steps=update*196608,
            successes=accounting['successes'],failures=accounting['failures'],mean_normalized_full_motion_return=mean,
            saved_steps=audit['verified_steps'],result_sha256=sha(folder+'result.json'))
        if is_main:point['checkpoint_timing']=declared['checkpoint_timing'];curve.append(point)
        else:gate=point
    validation=dict(status='finite2048_PPO_Sim_training_and_six_full40_points_originals_verified',
        original_JSON=39,original_byte_counts_and_SHA256_verified=True,
        original_job_uid=args.job_uid,original_job_exit_code=original_exit,
        original_receipt_export_failed=original_exit==1,recovery_job_uid=recovery_uid,
        original_and_recovery_source_hashes_bound=True,scientific_training_and_evaluation_complete=True,
        PPO_update_calls=2048,real_simulator_steps=402653184,imagined_transitions=0,world_model_updates=0,
        native_actor_and_critic_hash_chain_verified=True,all_real_reward_done_storage_checks_verified=True,
        all_normalization_counts_are_real_observations=True,retained_model_sets=7,saved_trace_updates=8,
        gate_existing_three_update_model=gate,main_points=curve,
        gate_original_files_and_all_six_main_full40_points_verified=True,
        all_saved_arrays_and_initial_states_audited_on_cloud=True,
        raw_NPZ_or_weights_downloaded=False,paper_PPO_hyperparameters_confirmed=False,
        full18h_or_multiGPU_comparison_complete=False,paper_results_reproduced=False)
    write('validation.json',validation);print(json.dumps(validation,indent=2))


if __name__=='__main__':main()
