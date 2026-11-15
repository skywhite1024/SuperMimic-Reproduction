"""Recover finite PPO-WM batch-learning originals and recompute all18 clip scores."""
import argparse
import gzip
import json
import math
from pathlib import Path
import re

from verify_horizon_grid_receipts import decode_group,sha

POINTS=(64,256,1024,2048)
EVAL_FILES=('result.json','protocol.json','accounting.json','audit/summary.json','audit/initial-states.json')


def verify_complete(output,plan,names):
    read=lambda name:json.loads((output/name).read_bytes())
    digest=lambda name:sha((output/name).read_bytes())
    allowed=['motion-subset.json']+['training/'+name for name in ('resolved-config.json','initial-model-state.json','native-checkpoint-manifest.json','result.json')]
    allowed += [f'eval-{step:06d}/'+name for step in POINTS for name in EVAL_FILES]+['result.json']
    assert names==allowed and len(names)==26
    report=read('result.json');cfg=read('training/resolved-config.json');training=read('training/result.json')
    subset=read('motion-subset.json');initial=read('training/initial-model-state.json');manifest=read('training/native-checkpoint-manifest.json')
    assert report['status']=='complete' and report['training_result_sha256']==digest('training/result.json')
    assert subset['status']=='all18_original_motion_slices_preserved'
    assert subset['selected']==plan['motion_subset']['selected']
    assert subset['source_motion_files']==plan['motion_subset']['source_motion_files']
    assert subset['clips']==cfg['clips']==18 and subset['reference_frames']==cfg['frames']==220352
    assert training['status']=='real_G1_native_network_PPO_WM_finite_training_completed'
    assert cfg['training_algorithm']=='PPO-WM' and cfg['training_purpose']=='finite-training'
    assert cfg['seed']==42 and cfg['num_envs']==4096 and cfg['outer_iterations']==2048
    assert cfg['ppo_runner']['batch_size']==report['batch_size']==plan['imagined_batch_size']
    assert cfg['ppo_runner']['num_steps_per_env']==report['rollout_steps']==plan['steps_per_rollout']
    assert cfg['termination_storage']==training['termination_mode']=='released_uint8'
    assert cfg['goal_backend']=='author eager goal math' and cfg['vendor_modified'] is False
    assert training['source_config_sha256']==digest('training/resolved-config.json')
    assert training['native_checkpoint_manifest_sha256']==digest('training/native-checkpoint-manifest.json')
    assert manifest==training['retained_native_checkpoints']
    assert [row['ppo_update'] for row in manifest]==[64,256,512,1024,2048]
    assert training['outer_iterations']==training['real_native_ppo_updates']==training['world_model_update_calls']==len(training['updates'])==2048
    assert training['real_collections']==20480 and training['real_simulator_steps']==83886080
    batch_steps=plan['imagined_batch_size']*plan['steps_per_rollout']
    assert training['imagined_transitions']==2048*batch_steps==sum(r['imagined_steps'] for r in training['updates'])
    assert training['initial_state_hashes']==initial['state_hashes']
    assert initial['dimensions']['goal']==640 and initial['dimensions']['history']==10 and initial['dimensions']['proprio']==93
    assert training['all_three_native_models_changed_and_finite']
    assert all(training['initial_state_hashes'][key]!=training['final_state_hashes'][key] for key in ('actor','critic','world_model'))
    truncated=0
    for i,row in enumerate(training['updates'],1):
        assert row['update']==i and row['imagined_steps']==batch_steps and row['storage_dtype']=='torch.uint8'
        assert row['actor_before']!=row['actor_after'] and row['critic_before']!=row['critic_after']
        assert row['fractional_probability_positions']==row['fractional_positions_truncated_to_zero']
        assert 0<=row['probability_min']<=row['probability_max']<=1
        assert all(math.isfinite(value) for value in row['losses'].values())
        if row['trace'] is not None:assert row['trace']['readback_verified'] and row['trace']['bytes']>0
        truncated+=row['fractional_positions_truncated_to_zero']
    previous=-1.
    for row in manifest:
        update=row['ppo_update']
        assert row['real_simulator_steps']==update*40960 and row['imagined_steps']==update*batch_steps
        assert row['seconds_since_training_entry']>previous;previous=row['seconds_since_training_entry']
        assert {r['file'] for r in row['files']}=={'actor.pt','critic.pt','world_model.pt','released-actor-critic.pt'}
        assert all(r['native_model_readback_verified'] and r['bytes']>0 and len(r['sha256'])==64 for r in row['files'])
        original={Path(r['path']).name:r['sha256'] for r in row['source_released_checkpoints']}
        native={r['file']:r['sha256'] for r in row['files']}
        assert native['world_model.pt']==original['world_model.pt'] and native['released-actor-critic.pt']==original['actor.pt']
    assert [row['ppo_update'] for row in report['points']]==list(POINTS)
    curve=[]
    for update,declared in zip(POINTS,report['points'],strict=True):
        prefix=f'eval-{update:06d}/'
        result,protocol,accounting,audit,states=[read(prefix+name) for name in EVAL_FILES]
        assert result['status']=='full_corpus_evaluation_completed'
        assert protocol['training']==training and protocol['training_algorithm']=='ppo-wm'
        assert protocol['motion_files']==subset['packed_motion_files']
        assert protocol['manual_reset_version']==result['manual_reset_version']==2
        assert protocol['num_envs']==18 and protocol['batch_order']=='index' and protocol['seed']==42
        assert not protocol['domain_randomization'] and not protocol['observation_noise']
        selected=protocol['checkpoint_selection'];snapshot=next(r for r in manifest if r['ppo_update']==update)
        assert selected['files']==snapshot['files'] and selected['evaluated_ppo_update']==update
        assert selected['parent_result_sha256']==digest('training/result.json')
        assert selected['resolved_config_sha256']==digest('training/resolved-config.json')
        assert selected['manifest_sha256']==digest('training/native-checkpoint-manifest.json')
        assert len(result['records'])==result['clips']==accounting['clips']==18
        assert sorted(r['index'] for r in result['records'])==list(range(18))
        total=0.;successes=0;steps=0
        for row in result['records']:
            source=subset['selected'][row['index']]
            assert (row['name'],row['reference_frames'])==(source['name'],source['reference_frames'])
            assert 0<row['steps']<=row['reference_frames'] and math.isfinite(row['reward_sum'])
            terms=row['termination_terms']
            success='motion_end' in terms and not any(not protocol['termination_terms'][name]['time_out'] for name in terms)
            assert row['success']==success
            if success:assert row['steps']==row['reference_frames']
            total+=row['reward_sum']/row['reference_frames']/.1;successes+=success;steps+=row['steps']
        mean=total/18
        assert accounting['reference_frames']==220352 and accounting['maximum_step_reward']==.1
        assert successes==result['successes']==accounting['successes']==audit['verified_successes']
        assert successes+accounting['failures']==18
        assert accounting['result_sha256']==audit['result_sha256']==digest(prefix+'result.json')
        assert accounting['protocol_sha256']==states['protocol_sha256']==digest(prefix+'protocol.json')
        assert audit['accounting_sha256']==digest(prefix+'accounting.json')
        assert states['trajectory_integrity_summary_sha256']==digest(prefix+'audit/summary.json')
        assert audit['all_trajectory_arrays_verified'] and audit['verified_clips']==18 and audit['verified_steps']==steps
        assert states['all_saved_initial_states_verified'] and states['saved_initial_states_verified']==18
        assert math.isclose(mean,accounting['mean_normalized_full_motion_return'],rel_tol=1e-13)
        assert declared['status']=='complete' and math.isclose(mean,declared['mean_normalized_full_motion_return'],rel_tol=1e-13)
        assert all(declared['receipts'][name]==digest(prefix+name) for name in EVAL_FILES)
        curve.append(dict(ppo_update=update,successes=successes,failures=18-successes,
            mean_normalized_full_motion_return=mean,real_simulator_steps=update*40960,cumulative_imagined_steps=update*batch_steps))
    return dict(status='finite_PPO_WM_batch_training_and_all18_evaluation_originals_verified',
        outer_updates=2048,real_simulator_steps=83886080,imagined_steps=2048*batch_steps,
        retained_native_model_sets=5,all_cloud_arrays_and_initial_states_verified=True,
        released_uint8_fractional_positions_truncated=truncated,points=curve,
        author_exact_motion_membership_confirmed=False,PPO_hyperparameters_author_confirmed=False,
        first_order_same_subset_comparison_complete=False,paper_results_reproduced=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log',type=Path,required=True);parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--require-complete',action='store_true')
    args=parser.parse_args();plan=json.loads(args.plan.read_bytes());data=args.log.read_bytes()
    raw=gzip.decompress(data) if args.log.suffix=='.gz' else data
    clean='\n'.join(re.sub(r'^\[[0-9TZ: .+-]+\] ','',line) for line in raw.decode().splitlines())
    assert 'PPO_WM_BATCH_BOOTSTRAP_SHA256 '+plan['command_sha256'] in clean
    assert all('PPO_WM_BATCH_SOURCE_SHA256 '+name+' '+value in clean for name,value in plan['source_files'].items())
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    blocks=re.findall(r'^PPO_WM_BATCH_ORIGINALS_BEGIN\n(.*?)^PPO_WM_BATCH_ORIGINALS_END$',clean,re.M|re.S)
    assert len(blocks)<=1
    validation=dict(status='final_originals_not_yet_emitted',original_JSONs=0);ledger=[]
    if blocks:
        allowed=['motion-subset.json']+['training/'+name for name in ('resolved-config.json','initial-model-state.json','native-checkpoint-manifest.json','result.json')]
        allowed += [f'eval-{step:06d}/'+name for step in POINTS for name in EVAL_FILES]+['result.json']
        names,ledger=decode_group(blocks[0],plan['output'],output,allowed)
        report=json.loads((output/'result.json').read_bytes())
        validation.update(status='failed_or_incomplete_originals_retained',original_JSONs=len(names),reported_status=report['status'],error=report.get('error'))
        if report['status']=='complete':
            validation.update(verify_complete(output,plan,names))
            assert 'PPO_WM_BATCH_PIPELINE_COMPLETED' in clean and 'PPO_WM_BATCH_JOB_EXIT=0' in clean
    updates=[json.loads(x) for x in re.findall(r'^PPO_WM_NATIVE_UPDATE (.+)$',clean,re.M)]
    checkpoints=[json.loads(x) for x in re.findall(r'^PPO_WM_NATIVE_CHECKPOINT (.+)$',clean,re.M)]
    batch_steps=plan['imagined_batch_size']*plan['steps_per_rollout']
    assert [r['update'] for r in updates]==sorted({r['update'] for r in updates})
    assert all(r['imagined_steps']==batch_steps and all(math.isfinite(v) for v in r['losses'].values()) for r in updates)
    for row in checkpoints:
        assert row['real_simulator_steps']==row['ppo_update']*40960
        assert row['imagined_steps']==row['ppo_update']*batch_steps
    validation.update(last_logged_native_update=updates[-1]['update'] if updates else None,
        logged_checkpoints_after_cloud_tensor_and_byte_readback_checks=checkpoints,
        progress_tags_are_not_independent_model_or_quality_verification=True)
    validation.update(job_uid=plan['uid'],source_commit=plan['source_commit'],log_sha256=sha(raw),
        all_recovered_original_bytes_and_SHA256_verified=bool(ledger),raw_models_and_NPZ_downloaded=False)
    for name,value in [('validation.json',validation),('receipt-ledger.json',dict(receipts=ledger))]:
        (output/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8',newline='\n')
    if args.require_complete:assert validation['status']=='finite_PPO_WM_batch_training_and_all18_evaluation_originals_verified','Complete scientific result not available; written originals retained'
    print(json.dumps(validation,indent=2))


if __name__=='__main__':main()
