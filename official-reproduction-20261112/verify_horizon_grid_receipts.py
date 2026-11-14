"""Recover scoped horizon-grid originals, including partial and failed phases."""
import argparse
import base64
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import zlib


def sha(data):return hashlib.sha256(data).hexdigest()


def decode_group(text, root, folder, allowed):
    metadata={};chunks={}
    for line in text.splitlines():
        if line.startswith('RECEIPT_META '):
            item=json.loads(line.removeprefix('RECEIPT_META '))
            assert item['i'] not in metadata;metadata[item['i']]=item
        match=re.fullmatch(r'RECEIPT_DATA (\d+) (\d+) (\S+)',line)
        if match:
            i,n=map(int,match.group(1,2));assert n not in chunks.setdefault(i,{})
            chunks[i][n]=match[3]
    assert metadata and set(metadata)==set(chunks)==set(range(len(metadata)))
    names=[]
    for i,item in sorted(metadata.items()):
        assert item['path'].startswith(root+'/')
        name=item['path'].removeprefix(root+'/');assert name in allowed and name not in names
        assert set(chunks[i])==set(range(item['chunks']))
        data=zlib.decompress(base64.b64decode(''.join(chunks[i][n] for n in range(item['chunks'])),validate=True))
        assert len(data)==item['bytes'] and sha(data)==item['sha256'];json.loads(data)
        path=folder/name;path.parent.mkdir(parents=True,exist_ok=True)
        if path.is_file():assert path.read_bytes()==data, 'Previously recovered original changed: '+name
        else:path.write_bytes(data)
        names.append(name)
    return names,list(metadata.values())


def verify_phase(folder,horizon,critic,phase,canonical,identities):
    read=lambda name:json.loads((folder/name).read_bytes())
    digest=lambda name:sha((folder/name).read_bytes())
    report=read('result.json')
    if report['status']!='complete':
        return dict(status='original_phase_failed_or_incomplete',reported_status=report['status'],
                    error=report.get('error'),points=report.get('points',[]))
    steps,interval,point=(21,10,11) if phase=='integration' else (16001,4000,16001)
    training,cfg,timing=[read('training/'+name) for name in ('result.json','resolved-config.json','checkpoint-times.json')]
    assert training['status']=='pilot_completed' and training['collection_steps']==steps
    assert training['critic_enabled']==critic and cfg['runner']['policy']['horizon']==horizon
    assert cfg['runner']['policy']['gamma']==.99 and ('lam' in cfg['runner']['policy'])==critic
    assert cfg['seed']==training['seed']==42 and cfg['num_envs']==training['num_envs']==4096
    assert cfg['runner']['save_interval']==interval and cfg['training_profile']=='paper-a5-single-gpu'
    assert cfg['runner']['history_length']==10 and cfg['runner']['actor_obs']=='proprio'
    assert cfg['goal_assembly_backend']=='eager' and not cfg['vendor_files_modified']
    expected=list(range(1,steps+1,interval));snapshots=training['retained_checkpoint_series']
    assert [r['collection_steps'] for r in snapshots]==[r['collection_steps'] for r in timing['checkpoints']]==expected
    models={'world_model.pt','actor.pt'}|({'critic.pt'} if critic else set())
    assert all({r['file'] for r in snapshot['files']}==models for snapshot in snapshots)
    final={Path(r['path']).name:r['sha256'] for r in training['checkpoints']}
    assert final=={r['file']:r['sha256'] for r in snapshots[-1]['files']}
    assert all(len({next(r['sha256'] for r in snapshot['files'] if r['file']==name) for snapshot in snapshots})==len(snapshots) for name in models)
    previous={key:-1. for key in ('seconds_since_program_entry','seconds_since_runner_run_entry')}
    for row in timing['checkpoints']:
        for event in ('callback_entry_after_native_save','retained_snapshot_complete','offline_artifact_logging_returned'):
            for key in previous:
                value=row[event][key];assert math.isfinite(value) and value>=previous[key]
                previous[key]=value
    prefix=f'eval-{point:09d}/'
    result,protocol,accounting,audit,initial=[read(prefix+name) for name in
        ('result.json','protocol.json','accounting.json','audit/summary.json','audit/initial-states.json')]
    assert result['status']=='full_corpus_evaluation_completed'
    assert result['clips']==protocol['clips']==accounting['clips']==len(result['records'])==40
    assert protocol['manual_reset_version']==result['manual_reset_version']==2
    assert protocol['motion_files']==canonical and protocol['num_envs']==40 and protocol['batch_order']=='index'
    assert not protocol['domain_randomization'] and not protocol['observation_noise']
    selected=protocol['checkpoint_selection']
    assert selected['parent_collection_steps']==steps and selected['evaluated_collection_steps']==point
    assert selected['parent_result_sha256']==digest('training/result.json')
    assert selected['resolved_config_sha256']==digest('training/resolved-config.json')
    snapshot=next(r for r in snapshots if r['collection_steps']==point)
    assert {Path(r['path']).name:r['sha256'] for r in selected['files']}=={r['file']:r['sha256'] for r in snapshot['files']}
    assert accounting['reference_frames']==441140 and accounting['maximum_step_reward']==.1
    assert sorted(r['index'] for r in result['records'])==list(range(40))
    total=0.;successes=0
    for row in result['records']:
        assert (row['name'],row['reference_frames'])==identities[row['index']]
        assert math.isfinite(row['reward_sum']) and 0<row['steps']<=row['reference_frames']
        terms=row['termination_terms']
        success='motion_end' in terms and not any(not protocol['termination_terms'][name]['time_out'] for name in terms)
        assert success==row['success']
        if success:assert row['steps']==row['reference_frames']
        successes+=success;total+=row['reward_sum']/row['reference_frames']/.1
    mean=total/40
    assert math.isclose(mean,accounting['mean_normalized_full_motion_return'],rel_tol=1e-13)
    assert successes==result['successes']==accounting['successes']==audit['verified_successes']
    assert successes+accounting['failures']==40
    assert accounting['result_sha256']==audit['result_sha256']==digest(prefix+'result.json')
    assert accounting['protocol_sha256']==initial['protocol_sha256']==digest(prefix+'protocol.json')
    assert audit['accounting_sha256']==digest(prefix+'accounting.json')
    assert initial['trajectory_integrity_summary_sha256']==digest(prefix+'audit/summary.json')
    assert audit['all_trajectory_arrays_verified'] and audit['verified_clips']==40
    assert initial['all_saved_initial_states_verified'] and initial['saved_initial_states_verified']==40
    assert audit['verified_steps']==sum(r['steps'] for r in result['records'])
    declared=report['points'];assert len(declared)==1 and declared[0]['collection_steps']==point and declared[0]['status']=='complete'
    assert math.isclose(declared[0]['mean_normalized_full_motion_return'],mean,rel_tol=1e-13)
    for field,name in [('result_sha256','result.json'),('protocol_sha256','protocol.json'),('accounting_sha256','accounting.json'),
                       ('trajectory_audit_sha256','audit/summary.json'),('initial_states_audit_sha256','audit/initial-states.json')]:
        assert declared[0][field]==digest(prefix+name)
    return dict(status='native_horizon_phase_originals_verified',horizon=horizon,critic_enabled=critic,phase=phase,
        training_collections=steps,evaluated_collection=point,successes=successes,failures=40-successes,
        mean_normalized_full_motion_return=mean,retained_model_sets=len(snapshots),model_files_per_set=len(models),
        fixed40_reference_frames=441140,all_cloud_trajectory_arrays_and_initial_states_verified=True,
        original_result_sha256=digest(prefix+'result.json'),raw_weights_or_arrays_downloaded=False,
        single_seed=True,paper_results_reproduced=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log',type=Path,required=True)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--require-complete',action='store_true')
    args=parser.parse_args();plan=json.loads(args.plan.read_bytes());source=args.log.read_bytes()
    raw=gzip.decompress(source) if args.log.suffix=='.gz' else source
    clean='\n'.join(re.sub(r'^\[[0-9TZ: .+-]+\] ','',line) for line in raw.decode().splitlines())
    assert 'HORIZON_GRID_BOOTSTRAP_SHA256 '+plan['command_sha256'] in clean
    assert all('HORIZON_GRID_SOURCE_SHA256 '+name+' '+value in clean for name,value in plan['source_files'].items())
    expected={f'h{h:03d}-'+variant+'/'+phase:(h,variant=='critic',phase)
              for h in plan['horizons'] for variant in ('critic','no-critic') for phase in ('integration','main')}
    public=Path(__file__).parent
    canonical=json.loads((public/'manual-reset-v2-lafan40-protocol.json').read_bytes())['motion_files']
    records=json.loads((public/'manual-reset-v2-lafan40-result.json').read_bytes())['records']
    identities={r['index']:(r['name'],r['reference_frames']) for r in records}
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True);groups=[];receipts=[]
    for match in re.finditer(r'^HORIZON_GRID_ORIGINALS_BEGIN ([^\n]+)\n(.*?)^HORIZON_GRID_ORIGINALS_END \1$',clean,re.M|re.S):
        label,text=match.group(1,2);assert label in expected and label not in [r['label'] for r in groups]
        h,critic,phase=expected[label];point=11 if phase=='integration' else 16001
        allowed=['training/'+name for name in ('result.json','resolved-config.json','checkpoint-times.json')]
        allowed += [f'eval-{point:09d}/'+name for name in ('result.json','protocol.json','accounting.json','audit/summary.json','audit/initial-states.json')]
        allowed+=['result.json'];folder=output/label
        names,ledger=decode_group(text,plan['output']+'/'+label,folder,allowed)
        validation=verify_phase(folder,h,critic,phase,canonical,identities)
        if validation['status']=='native_horizon_phase_originals_verified':assert names==allowed
        (folder/'validation.json').write_text(json.dumps(validation,indent=2)+'\n',encoding='utf-8',newline='\n')
        groups.append(dict(label=label,original_JSONs=len(names),**validation));receipts.extend(ledger)
    summary=re.search(r'^HORIZON_GRID_SUMMARY_BEGIN\n(.*?)^HORIZON_GRID_SUMMARY_END$',clean,re.M|re.S)
    if summary:
        _,ledger=decode_group(summary[1],plan['output'],output,['result.json']);receipts.extend(ledger)
    complete=len(groups)==8 and all(r['status']=='native_horizon_phase_originals_verified' for r in groups) and summary is not None
    if complete:
        assert json.loads((output/'result.json').read_bytes())['status']=='complete'
        assert 'HORIZON_GRID_PIPELINE_COMPLETED' in clean and 'HORIZON_GRID_JOB_EXIT=0' in clean
    result=dict(status='complete_four_case_grid_originals_verified' if complete else 'partial_grid_originals_verified',
        job_uid=plan['uid'],source_commit=plan['source_commit'],log_original_bytes=len(raw),log_sha256=sha(raw),
        transport='downloaded offline gzip' if args.log.suffix=='.gz' else 'captured live terminal text',
        all_recovered_JSON_original_byte_counts_and_SHA256_verified=True,
        original_JSONs=len(receipts),groups=groups,remaining_scopes=sorted(set(expected)-{r['label'] for r in groups}),
        source_and_model_hashes_bound=bool(groups) and all(r['status']=='native_horizon_phase_originals_verified' for r in groups),
        paper_training_episode_curves_completed=False,multiple_seeds=False,paper_results_reproduced=False)
    for name,value in [('validation.json',result),('receipt-ledger.json',dict(receipts=receipts))]:
        (output/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8',newline='\n')
    if args.require_complete:assert complete, 'Full grid not complete; recovered originals and validation retained'
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
