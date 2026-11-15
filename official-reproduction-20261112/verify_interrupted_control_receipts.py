"""Recover original interrupted-control evaluation receipts and recompute all points."""
import argparse
import gzip
import json
import math
from pathlib import Path
import re

from verify_horizon_grid_receipts import decode_group, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('log', 'plan', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_bytes())
    raw = args.log.read_bytes()
    if args.log.suffix == '.gz': raw = gzip.decompress(raw)
    clean = '\n'.join(re.sub(r'^\[[0-9TZ: .+-]+\] ', '', line) for line in raw.decode().splitlines())
    assert 'INTERRUPTED_CONTROL_BOOTSTRAP_SHA256 '+plan['command_sha256'] in clean
    assert all('INTERRUPTED_CONTROL_SOURCE_SHA256 '+name+' '+value in clean for name,value in plan['source_files'].items())
    points = plan['evaluation_collections']
    assert points == [1,4001,8001,12001,16001,20001]
    root = '/mnt/afs/260010076/SuperMimic/official-20261112'
    parent = plan['source_training_run'].removeprefix(root+'/')
    output = plan['output'].removeprefix(root+'/')
    evidence = 'launch/'+plan['source_commit']+'-'+Path(plan['output']).name+'/interrupted-control-source-evidence-1116.json'
    meta = [parent+'/'+name for name in ('resolved-config.json','checkpoint-times.json','learner-counts.json')]
    meta += [parent+f'/checkpoint-series/step-{step:09d}/manifest.json' for step in points]
    files = [evidence, output+'/parent-readback.json', *meta]
    tails = ('result.json','protocol.json','accounting.json','audit/summary.json','audit/initial-states.json')
    files += [output+f'/eval-{step:09d}/'+name for step in points for name in tails]
    files += [output+'/result.json']
    assert len(files)==plan['expected_original_JSONs_if_complete']==42
    groups = re.findall(r'^INTERRUPTED_CONTROL_ORIGINALS_BEGIN\n(.*?)^INTERRUPTED_CONTROL_ORIGINALS_END$',clean,re.M|re.S)
    assert len(groups)<=1
    args.output.mkdir(parents=True,exist_ok=True)
    validation = dict(status='originals_not_yet_emitted',original_JSONs=0,paper_results_reproduced=False)
    ledger = []
    if groups:
        names,ledger = decode_group(groups[0],root,args.output,files)
        read = lambda p:json.loads((args.output/p).read_bytes())
        digest = lambda p:sha((args.output/p).read_bytes())
        report = read(output+'/result.json')
        validation.update(status='original_failed_or_partial_job_retained',original_JSONs=len(names),reported_status=report['status'])
        if report['status']=='complete':
            assert names==files and 'INTERRUPTED_CONTROL_COMPLETED' in clean and 'INTERRUPTED_CONTROL_JOB_EXIT=0' in clean
            observed = read(evidence)
            assert digest(evidence)==plan['source_files']['interrupted-control-source-evidence-1116.json']
            assert observed['parent_platform_status']=='stopped_by_user' and observed['parent_job_uid']==plan['source_stopped_job_uid']
            assert observed['original_training_directory']==plan['source_training_run']
            cfg = read(parent+'/resolved-config.json')
            assert cfg['seed']==42 and cfg['num_envs']==4096 and cfg['collection_steps']==64001
            assert cfg['goal_assembly_backend']=='eager' and cfg['corpus']['name']=='lafan-locomotion18'
            assert cfg['corpus']['clips']==18 and cfg['corpus']['frames']==220352
            parent_readback = read(output+'/parent-readback.json')
            assert parent_readback['model_sets']==6 and parent_readback['parent_training_complete'] is False
            assert parent_readback['original_stop_evidence_SHA256']==digest(evidence)
            assert all(parent_readback['original_metadata_SHA256'][root+'/'+p]==digest(p) for p in meta)
            assert report['training_reexecuted'] is False and report['parent_training_complete'] is False
            assert report['original_parent_metadata_unchanged'] and report['full_training_resume_available'] is False
            timing = read(parent+'/checkpoint-times.json')
            assert [r['collection_steps'] for r in timing['checkpoints']]==points
            subset = json.loads(Path(__file__).with_name('ppo-wm-batch-learning-1115').joinpath(
                'ppo-wm-4k32-1115i/motion-subset.json').read_bytes())
            data = []
            motion_files = None
            for step,declared,expected in zip(points,report['points'],observed['logged_snapshots'],strict=True):
                manifest = parent+f'/checkpoint-series/step-{step:09d}/manifest.json'
                assert read(manifest)==expected and expected['collection_steps']==step
                prefix = output+f'/eval-{step:09d}/'
                result,protocol,account,arrays,states=[read(prefix+p) for p in tails]
                assert result['status']=='full_corpus_evaluation_completed'
                assert result['clips']==protocol['clips']==account['clips']==len(result['records'])==18
                assert protocol['manual_reset_version']==result['manual_reset_version']==2
                assert protocol['num_envs']==18 and protocol['batch_order']=='index'
                assert protocol['domain_randomization'] is False and protocol['observation_noise'] is False
                if motion_files is None:motion_files=protocol['motion_files']
                assert protocol['motion_files']==motion_files
                assert motion_files==subset['packed_motion_files']
                selected=protocol['checkpoint_selection']
                assert selected['parent_training_complete'] is False and selected['evaluated_collection_steps']==step
                assert selected['parent_stop_evidence_sha256']==digest(evidence)
                assert selected['manifest_sha256']==digest(manifest)
                assert selected['resolved_config_sha256']==digest(parent+'/resolved-config.json')
                assert {Path(r['path']).name:r['sha256'] for r in selected['files']}=={r['file']:r['sha256'] for r in expected['files']}
                assert sorted(r['index'] for r in result['records'])==list(range(18))
                successes=steps=frames=0;total=0.
                for row in result['records']:
                    identity=subset['selected'][row['index']]
                    assert (row['name'],row['reference_frames'])==(identity['name'],identity['reference_frames'])
                    success='motion_end' in row['termination_terms'] and not any(not protocol['termination_terms'][name]['time_out'] for name in row['termination_terms'])
                    assert row['success']==success and 0<row['steps']<=row['reference_frames']
                    assert not success or row['steps']==row['reference_frames']
                    assert math.isfinite(row['reward_sum']) and row['reward_sum']<=row['steps']*.1+1e-4
                    total+=row['reward_sum']/(row['reference_frames']*.1)
                    successes+=int(success);steps+=row['steps'];frames+=row['reference_frames']
                mean=total/18
                assert successes==result['successes']==account['successes']==arrays['verified_successes']
                assert frames==220352 and successes+result['failures']==18
                assert math.isclose(mean,account['mean_normalized_full_motion_return'],rel_tol=1e-13)
                assert declared['collection_steps']==step and declared['status']=='complete'
                assert math.isclose(mean,declared['mean_normalized_full_motion_return'],rel_tol=1e-13)
                assert account['result_sha256']==arrays['result_sha256']==digest(prefix+'result.json')
                assert account['protocol_sha256']==states['protocol_sha256']==digest(prefix+'protocol.json')
                assert arrays['accounting_sha256']==digest(prefix+'accounting.json')
                assert states['trajectory_integrity_summary_sha256']==digest(prefix+'audit/summary.json')
                assert arrays['all_trajectory_arrays_verified'] and arrays['verified_steps']==steps and arrays['verified_clips']==18
                assert states['all_saved_initial_states_verified'] and states['saved_initial_states_verified']==18
                clock=next(r for r in timing['checkpoints'] if r['collection_steps']==step)
                data.append(dict(collection_steps=step,real_simulator_steps=step*4096,
                    successes=successes,failures=18-successes,mean_normalized_full_motion_return=mean,
                    original_training_checkpoint_clock=clock))
            validation.update(status='all_six_interrupted_saved_snapshots_full18_evaluation_originals_verified',
                points=data,all108_native_clip_records_recomputed=True,all_cloud_arrays_and_initial_states_audited=True,
                parent_training_complete=False,training_reexecuted=False,full_training_resume_available=False)
    validation.update(job_uid=plan['uid'],source_commit=plan['source_commit'],log_sha256=sha(raw),
        raw_full_models_and_trajectories_downloaded=False)
    for name,value in [('validation.json',validation),('receipt-ledger.json',dict(receipts=ledger))]:
        (args.output/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8',newline='\n')
    if args.require_complete:assert validation['status']=='all_six_interrupted_saved_snapshots_full18_evaluation_originals_verified'
    print(json.dumps(validation,indent=2))


if __name__=='__main__':
    main()
