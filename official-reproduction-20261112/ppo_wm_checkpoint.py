"""Select native inference weights bound to completed PPO-WM learning receipts."""
import hashlib
import json
from pathlib import Path


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def select_ppo_wm_checkpoint(run, update=None):
    run=Path(run).resolve()
    training=json.loads((run/'result.json').read_bytes())
    config=json.loads((run/'resolved-config.json').read_bytes())
    manifest=json.loads((run/'native-checkpoint-manifest.json').read_bytes())
    if training['status']!='real_G1_native_network_PPO_WM_finite_training_completed':
        raise ValueError('Completed finite PPO-WM training is required')
    if config['training_algorithm']!='PPO-WM' or config['training_purpose']!='finite-training':
        raise ValueError('PPO-WM learning configuration required')
    assert training['source_config_sha256']==digest(run/'resolved-config.json')
    assert training['native_checkpoint_manifest_sha256']==digest(run/'native-checkpoint-manifest.json')
    assert manifest==training['retained_native_checkpoints']
    total=config['outer_iterations']
    assert total==training['outer_iterations']==training['real_native_ppo_updates']==len(training['updates'])
    assert training['real_simulator_steps']==total*10*4096
    assert training['imagined_transitions']==total*config['ppo_runner']['batch_size']*config['ppo_runner']['num_steps_per_env']
    update=total if update is None else update
    rows=[row for row in manifest if row['ppo_update']==update]
    if len(rows)!=1:raise ValueError('Exactly one retained PPO-WM checkpoint required')
    row=rows[0]
    assert row['real_collection_steps']==update*10 and row['real_simulator_steps']==update*10*4096
    assert row['imagined_steps']==update*config['ppo_runner']['batch_size']*config['ppo_runner']['num_steps_per_env']
    required={'actor.pt','critic.pt','world_model.pt','released-actor-critic.pt'}
    assert {r['file'] for r in row['files']}==required and len(row['files'])==4
    checkpoints={}
    for record in row['files']:
        path=Path(record['path']).resolve()
        if not path.is_relative_to(run/'native-checkpoints') or path.name!=record['file']:
            raise ValueError('PPO-WM checkpoint path escaped its native snapshot directory')
        if path.stat().st_size!=record['bytes'] or digest(path)!=record['sha256'] or not record['native_model_readback_verified']:
            raise ValueError('PPO-WM native model readback/bytes/hash mismatch')
        checkpoints[path.name]=path
    selection=dict(kind='PPO-WM_native_inference_snapshot',training_algorithm='PPO-WM',
        parent_result_sha256=digest(run/'result.json'),resolved_config_sha256=digest(run/'resolved-config.json'),
        manifest_sha256=digest(run/'native-checkpoint-manifest.json'),parent_ppo_updates=total,
        evaluated_ppo_update=update,parent_collection_steps=total*10,evaluated_collection_steps=update*10,
        cumulative_imagined_steps=row['imagined_steps'],files=row['files'],
        deterministic_native_actor_mean=True,full_training_state_resume_available=False)
    return training,config,checkpoints,selection
