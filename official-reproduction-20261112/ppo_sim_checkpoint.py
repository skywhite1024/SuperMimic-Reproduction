"""Bind real-simulator PPO models to their own completed training receipts."""
import hashlib
import json
from pathlib import Path
import types


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def select_ppo_sim_checkpoint(run, update=None):
    run = Path(run).resolve()
    result_path = run / 'result.json'
    config_path = run / 'resolved-config.json'
    manifest_path = run / 'checkpoint-manifest.json'
    training = json.loads(result_path.read_bytes())
    config = json.loads(config_path.read_bytes())
    manifest = json.loads(manifest_path.read_bytes())
    if training['status'] not in ('real_G1_simulator_PPO_integration_completed',
                                  'real_G1_simulator_PPO_finite_training_completed'):
        raise ValueError('Completed real-simulator PPO training required')
    if (config['training_algorithm'] != 'PPO-Sim'
            or not config['no_world_model_constructed']
            or config['world_model_updates'] != 0 or config['imagined_transitions'] != 0
            or training['world_model_updates'] != 0 or training['imagined_transitions'] != 0):
        raise ValueError('Expected explicitly simulator-only PPO receipts')
    if (digest(config_path) != training['source_config_sha256']
            or digest(manifest_path) != training['checkpoint_manifest_sha256']
            or manifest != training['checkpoint_sets']):
        raise ValueError('PPO parent/config/manifest binding mismatch')
    total = training['ppo_update_calls']
    if (total != config['outer_updates'] or len(training['updates']) != total
            or training['collection_steps'] != total * config['rollout_steps']
            or training['simulator_steps'] != training['collection_steps'] * config['num_envs']):
        raise ValueError('Inconsistent real PPO update/transition counters')
    if update is None:
        update = total
    rows = [row for row in manifest if row['ppo_update'] == update]
    if len(rows) != 1 or not 1 <= update <= total:
        raise ValueError('Exactly one retained PPO update required')
    row = rows[0]
    if row['collection_steps'] != update * config['rollout_steps']:
        raise ValueError('PPO snapshot collection count mismatch')
    if len(row['files']) != 3 or {r['file'] for r in row['files']} != {'actor.pt', 'critic.pt', 'ppo.pt'}:
        raise ValueError('PPO must have actor, critic and optimizer models; no world model')
    checkpoints = {}
    for record in row['files']:
        path = Path(record['path']).resolve()
        if (not path.is_relative_to(run / 'checkpoints') or path.name != record['file']
                or path.stat().st_size != record['bytes'] or digest(path) != record['sha256']
                or not record['model_state_readback_verified']):
            raise ValueError('PPO model path, bytes, hash or readback mismatch')
        checkpoints[path.name] = path
    selection = dict(kind='PPO-Sim_native_actor_snapshot', training_algorithm='PPO-Sim',
        parent_result_sha256=digest(result_path), resolved_config_sha256=digest(config_path),
        manifest_sha256=digest(manifest_path), parent_ppo_updates=total,
        evaluated_ppo_update=update, parent_collection_steps=training['collection_steps'],
        evaluated_collection_steps=row['collection_steps'], files=row['files'],
        no_world_model_loaded=True, deterministic_native_actor_mean=True)
    return training, config, checkpoints, selection


def build_ppo_sim_eval_actor(runner_cfg, env, actor_path, device):
    """Build only native representation/context/actor, without learned dynamics."""
    import torch
    from supermimic.representations import create_representation
    from supermimic.runners.supermimic_runner import build_actor
    from supermimic.sim_context import SimContext
    from supermimic.utils.helpers import unpack_obs_dims
    observations, _ = env.reset()
    full_width, goal_width = unpack_obs_dims(observations)
    context = SimContext.from_env(env.unwrapped)
    rep = create_representation(runner_cfg.representation, full_width, env.num_actions,
        env.unwrapped.step_dt, env.unwrapped.cfg.decimation,
        env.unwrapped.scene['robot'].data.default_joint_pos[0].to(device),
        actor_obs_mode=runner_cfg.actor_obs).to(device)
    rep.bind(env.unwrapped, context)
    shape = types.SimpleNamespace(representation=rep, num_actions=env.num_actions,
                                  num_goals=goal_width)
    actor = build_actor(runner_cfg, shape, device)
    actor.load_state_dict(torch.load(actor_path, map_location=device, weights_only=True))
    return actor, rep, context
