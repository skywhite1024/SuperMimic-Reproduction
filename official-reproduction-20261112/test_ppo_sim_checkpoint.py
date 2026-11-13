"""PPO receipts select actual native models and reject stale/corrupt evidence."""
import json
import types

import pytest
import torch

from ppo_sim_checkpoint import digest, select_ppo_sim_checkpoint, build_ppo_sim_eval_actor


def completed_run(root):
    config = dict(training_algorithm='PPO-Sim', no_world_model_constructed=True,
                  world_model_updates=0, imagined_transitions=0, outer_updates=3,
                  rollout_steps=24, num_envs=8192)
    (root / 'resolved-config.json').write_text(json.dumps(config))
    manifest = []
    for update in (1, 3):
        folder = root / 'checkpoints' / f'update-{update:03d}'
        folder.mkdir(parents=True)
        files = []
        for name in ('actor.pt', 'critic.pt', 'ppo.pt'):
            path = folder / name
            torch.save({'weight': torch.tensor([float(update)])}, path)
            files.append(dict(file=name, path=str(path), bytes=path.stat().st_size,
                              sha256=digest(path), model_state_readback_verified=True))
        manifest.append(dict(ppo_update=update, collection_steps=update*24, files=files))
    (root / 'checkpoint-manifest.json').write_text(json.dumps(manifest))
    training = dict(status='real_G1_simulator_PPO_integration_completed', ppo_update_calls=3,
                    collection_steps=72, simulator_steps=589824, updates=[{}, {}, {}],
                    checkpoint_sets=manifest, imagined_transitions=0, world_model_updates=0,
                    source_config_sha256=digest(root / 'resolved-config.json'),
                    checkpoint_manifest_sha256=digest(root / 'checkpoint-manifest.json'))
    (root / 'result.json').write_text(json.dumps(training))
    return training


def test_ppo_snapshot_native_files_and_parent_binding(tmp_path):
    completed_run(tmp_path)
    _, _, weights, selection = select_ppo_sim_checkpoint(tmp_path, 1)
    assert torch.load(weights['actor.pt'], weights_only=True)['weight'].item() == 1
    assert selection['evaluated_ppo_update'] == 1
    assert selection['evaluated_collection_steps'] == 24
    assert selection['parent_collection_steps'] == 72
    assert selection['no_world_model_loaded']
    _, _, final, _ = select_ppo_sim_checkpoint(tmp_path)
    assert torch.load(final['actor.pt'], weights_only=True)['weight'].item() == 3
    with pytest.raises(ValueError, match='retained PPO update'):
        select_ppo_sim_checkpoint(tmp_path, 2)


def test_reject_running_first_order_stale_config_and_corrupt_ppo(tmp_path):
    training = completed_run(tmp_path)
    result = tmp_path / 'result.json'
    for status in ('training', 'pilot_completed'):
        result.write_text(json.dumps(dict(training, status=status)))
        with pytest.raises(ValueError, match='Completed real-simulator PPO'):
            select_ppo_sim_checkpoint(tmp_path)
    result.write_text(json.dumps(training))
    config = tmp_path / 'resolved-config.json'
    original = config.read_bytes()
    config.write_bytes(original + b' ')
    with pytest.raises(ValueError, match='binding mismatch'):
        select_ppo_sim_checkpoint(tmp_path)
    config.write_bytes(original)
    actor = tmp_path / 'checkpoints/update-003/actor.pt'
    actor.write_bytes(b'corrupt weights')
    with pytest.raises(ValueError, match='bytes, hash'):
        select_ppo_sim_checkpoint(tmp_path)


def test_native_actor_weights_normalizers_and_mean_restore_without_dynamics(tmp_path, monkeypatch):
    from supermimic.runners.supermimic_runner import SuperMimicRunnerCfg, build_actor
    from supermimic.tasks.task_spec import build_actor_obs_td
    import supermimic.runners.supermimic_runner as runner_module
    import supermimic.representations as representations
    from supermimic.sim_context import SimContext
    from paper_profile import configure_runner
    torch.manual_seed(42)
    torch.set_num_threads(2)
    cfg = SuperMimicRunnerCfg(); configure_runner(cfg)
    rep = types.SimpleNamespace(rep_state_dim=64, proprio_obs_dim=lambda n:93)
    rep.to = lambda device: rep
    rep.bind = lambda env, context: None
    original = build_actor(cfg, types.SimpleNamespace(representation=rep, num_actions=29,
                                                       num_goals=670), 'cpu')
    observation = build_actor_obs_td({'motion':torch.randn(2,640)},torch.randn(2,10,93))
    original.update_normalization(observation)
    original.eval()
    expected = original(observation).detach()
    path = tmp_path / 'actor.pt'; torch.save(original.state_dict(),path)
    base = types.SimpleNamespace(step_dt=.02, cfg=types.SimpleNamespace(decimation=4),
        scene={'robot':types.SimpleNamespace(data=types.SimpleNamespace(default_joint_pos=torch.zeros(2,29)))})
    obs = {'full_state':torch.zeros(2,71), 'goal':torch.zeros(2,670)}
    env = types.SimpleNamespace(reset=lambda:(obs,{}),unwrapped=base,num_actions=29)
    monkeypatch.setattr(SimContext,'from_env',classmethod(lambda cls,base:types.SimpleNamespace()))
    monkeypatch.setattr(representations,'create_representation',lambda *args,**kwargs:rep)
    def forbidden(*args,**kwargs):raise AssertionError('PPO inference must not construct dynamics')
    monkeypatch.setattr(runner_module,'build_world_model',forbidden)
    restored, _, _ = build_ppo_sim_eval_actor(cfg,env,path,'cpu')
    restored.eval()
    torch.testing.assert_close(restored(observation),expected,rtol=0,atol=0)
    assert restored.obs_normalizer.count.item()==original.obs_normalizer.count.item()>0
    assert set(restored.state_dict())==set(original.state_dict())
