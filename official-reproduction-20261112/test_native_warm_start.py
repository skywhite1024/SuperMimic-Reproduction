"""Verify real tensor restoration, declared resets and corrupted-source rejection."""
import json
import types

import pytest
import torch

from checkpoint_series import preserve_checkpoint
from native_warm_start import read_snapshot, restore_snapshot, learned_parameter_hashes


def fixture_snapshot(tmp_path):
    cfg={key:None for key in ('actor_obs','history_length','dynamics_hidden_dims',
        'actor_hidden_dims','critic_hidden_dims','latent_dim','wm','policy','replay_ratio','buffer_size')}
    (tmp_path/'resolved-config.json').write_text(json.dumps(dict(
        corpus=dict(name='bones-raw'),training_profile='paper-a5-single-gpu',runner=cfg)))
    modules=[torch.nn.Linear(3,2),torch.nn.Linear(2,2),torch.nn.Linear(3,2),torch.nn.Linear(3,1)]
    modules[2].register_buffer('observations',torch.tensor(172032))
    live=tmp_path/'live';live.mkdir()
    for name,value in [('world_model.pt',dict(dynamics=modules[0].state_dict(),sim_context=modules[1].state_dict())),
                       ('actor.pt',modules[2].state_dict()),('critic.pt',modules[3].state_dict())]:
        torch.save(value,live/name)
    preserve_checkpoint([live/name for name in ('world_model.pt','actor.pt','critic.pt')],
                        tmp_path/'checkpoint-series',11)
    path=tmp_path/'checkpoint-series/step-000000011/manifest.json'
    return path,cfg,modules


def test_restores_actual_weights_and_normalizer_buffers_with_reset_declaration(tmp_path):
    path,cfg,source=fixture_snapshot(tmp_path)
    payloads,evidence=read_snapshot(path,11,cfg,'bones-raw')
    target=[torch.nn.Linear(3,2),torch.nn.Linear(2,2),torch.nn.Linear(3,2),torch.nn.Linear(3,1)]
    target[2].register_buffer('observations',torch.tensor(0))
    class Bundle:
        def load_bundle(self,values):
            target[0].load_state_dict(values['dynamics'])
            target[1].load_state_dict(values['sim_context'])
    runner=types.SimpleNamespace(_wm=Bundle(),_dynamics=target[0],_sim_context=target[1],
        policy_learner=types.SimpleNamespace(actor=target[2],critic=target[3]))
    result=restore_snapshot(runner,payloads)
    assert result['all_four_native_state_dictionaries_equal_source']
    assert target[2].observations.item()==172032
    assert torch.equal(target[2].weight,source[2].weight)
    assert evidence['exact_training_continuation'] is False
    assert evidence['optimizer_state_restored'] is False
    assert evidence['replay_buffer_restored'] is False
    before=learned_parameter_hashes(runner)
    target[2].observations.add_(1)
    assert learned_parameter_hashes(runner)==before  # A count change is not learning.
    optimizer=torch.optim.SGD(target[2].parameters(),lr=.1)
    target[2](torch.ones(4,3)).square().mean().backward();optimizer.step()
    after=learned_parameter_hashes(runner)
    assert after['actor']!=before['actor']
    assert after['world_model']==before['world_model'] and after['critic']==before['critic']


def test_rejects_wrong_step_configuration_and_corrupted_model(tmp_path):
    path,cfg,_=fixture_snapshot(tmp_path)
    with pytest.raises(ValueError,match='step/status'):
        read_snapshot(path,10,cfg,'bones-raw')
    with pytest.raises(ValueError,match='configuration mismatch'):
        read_snapshot(path,11,{**cfg,'latent_dim':31},'bones-raw')
    model=path.parent/'actor.pt'
    data=model.read_bytes();model.write_bytes(bytes([data[0]^1])+data[1:])
    with pytest.raises(ValueError,match='bytes/hash'):
        read_snapshot(path,11,cfg,'bones-raw')
