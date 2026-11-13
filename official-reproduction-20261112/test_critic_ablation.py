"""Exercise real native learner updates and strict two-file retained models."""
import dataclasses
import json
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from checkpoint_series import preserve_checkpoint
from paper_profile import configure_runner, configure_critic_ablation, install_paper_runner, build_adam_optimizers
from select_completed_checkpoint import digest, select_completed_checkpoint


class Actor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
    def forward(self, obs):
        return obs['actor'] * self.weight


class ImaginedEnv:
    max_step_reward = 1.0
    def __init__(self):
        self.dynamics = torch.nn.Linear(1, 1)
        self.representation = torch.nn.Linear(1, 1)
        self.representation.dt = 1.0
    def sample_initial_states(self):
        return TensorDict({}, batch_size=[4])
    def reset(self, batch):
        return self.obs()
    def obs(self):
        return TensorDict({'actor':torch.ones(4,1), 'critic':torch.ones(4,1)},batch_size=[4])
    def step(self, actions):
        return self.obs(), 1.0 + actions.squeeze(-1), torch.zeros(4), {}


@pytest.mark.parametrize('without_critic',[False,True])
def test_native_update_with_a5_optimizer_and_matched_gamma(monkeypatch, without_critic):
    import supermimic.learners.policy_learner as policy
    import supermimic.learners.wm_learner as wm
    from supermimic.runners.supermimic_runner import SuperMimicRunnerCfg
    from supermimic.runners.run_config import load_runner_cfg
    # Track the process-local imported bindings so this check restores them.
    monkeypatch.setattr(policy,'build_optimizers',policy.build_optimizers)
    monkeypatch.setattr(wm,'build_optimizers',wm.build_optimizers)
    cfg = SuperMimicRunnerCfg()
    configure_runner(cfg)
    shared = dataclasses.asdict(cfg.policy)
    description = configure_critic_ablation(cfg,8,without_critic)
    assert cfg.policy.gamma == description['gamma'] == 0.99
    for key,value in dataclasses.asdict(cfg.policy).items():
        assert value == (8 if key=='horizon' else shared[key])
    rebuilt = load_runner_cfg({'runner_cfg':dataclasses.asdict(cfg)})
    assert type(rebuilt.policy) is type(cfg.policy)
    class Runner:
        def __init__(self):
            self.cfg = cfg
            env = ImaginedEnv()
            self.world_model_learner = SimpleNamespace(optimizers=build_adam_optimizers([env.dynamics],2e-4))
            actor=Actor()
            if without_critic:
                self.policy_learner=policy.BpttPolicyLearner(actor,env,cfg.policy)
            else:
                self.policy_learner=policy.ActorCriticPolicyLearner(actor,torch.nn.Linear(1,1),env,cfg.policy)
        def step(self):
            return self.policy_learner.update()
    runner = install_paper_runner(Runner)()
    old = runner.policy_learner.actor.weight.detach().clone()
    stats=runner.step()
    assert torch.isfinite(stats['Policy/reward'])
    assert runner.policy_learner.actor.weight > old
    assert len(runner._paper_optimizers) == (2 if without_critic else 3)
    assert ('Policy/value_function' in stats) is (not without_critic)
    if without_critic:
        assert not hasattr(runner.policy_learner,'critic')
        wanted=sum(0.99**i for i in range(8))/8
        assert stats['Policy/reward'].item() == pytest.approx(wanted)


def test_two_model_selection_requires_explicit_no_critic_parent(tmp_path):
    live=tmp_path/'live'
    live.mkdir()
    paths=[live/name for name in ['world_model.pt','actor.pt']]
    for path in paths:
        torch.save({'weight':torch.tensor([11.0])},path)
    receipt=preserve_checkpoint(paths,tmp_path/'checkpoint-series',11,without_critic=True)
    for path in paths:
        torch.save({'weight':torch.tensor([21.0])},path)
    result=dict(status='pilot_completed',collection_steps=21,critic_enabled=False,
        retained_checkpoint_series=[receipt],checkpoints=[dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p)) for p in paths])
    dest=tmp_path/'result.json'
    dest.write_text(json.dumps(result))
    _,selected,_=select_completed_checkpoint(tmp_path,11)
    assert set(selected)=={'world_model.pt','actor.pt'}
    assert torch.load(selected['actor.pt'],weights_only=True)['weight'].item()==11
    del result['critic_enabled']
    dest.write_text(json.dumps(result))
    with pytest.raises(ValueError,match='declared native'):
        select_completed_checkpoint(tmp_path,11)
    with pytest.raises(ValueError,match='complete declared'):
        preserve_checkpoint(paths,tmp_path/'other',21)
