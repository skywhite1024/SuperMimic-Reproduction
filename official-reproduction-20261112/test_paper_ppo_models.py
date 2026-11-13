"""Check exact native functions, gradients and real RSL PPO compatibility."""

import copy

import torch
from tensordict import TensorDict
from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage

from supermimic.models import MultiEncoderAE, build_mlp
from supermimic.tasks.task_spec import TaskSpec, build_actor_obs_td
from paper_ppo_models import NativeMotionPpoActor, NativePpoCritic


def make_actor():
    return MultiEncoderAE(obs_dim=93, action_dim=29,
        task_specs=[TaskSpec('motion', 18, lambda *args: None),
                    TaskSpec('velocity', 3, lambda *args: None)],
        slot_mix={'motion':1.,'velocity':0.}, latent_dim=32,
        hidden_dims=(2048,1024,512,256), latent_bc_weight=.1, history_length=10)


def observations(n):
    history, goal = torch.randn(n,10,93), torch.randn(n,18)
    flat = torch.cat([history.flatten(1),goal],dim=-1)
    return build_actor_obs_td({'motion':goal},history), flat


def test_native_actor_forward_normalization_and_gradients_are_preserved():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    native = make_actor()
    rng = torch.get_rng_state().clone()
    wrapper = NativeMotionPpoActor(copy.deepcopy(native),action_dim=29)
    assert torch.equal(rng,torch.get_rng_state())
    td, flat = observations(4)
    wrapped_td = TensorDict({'actor':flat},batch_size=[4])
    native.train(); wrapper.train()
    native.update_normalization(td); wrapper.update_normalization(wrapped_td)
    for name, value in native.state_dict().items():
        torch.testing.assert_close(value,wrapper.native_actor.state_dict()[name],rtol=0,atol=0)
    native.eval(); wrapper.eval()
    direct, wrapped = native(td), wrapper(wrapped_td)
    torch.testing.assert_close(direct,wrapped,rtol=0,atol=0)
    direct.square().sum().backward(); wrapped.square().sum().backward()
    for (name,a),(other,b) in zip(native.named_parameters(),wrapper.native_actor.named_parameters(),strict=True):
        assert name == other
        if a.grad is None:
            assert b.grad is None
        else:
            torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)


def test_real_rsl_ppo_updates_exact_native_models_and_gaussian_distribution():
    torch.manual_seed(42)
    torch.set_num_threads(2)
    n, horizon = 8, 8
    actor = NativeMotionPpoActor(make_actor(),action_dim=29)
    critic_native = build_mlp(119,(256,128),1)
    rng = torch.get_rng_state().clone()
    critic = NativePpoCritic(critic_native,input_dim=119)
    assert torch.equal(rng,torch.get_rng_state())
    def obs():
        _,flat = observations(n)
        return TensorDict({'actor':flat,'critic':torch.randn(n,119)},batch_size=[n])
    initial = obs()
    torch.testing.assert_close(critic(initial),critic_native(initial['critic']),rtol=0,atol=0)
    storage = RolloutStorage('rl',n,horizon,initial,[29],device='cpu')
    storage.dones = torch.zeros_like(storage.dones,dtype=torch.float32)
    ppo = PPO(actor,critic,storage,num_learning_epochs=1,num_mini_batches=2,device='cpu')
    before = actor.native_actor.decoder[-1].weight.detach().clone()
    ppo.eval_mode()
    with torch.no_grad():
        current = initial
        for _ in range(horizon):
            actions = ppo.act(current)
            following = obs()
            ppo.process_env_step(following,1-actions.square().mean(-1),torch.full((n,),.25),{})
            current = following
        ppo.compute_returns(current)
    assert torch.equal(storage.dones,torch.full_like(storage.dones,.25))
    losses = ppo.update()
    assert losses and all(torch.isfinite(torch.tensor(value)) for value in losses.values())
    assert not torch.equal(before,actor.native_actor.decoder[-1].weight)
    assert all(torch.isfinite(parameter).all() for parameter in [*actor.parameters(),*critic.parameters()])
