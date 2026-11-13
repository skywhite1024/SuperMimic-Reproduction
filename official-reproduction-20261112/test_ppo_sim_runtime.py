"""Exercise simulator-only transport, timeout bootstrapping and native PPO."""
import types

import numpy as np
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage
from supermimic.models import build_mlp
from supermimic.runners.reset_sampling import ReturnSampler, ReturnSamplerCfg
from supermimic.runners.supermimic_runner import SuperMimicRunnerCfg, build_actor

from paper_profile import configure_runner
from paper_ppo_models import NativeMotionPpoActor, NativePpoCritic
from ppo_sim_runtime import SimulatorPolicyInput, SimulatorPpoLoop


class FakeRepresentation:
    rep_state_dim=64
    def proprio_obs_dim(self,n):return 93
    def add_proprio_noise(self,sensor):return sensor.clone()
    def represent(self,full):return full[:,:64]
    def extract_aux_state(self,full):return full[:,:7]
    def compute_actor_sensor(self,represented,context,*,aux_state,sim_actor_obs,noisy):return sim_actor_obs
    def build_actor_obs(self,history,goal):return torch.cat([history.flatten(1),goal],dim=-1)


class SyntheticSimulator:
    def __init__(self):
        self.num_envs=8;self.unwrapped=self;self.t=0;self.weights=[]
        self.motion=types.SimpleNamespace(time_steps=torch.arange(8))
        self.motion.set_sampling_distribution=lambda weights:self.weights.append(weights.clone())
        self.command_manager=types.SimpleNamespace(get_term=lambda name:self.motion)
        self.termination_manager=types.SimpleNamespace(terminated=torch.zeros(8,dtype=torch.bool),
            time_outs=torch.zeros(8,dtype=torch.bool),get_term=lambda name:self.clip_end)
        self.clip_end=torch.zeros(8,dtype=torch.bool)
    def get_observations(self):
        return dict(actor=torch.full((8,93),self.t/10),
            full_state=torch.arange(71).float().repeat(8,1)/100+self.t/100,
            goal=torch.arange(670).float().repeat(8,1)/1000)
    def step(self,actions):
        assert actions.shape==(8,29)
        self.t+=1;self.motion.time_steps+=1
        self.termination_manager.terminated[:]=False
        self.termination_manager.terminated[0]=True
        self.termination_manager.time_outs[:]=False
        self.termination_manager.time_outs[1:3]=True
        self.clip_end[:]=False;self.clip_end[2]=True
        dones=(self.termination_manager.terminated|self.termination_manager.time_outs).long()
        return self.get_observations(),torch.arange(1,9).float()/100,dones,


def test_simulator_only_native_update_history_timeout_and_return_sampler(tmp_path):
    torch.manual_seed(42);torch.set_num_threads(2)
    env=SyntheticSimulator()
    step=env.step
    def complete_step(actions):
        following,rewards,dones=step(actions)
        return following,rewards,dones,{'time_outs':env.termination_manager.time_outs.clone()}
    env.step=complete_step
    representation=FakeRepresentation()
    context=types.SimpleNamespace(read=lambda _:torch.zeros(8,37))
    inputs=SimulatorPolicyInput(representation,context,10,'cpu')
    native_cfg=SuperMimicRunnerCfg();configure_runner(native_cfg)
    wm_shape=types.SimpleNamespace(representation=representation,num_actions=29,num_goals=670)
    actor=NativeMotionPpoActor(build_actor(native_cfg,wm_shape,'cpu'),action_dim=29)
    critic=NativePpoCritic(build_mlp(771,(256,128),1),input_dim=771)
    inputs.seed(env.get_observations());template=inputs.build(env.get_observations(),env)
    storage=RolloutStorage('rl',8,4,template,[29],device='cpu')
    ppo=PPO(actor,critic,storage,num_learning_epochs=1,num_mini_batches=2,device='cpu')
    sampler=ReturnSampler(ReturnSamplerCfg(horizon=2),4,1000,8,.1,device='cpu')
    loop=SimulatorPpoLoop(env,ppo,inputs,4,sampler=sampler)
    result=loop.update(tmp_path/'trace.npz')
    assert result['imagined_transitions']==result['world_model_updates']==0
    assert result['collection_steps']==4 and result['simulator_steps']==32
    assert result['native_actor_and_critic_changed_and_finite']
    assert result['binary_done_rows']==12 and result['timeout_rows']==8
    assert result['native_actor_normalization_count']==32
    assert len(env.weights)==result['sampler_distribution_updates']==2
    assert all(torch.isfinite(weights).all() and torch.isclose(weights.sum(),torch.tensor(1.)) for weights in env.weights)
    torch.testing.assert_close(inputs.history[0],torch.full((10,93),.4),rtol=0,atol=0)
    torch.testing.assert_close(inputs.history[3,:,0],torch.tensor([0.,0.,0.,0.,0.,0.,.1,.2,.3,.4]),rtol=0,atol=0)
    with np.load(tmp_path/'trace.npz',allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['raw_rewards'],np.tile(np.arange(1,9,dtype=np.float32)/100,(4,1)))
        np.testing.assert_array_equal(saved['dones'],np.tile([1,1,1,0,0,0,0,0],(4,1)))
        np.testing.assert_array_equal(saved['timeouts'],np.tile([False,True,True,False,False,False,False,False],(4,1)))
        np.testing.assert_array_equal(saved['stored_rewards'][:,[0,3,4,5,6,7]],saved['raw_rewards'][:,[0,3,4,5,6,7]])
        assert np.any(saved['stored_rewards'][:,1:3]!=saved['raw_rewards'][:,1:3])
