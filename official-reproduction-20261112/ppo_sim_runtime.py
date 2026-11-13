"""PPO updates from real simulator transitions with native observation history.

No learned-world-model transition or replay-buffer update is used here.
The caller supplies native actor/critic models and explicit PPO parameters.
"""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from tensordict import TensorDict

from supermimic.representations import extract_actor_goal
from ppo_wm_native_runtime import state_sha256


class SimulatorPolicyInput:
    def __init__(self, representation, sim_context, history_length, device):
        self.representation=representation
        self.sim_context=sim_context
        self.history_length=history_length
        self.device=device
        self.history=None

    def seed(self, observations):
        sensor=self.representation.add_proprio_noise(observations['actor'].to(self.device))
        self.history=sensor.unsqueeze(1).repeat(1,self.history_length,1)

    def advance(self, observations, dones):
        sensor=self.representation.add_proprio_noise(observations['actor'].to(self.device))
        shifted=torch.cat([self.history[:,1:],sensor.unsqueeze(1)],dim=1)
        self.history=torch.where(dones.bool().reshape(-1,1,1),
            sensor.unsqueeze(1).repeat(1,self.history_length,1),shifted)

    def build(self, observations, env):
        full=observations['full_state'].to(self.device)
        represented=self.representation.represent(full)
        context=self.sim_context.read(env.unwrapped)
        # Preserve the native collection-side sensor call, including its RNG
        # consumption; the history ring already contains the current sample.
        sensor=self.representation.compute_actor_sensor(represented,context,
            aux_state=self.representation.extract_aux_state(full),
            sim_actor_obs=observations['actor'].to(self.device),noisy=True)
        assert sensor.shape[-1]==self.history.shape[-1]
        goal=observations['goal'].to(self.device)
        actor=self.representation.build_actor_obs(self.history,extract_actor_goal(goal))
        critic=torch.cat([represented,goal,context],dim=-1)
        return TensorDict({'actor':actor,'critic':critic},batch_size=[env.num_envs],device=self.device)


class SimulatorPpoLoop:
    def __init__(self, env, algorithm, policy_input, rollout_steps, logger=None, sampler=None,
                 require_parameter_change=True):
        self.env=env
        self.algorithm=algorithm
        self.policy_input=policy_input
        self.rollout_steps=rollout_steps
        self.logger=logger
        self.sampler=sampler
        self.require_parameter_change=require_parameter_change
        self.motion_cmd=env.unwrapped.command_manager.get_term('motion')
        self.device=algorithm.device
        self.updates=[]
        self.collections=0
        self.simulator_steps=0
        self.sampler_updates=0
        with torch.no_grad():
            obs=env.get_observations()
            self.policy_input.seed(obs)
            self.observations=self.policy_input.build(obs,env)

    def update(self, trace_path=None):
        alg=self.algorithm
        alg.train_mode()
        alg.storage.clear()
        actor_before=state_sha256(alg.actor.native_actor)
        critic_before=state_sha256(alg.critic.native_critic)
        actor_parameters_before=[parameter.detach().clone() for parameter in alg.actor.native_actor.parameters()]
        rewards_all=[];expected_rewards=[];dones_all=[];timeouts_all=[]
        started=time.monotonic()
        with torch.inference_mode():
            for _ in range(self.rollout_steps):
                frames=self.motion_cmd.time_steps.to(self.device).clone()
                actions=alg.act(self.observations)
                values=alg.transition.values.detach().clone().flatten()
                following,rewards,dones,extras=self.env.step(actions)
                rewards=rewards.to(self.device);dones=dones.to(self.device)
                if not torch.isfinite(rewards).all() or not ((dones==0)|(dones==1)).all():
                    raise ValueError('Expected finite simulator rewards and binary termination')
                self.policy_input.advance(following,dones)
                self.observations=self.policy_input.build(following,self.env)
                if not all(torch.isfinite(value).all() for value in self.observations.values()):
                    raise ValueError('Nonfinite simulator policy observations')
                timeout=extras.get('time_outs',torch.zeros_like(dones,dtype=torch.bool)).to(self.device)
                assert timeout.shape==dones.shape
                rewards_all.append(rewards.clone())
                dones_all.append(dones.clone());timeouts_all.append(timeout.clone())
                expected_rewards.append(rewards+alg.gamma*values*timeout)
                alg.process_env_step(self.observations,rewards,dones,extras)
                manager=self.env.unwrapped.termination_manager
                terminated=manager.terminated.to(self.device,torch.bool)
                if self.logger is not None:
                    self.logger.process_env_step(rewards,terminated,extras)
                if self.sampler is not None:
                    truncated=manager.time_outs.to(self.device,torch.bool)
                    clip_end=manager.get_term('motion_end').to(self.device,torch.bool)
                    weights=self.sampler.record(frames,rewards,terminated&~truncated,
                        clip_end,truncated&~clip_end)
                    if weights is not None:
                        self.motion_cmd.set_sampling_distribution(weights)
                        self.sampler_updates+=1
                self.collections+=1
                self.simulator_steps+=self.env.num_envs
            alg.compute_returns(self.observations)
        collect_seconds=time.monotonic()-started
        expected=torch.stack(expected_rewards)
        stored=alg.storage.rewards[...,0]
        assert torch.equal(stored,expected)
        done_tensor=torch.stack(dones_all)
        assert torch.equal(alg.storage.dones[...,0],done_tensor.to(alg.storage.dones.dtype))
        row=dict(update=len(self.updates)+1,collection_steps=self.collections,
            simulator_steps=self.simulator_steps,rollout_steps=self.rollout_steps,
            num_envs=self.env.num_envs,source='MuJoCo env.step only',
            binary_done_rows=int(done_tensor.sum()),timeout_rows=int(torch.stack(timeouts_all).sum()),
            reward_sum=float(torch.stack(rewards_all).sum()),
            stored_rewards_match_native_RSL_timeout_bootstrap=True,
            stored_dones_match_simulator=True,collect_seconds=collect_seconds,
            sampler_distribution_updates=self.sampler_updates)
        if trace_path is not None:
            trace_path=Path(trace_path);trace_path.parent.mkdir(parents=True,exist_ok=True)
            arrays=dict(raw_rewards=torch.stack(rewards_all).cpu().numpy(),
                stored_rewards=stored.cpu().numpy(),dones=done_tensor.cpu().numpy(),
                timeouts=torch.stack(timeouts_all).cpu().numpy())
            np.savez_compressed(trace_path,**arrays)
            with np.load(trace_path,allow_pickle=False) as saved:
                assert set(saved.files)==set(arrays)
                assert all(np.array_equal(saved[key],value) for key,value in arrays.items())
            row['trace']=dict(path=str(trace_path),bytes=trace_path.stat().st_size,
                sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest(),readback_verified=True)
        started=time.monotonic()
        losses=alg.update()
        row['learn_seconds']=time.monotonic()-started
        row['losses']={key:float(value) for key,value in losses.items()}
        assert losses and all(np.isfinite(value) for value in row['losses'].values())
        for model in (alg.actor,alg.critic):
            assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
        actor_after=state_sha256(alg.actor.native_actor)
        critic_after=state_sha256(alg.critic.native_critic)
        actor_parameters_changed=any(not torch.equal(before,after) for before,after in
            zip(actor_parameters_before,alg.actor.native_actor.parameters(),strict=True))
        critic_changed=critic_before!=critic_after
        if self.require_parameter_change:
            assert actor_before!=actor_after and actor_parameters_changed and critic_changed
        row.update(actor_before=actor_before,actor_after=actor_after,
            critic_before=critic_before,critic_after=critic_after,
            native_actor_parameters_changed=actor_parameters_changed,
            native_critic_state_changed=critic_changed,
            native_actor_and_critic_changed_and_finite=actor_parameters_changed and critic_changed,
            native_model_parameters_finite=True,
            imagined_transitions=0,world_model_updates=0,
            native_actor_normalization_count=int(alg.actor.native_actor.obs_normalizer.count))
        self.updates.append(row)
        print('PPO_SIM_ACTUAL_UPDATE '+json.dumps(row),flush=True)
        return row
