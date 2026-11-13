"""Check the new instrumentation against actual native models and RSL updates."""
import types

import numpy as np
import torch
from tensordict import TensorDict

from supermimic.runners.ppo_runner import PpoRunnerCfg
from supermimic.runners.supermimic_runner import SuperMimicRunnerCfg
from supermimic.representations import actor_goal_dim, NUM_GOAL_FRAMES
from paper_profile import configure_runner
from ppo_wm_native_runtime import make_native_ppo_runner, state_sha256


def test_native_runtime_records_real_ppo_and_exact_storage_conversion(tmp_path):
    torch.set_num_threads(2)
    initial_hashes=[]
    num_goals=NUM_GOAL_FRAMES*(2*29+9)
    goal_width=actor_goal_dim(num_goals)
    actor_width=930+goal_width
    critic_width=64+num_goals+16
    for mode in ('released_uint8','diagnostic_float'):
        torch.manual_seed(42)
        native_cfg=SuperMimicRunnerCfg()
        configure_runner(native_cfg)
        runner_class=make_native_ppo_runner(native_cfg,mode,tmp_path/mode)
        runner=runner_class.__new__(runner_class)
        runner.device='cpu'
        runner.cfg=PpoRunnerCfg(batch_size=8,num_steps_per_env=4,
            num_learning_epochs=1,num_mini_batches=2,actor_obs='proprio',history_length=10)
        representation=types.SimpleNamespace(proprio_obs_dim=lambda n:93,
            rep_state_dim=64,actor_obs_dim=lambda n,g,h:93*h+actor_goal_dim(g))
        runner._wm=types.SimpleNamespace(representation=representation,num_actions=29,num_goals=num_goals)
        runner._representation=representation
        runner._num_actions=29
        runner._num_goals=num_goals
        runner._sim_context=types.SimpleNamespace(output_dim=16)
        class SyntheticEnv:
            def obs(self):
                return TensorDict({'actor':torch.randn(8,actor_width),'critic':torch.randn(8,critic_width)},batch_size=[8])
            def reset(self):
                return self.obs()
            def step(self,actions):
                return self.obs(),1-actions.square().mean(-1),torch.linspace(.1,1.,8),{}
        runner._create_wm_env=lambda _:SyntheticEnv()
        runner._init_policy()
        initial_hashes.append(state_sha256(runner.alg.actor.native_actor))
        assert runner.alg.actor.mlp.goal_width==goal_width==640
        assert runner.alg.actor.mlp.input_width==actor_width==1570
        runner.imagine()
        record=runner.integration_updates[0]
        assert record['actor_and_critic_changed'] and record['imagined_steps']==32
        assert record['fractional_probability_positions']==28
        assert record['fractional_positions_truncated_to_zero']==(28 if mode=='released_uint8' else 0)
        with np.load(record['trace']['path'],allow_pickle=False) as saved:
            assert saved['probabilities'].shape==(4,8)
            if mode=='diagnostic_float':
                assert np.array_equal(saved['probabilities'],saved['stored_termination'])
            else:
                assert saved['stored_termination'].dtype==np.uint8
                assert np.array_equal(saved['stored_termination'],saved['probabilities'].astype(np.uint8))
    assert initial_hashes[0]==initial_hashes[1]
