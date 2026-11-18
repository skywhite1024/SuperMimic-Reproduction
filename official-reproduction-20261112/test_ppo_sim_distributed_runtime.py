"""Exercise native PPO gradient synchronization with independent rank observations."""

from datetime import timedelta
import types

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage
from supermimic.models import build_mlp
from supermimic.runners.reset_sampling import ReturnSamplerCfg
from supermimic.runners.supermimic_runner import SuperMimicRunnerCfg, build_actor

from native_data_parallel import SharedReturnSampler
from paper_ppo_models import NativeMotionPpoActor, NativePpoCritic
from paper_profile import configure_runner
from ppo_sim_runtime import SimulatorPolicyInput, SimulatorPpoLoop
from ppo_wm_native_runtime import state_sha256
from run_ppo_sim_integration import leader_clock, matched_rank_state
from test_ppo_sim_runtime import FakeRepresentation, SyntheticSimulator


def check_rank(rank, address):
    torch.manual_seed(42 + rank)
    torch.set_num_threads(2)
    dist.init_process_group("gloo", init_method=address, rank=rank, world_size=2, timeout=timedelta(seconds=45))
    try:
        env = SyntheticSimulator()
        env.t = rank
        original_step = env.step

        def step(actions):
            observation, rewards, dones = original_step(actions)
            return observation, rewards + .01 * rank, dones, {
                "time_outs": env.termination_manager.time_outs.clone()}

        env.step = step
        representation = FakeRepresentation()
        context = types.SimpleNamespace(read=lambda _: torch.zeros(8, 37))
        inputs = SimulatorPolicyInput(representation, context, 10, "cpu")
        cfg = SuperMimicRunnerCfg()
        configure_runner(cfg)
        shape = types.SimpleNamespace(representation=representation, num_actions=29, num_goals=670)
        actor = NativeMotionPpoActor(build_actor(cfg, shape, "cpu"), action_dim=29)
        critic = NativePpoCritic(build_mlp(771, (32, 16), 1), input_dim=771)
        inputs.seed(env.get_observations())
        template = inputs.build(env.get_observations(), env)
        storage = RolloutStorage("rl", 8, 4, template, [29], device="cpu")
        ppo = PPO(actor, critic, storage, num_learning_epochs=1, num_mini_batches=2,
                  device="cpu", multi_gpu_cfg={"global_rank": rank, "local_rank": rank, "world_size": 2})
        ppo.broadcast_parameters()
        sampler = SharedReturnSampler(ReturnSamplerCfg(horizon=2), 4, 1000, 8, .1, "cpu")
        before = matched_rank_state(torch, state_sha256(actor), 2)
        loop = SimulatorPpoLoop(env, ppo, inputs, 4, sampler=sampler)
        result = loop.update()
        assert result["simulator_steps"] == 32 and result["native_actor_normalization_count"] == 64
        assert result["sampler_distribution_updates"] == 2
        assert leader_clock(torch, 1.25 + rank, rank, "cpu") == 1.25
        final = matched_rank_state(torch, (state_sha256(actor), state_sha256(critic),
                                           state_sha256({"value": sampler.native.value})), 2)
        assert final[0] != before and result["native_actor_and_critic_changed_and_finite"]
    finally:
        dist.destroy_process_group()


def test_actual_two_cpu_ranks_keep_ppo_models_normalizer_and_clock_in_sync(tmp_path):
    mp.spawn(check_rank, args=((tmp_path / "ppo-sim-gloo").as_uri(),), nprocs=2, join=True)
