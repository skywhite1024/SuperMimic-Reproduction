"""Two actual Gloo processes against independently concatenated references."""
import copy
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import types

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from native_data_parallel import (broadcast_module_state,optimize_mean_gradient,
                                 SharedReturnSampler)


def process_check(rank,store,output):
    from rsl_rl.modules import EmpiricalNormalization
    from supermimic.models import build_mlp
    from supermimic.runners.reset_sampling import ReturnSampler,ReturnSamplerCfg
    from supermimic.runners.supermimic_runner import SuperMimicRunnerCfg,build_actor
    from supermimic.tasks.task_spec import build_actor_obs_td
    from paper_profile import configure_runner
    torch.set_num_threads(1)
    # This reference runs BEFORE any process group exists, so its original
    # normalizers cannot silently reduce already concatenated samples again.
    assert not dist.is_initialized()
    torch.manual_seed(123)
    cfg=SuperMimicRunnerCfg();configure_runner(cfg)
    shape=types.SimpleNamespace(representation=types.SimpleNamespace(rep_state_dim=64,
        proprio_obs_dim=lambda n:93),num_actions=29,num_goals=670)
    actor_reference=build_actor(cfg,shape,'cpu')
    torch.manual_seed(210)
    histories=torch.randn(4,10,93);goals=torch.randn(4,640)
    for shift in (0.,.3):
        actor_reference.update_normalization(build_actor_obs_td({'motion':goals+shift},histories+shift))
    # Native actor normalization samples only the latest history frame.
    assert actor_reference.obs_normalizer.count.item()==4*2
    dist.init_process_group('gloo',rank=rank,world_size=2,init_method=store,timeout=timedelta(seconds=45))
    try:
        torch.manual_seed(73+rank)
        critic=build_mlp(771,(256,128),1)
        broadcast_module_state(critic)
        reference=copy.deepcopy(critic)
        optimizer=torch.optim.Adam(critic.parameters(),lr=2e-4*2**.5)
        ref_optimizer=torch.optim.Adam(reference.parameters(),lr=2e-4*2**.5)
        torch.manual_seed(91)
        x=torch.randn(4,771);target=torch.tensor([[.5],[1.5],[-1.],[2.]])
        loss=(critic(x[rank*2:(rank+1)*2])-target[rank*2:(rank+1)*2]).square().mean()
        norm=optimize_mean_gradient([optimizer],loss,critic.parameters(),max_norm=.1,clip_value=.002)
        ref_loss=(reference(x)-target).square().mean()
        ref_optimizer.zero_grad(set_to_none=True);ref_loss.backward()
        ref_norm=torch.nn.utils.clip_grad_norm_(reference.parameters(),max_norm=.1)
        torch.nn.utils.clip_grad_value_(reference.parameters(),clip_value=.002)
        ref_optimizer.step()
        max_parameter_error=max(float((actual.detach()-expected.detach()).abs().max()) for actual,expected in
            zip(critic.parameters(),reference.parameters(),strict=True))
        max_gradient_error=max(float((actual.grad-expected.grad).abs().max()) for actual,expected in
            zip(critic.parameters(),reference.parameters(),strict=True))
        torch.testing.assert_close(norm,ref_norm,atol=2e-6,rtol=2e-6)
        print(json.dumps(dict(rank=rank,max_parameter_error=max_parameter_error,
            max_gradient_error=max_gradient_error,largest_parameter_differences=sorted([
            dict(name=name,error=float((value.detach()-reference.state_dict()[name]).abs().max()))
            for name,value in critic.named_parameters()],key=lambda row:row['error'],reverse=True)[:3])),flush=True)
        # Concatenated and two-shard FP32 GEMMs round differently. Adam can
        # amplify tiny near-zero gradient differences through its epsilon.
        # Check gradients tightly and account for that first-step transform,
        # rather than asserting bitwise single-batch optimizer equivalence.
        assert max_gradient_error<1e-7
        lr=2e-4*2**.5;epsilon=1e-8
        for actual,expected in zip(critic.parameters(),reference.parameters(),strict=True):
            analytic=-lr*(actual.grad/(actual.grad.abs()+epsilon)
                          -expected.grad/(expected.grad.abs()+epsilon))
            torch.testing.assert_close(actual.detach()-expected.detach(),analytic,atol=1e-7,rtol=.01)
        hasher=hashlib.sha256()
        for name,value in critic.state_dict().items():
            hasher.update(name.encode());hasher.update(value.detach().numpy().tobytes())
        critic_hash=hasher.hexdigest()
        check=torch.tensor(list(hasher.digest()),dtype=torch.uint8)
        peers=[torch.empty_like(check) for _ in range(2)]
        dist.all_gather(peers,check)
        assert torch.equal(peers[0],peers[1])

        torch.manual_seed(123+rank)
        actor=build_actor(cfg,shape,'cpu');broadcast_module_state(actor)
        normalizers=[module for module in actor.modules() if isinstance(module,EmpiricalNormalization)]
        for shift in (0.,.3):
            actor.update_normalization(build_actor_obs_td({'motion':goals[rank*2:(rank+1)*2]+shift},
                histories[rank*2:(rank+1)*2]+shift))
        assert actor.obs_normalizer.count.item()==4*2
        for name,value in actor.state_dict().items():
            torch.testing.assert_close(value,actor_reference.state_dict()[name],atol=1e-6,rtol=1e-6)
        actor_state_error=max(float((value-actor_reference.state_dict()[name]).abs().max())
            for name,value in actor.state_dict().items() if value.is_floating_point())
        actor.eval();actor_reference.eval()
        observation=build_actor_obs_td({'motion':goals},histories)
        with torch.no_grad():
            actual_mean=actor(observation);reference_mean=actor_reference(observation)
            torch.testing.assert_close(actual_mean,reference_mean,atol=2e-6,rtol=2e-6)
            actor_output_error=float((actual_mean-reference_mean).abs().max())
        assert sorted(n.count.item() for n in normalizers)==[0,8,8]

        sampler_cfg=ReturnSamplerCfg(horizon=2)
        shared=SharedReturnSampler(sampler_cfg,4,100,2,.1,'cpu')
        sampler_reference=ReturnSampler(sampler_cfg,4,100,4,.1,device='cpu')
        for step in range(4):
            frames=torch.tensor([step,step+2,step+50,step+70])
            rewards=torch.tensor([.01,.02,.03,.04])*(step+1)
            terminated=torch.tensor([True,False,False,False])
            clip=torch.tensor([False,step%2==0,False,False])
            timeout=torch.tensor([False,False,True,False])
            values=[frames,rewards,terminated,clip,timeout]
            actual=shared.record(*(value[rank*2:(rank+1)*2] for value in values))
            expected=sampler_reference.record(*values)
            assert (actual is None)==(expected is None)
            if actual is not None:torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        for name in ('value','count','effort'):
            torch.testing.assert_close(getattr(shared.native,name),getattr(sampler_reference,name),atol=0,rtol=0)
        assert shared.records==4 and shared.distribution_updates==2
        # A rank-local unused parameter is still averaged with zero from peers.
        from native_data_parallel import average_gradients
        unused=torch.nn.Parameter(torch.ones(3));unused.grad=torch.full((3,),2.) if rank==0 else None
        average_gradients([unused]);torch.testing.assert_close(unused.grad,torch.ones(3),atol=0,rtol=0)
        (Path(output)/f'rank-{rank}.json').write_text(json.dumps(dict(status='CPU_two_rank_collectives_verified',
            rank=rank,world_size=2,backend=dist.get_backend(),native_critic_input=771,
            actual_gradient_allreduce=True,clipping_after_global_average=True,
            maximum_parameter_error=max_parameter_error,maximum_gradient_error=max_gradient_error,
            single_concatenated_FP32_optimizer_bitwise_equivalence_claimed=False,
            FP32_Adam_step_difference_explained_by_observed_gradient_rounding=True,
            native_critic_state_sha256=critic_hash,rank_critic_state_hashes_identical=True,
            native_actor_input=dict(proprio=93,history=10,goal=640),
            normalizers=len(normalizers),active_normalizers=sum(n.count.item()>0 for n in normalizers),
            inactive_normalizers_preserved=True,normalizer_sample_counts=[0,8,8],
            normalizer_reference_was_computed_before_process_group=True,
            normalizer_extra_sample_gathering=False,maximum_actor_state_error=actor_state_error,
            maximum_actor_mean_error=actor_output_error,
            normalization_matches_independent_single_process_reference=True,
            sampler_records=4,sampler_distribution_updates=2,sampler_state_exact=True,
            locally_unused_parameter_averaged_as_zero=True,
            simulator_or_NCCL_executed=False,paper_two_H100_training_reproduced=False),indent=2)+'\n')
    finally:
        dist.destroy_process_group()


def test_two_actual_processes_native_models_and_return_sampler(tmp_path):
    store=(tmp_path/'gloo-store').resolve().as_uri()
    mp.spawn(process_check,args=(store,str(tmp_path)),nprocs=2,join=True)
    rows=[json.loads((tmp_path/f'rank-{rank}.json').read_bytes()) for rank in range(2)]
    assert [row['rank'] for row in rows]==[0,1]
    assert all(row['status']=='CPU_two_rank_collectives_verified' for row in rows)
