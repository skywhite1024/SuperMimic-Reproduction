"""Finite two-H100 native first-order gate, throughput or training.

Uses native real collection, replay, world-model and actor/critic losses. Python
gradient collectives require eager learner updates, so the measured throughput
applies to this adapter.
"""
import argparse
import dataclasses
from datetime import datetime,timedelta,timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import time


STEADY_WINDOW = (201, 1001)
BONES_QUALITY_COLLECTIONS = 40001
BONES_CLIPS = 129543


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False,default=str)+'\n',encoding='utf-8',newline='\n')


def tensor_hash(value):
    value=value.detach().contiguous().cpu()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def model_hash(module):
    result=hashlib.sha256()
    for name,value in module.state_dict().items():
        result.update(name.encode());result.update(str(value.shape).encode())
        result.update(str(value.dtype).encode());result.update(bytes.fromhex(tensor_hash(value)))
    return result.hexdigest()


def parameter_hash(module):
    result=hashlib.sha256()
    for name,value in module.named_parameters():
        result.update(name.encode());result.update(bytes.fromhex(tensor_hash(value)))
    return result.hexdigest()


def save_rank_models(folder, runner, modules, device, torch):
    checkpoints = []
    payloads = {"world_model.pt": runner._wm.bundle(),
                "actor.pt": modules["actor"].state_dict(), "critic.pt": modules["critic"].state_dict()}
    for name, payload in payloads.items():
        path = folder / name
        torch.save(payload, path)
        loaded = torch.load(path, map_location=device, weights_only=True)
        if name == "world_model.pt":
            assert set(loaded) == {"dynamics", "sim_context"}
            for part, module in (("dynamics", runner._dynamics), ("sim_context", runner._sim_context)):
                assert set(loaded[part]) == set(module.state_dict())
                assert all(torch.equal(loaded[part][key], value) for key, value in module.state_dict().items())
        else:
            assert set(loaded) == set(payload)
            assert all(torch.equal(loaded[key], value) for key, value in payload.items())
        checkpoints.append({"file": name, "path": str(path), "bytes": path.stat().st_size,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "native_model_readback_verified": True})
    return checkpoints


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motions',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps', type=int, choices=(21, 1001, BONES_QUALITY_COLLECTIONS), default=21)
    parser.add_argument('--expected-clips', type=int, choices=(40, BONES_CLIPS), default=40)
    args=parser.parse_args()
    if (args.steps, args.expected_clips) not in {(21, 40), (1001, BONES_CLIPS), (BONES_QUALITY_COLLECTIONS, BONES_CLIPS)}:
        parser.error('Use the 21-step LAFAN gate, 1001-step BONES throughput or 40001-step BONES training')
    os.environ.update(MUJOCO_GL='disable',WANDB_MODE='offline')
    import numpy as np
    import torch
    import torch.distributed as dist
    from rsl_rl.modules import EmpiricalNormalization
    import supermimic.representations as representations
    from supermimic.runners.supermimic_runner import SuperMimicRunner,Ratio
    from supermimic.scripts.train import TrainConfig
    from supermimic.scripts.train_utils import create_env,setup_device
    from paper_profile import configure_runner,build_adam_optimizers,cosine_lr,set_learning_rate_in_place
    from native_data_parallel import install_native_runner_sync
    import supermimic.learners.policy_learner as policy_module
    import supermimic.learners.wm_learner as wm_module

    if int(os.environ.get('WORLD_SIZE','0'))!=2 or torch.cuda.device_count()!=2:
        raise RuntimeError('Exactly two allocated visible GPUs and two torchrun ranks required')
    rank=int(os.environ['RANK']);local_rank=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    properties=torch.cuda.get_device_properties(local_rank)
    if 'H100' not in properties.name or properties.total_memory<75_000_000_000:
        raise RuntimeError('This fixed integration requires full-memory H100 GPUs')
    dist.init_process_group('nccl',timeout=timedelta(seconds=180),device_id=torch.device(f'cuda:{local_rank}'))
    output=args.output.resolve();motions=args.motions.resolve()
    env=None;report=dict(status='starting',rank=rank,local_rank=local_rank,world_size=2,
        backend=dist.get_backend(),paper_results_reproduced=False,
        quality_evaluation_completed=False,full_training_state_resume_available=False)
    try:
        if rank==0:output.mkdir(parents=True,exist_ok=False)
        dist.barrier()
        folder=output/f'rank-{rank}';folder.mkdir(exist_ok=False)
        os.chdir(folder)
        device=setup_device(f'cuda:{local_rank}')
        cfg=TrainConfig.from_task('SuperMimic-GenCoord-G1');configure_runner(cfg.runner)
        cfg.runner.max_iterations=args.steps
        base_lrs=dict(world_model=cfg.runner.wm.lr,policy=cfg.runner.policy.lr)
        cfg.runner.wm.lr*=math.sqrt(2);cfg.runner.policy.lr*=math.sqrt(2)
        cfg.env.seed=42+rank;cfg.env.scene.num_envs=4096
        cfg.env.commands['motion'].motion_file=str(motions)
        cfg.env.commands['motion'].sampling_mode='external'
        assert np.load(motions/'motion_ends.npy').shape==(args.expected_clips,)
        representations._compiled_assemble_goal=representations._assemble_goal
        policy_module.build_optimizers=build_adam_optimizers
        wm_module.build_optimizers=build_adam_optimizers
        torch.manual_seed(42+rank);np.random.seed(42+rank);random.seed(42+rank)
        env=create_env(cfg.env,4096,device)
        # Share model initialization, then use independent real/replay streams.
        torch.manual_seed(42)
        runner=SuperMimicRunner(env,cfg.runner,'SuperMimic-GenCoord-G1',device=device)
        sync=install_native_runner_sync(runner)
        optimizers=[*runner.world_model_learner.optimizers,*runner.policy_learner.actor_optimizers,
                    *runner.policy_learner.critic_optimizers]
        assert len(optimizers)==3 and all(type(optimizer) is torch.optim.Adam for optimizer in optimizers)
        normalizer_source=Path(inspect.getfile(EmpiricalNormalization))
        config=dict(task='SuperMimic-GenCoord-G1',algorithm='native_first_order_two_rank',
            rank=rank,local_rank=local_rank,world_size=2,base_seed=42,environment_seed=42+rank,
            shared_initial_model_seed=42,local_num_envs=4096,global_num_envs=8192,
            collection_steps=args.steps,runner=dataclasses.asdict(cfg.runner),environment=dataclasses.asdict(cfg.env),
            base_learning_rates=base_lrs,learning_rate_multiplier=math.sqrt(2),
            synchronization=sync,goal_backend='author eager goal math',
            goal_assembly_backend='eager',
            optimizer='Adam for all three native networks, existing explicit A5 adaptation',
            learning_rate_schedule='sqrt2 times existing cosine schedule, updated before each local collection',
            native_RSL_normalizer_source_sha256=hashlib.sha256(normalizer_source.read_bytes()).hexdigest(),
            learner_CUDA_graphs_disabled=True,paper_performance_equivalence_claimed=False,
            full_training_state_resume_available=False)
        write(folder/'resolved-config.json',config)
        modules=dict(dynamics=runner._dynamics,actor=runner.policy_learner.actor,critic=runner.policy_learner.critic)
        report['initial_model_state_hashes']={name:model_hash(module) for name,module in modules.items()}
        report['initial_learned_parameter_hashes']={name:parameter_hash(module) for name,module in modules.items()}
        report['sim_context_state_sha256']=model_hash(runner._sim_context)
        report['gpu']=torch.cuda.get_device_name(local_rank)
        report['device_properties']=str(torch.cuda.get_device_properties(local_rank))
        torch.manual_seed(42+rank)
        env.reset();runner._seed_sim_history()
        first=env.get_observations()['full_state']
        assert torch.isfinite(first).all()
        report['initial_real_state_sha256']=tensor_hash(first)
        should_train=Ratio(env.num_envs/cfg.runner.wm.num_batches/cfg.runner.wm.batch_size/
            cfg.runner.wm.horizon*cfg.runner.replay_ratio)
        started=time.monotonic();outer_updates=0;local_sim_steps=0;observations=[]
        intermediate_snapshots = []
        if args.steps == BONES_QUALITY_COLLECTIONS:
            observed_steps = (1, *STEADY_WINDOW, 10001, 20001, 30001, BONES_QUALITY_COLLECTIONS)
        else:
            observed_steps = (1, 11, 21) if args.steps == 21 else (1, *STEADY_WINDOW)
        steady_started = None
        steady_elapsed = None
        for step in range(args.steps):
            set_learning_rate_in_place(optimizers,math.sqrt(2)*cosine_lr(step))
            local_sim_steps+=runner.step()
            for _ in range(should_train(step)):
                if runner.replay_buffer.num_transitions<cfg.runner.wm.horizon:continue
                data=(runner.replay_buffer.sample(cfg.runner.wm.horizon,cfg.runner.wm.batch_size)
                      for _ in range(cfg.runner.wm.num_batches))
                wm_logs=runner.world_model_learner.update(data)
                policy_logs=runner.policy_learner.update()
                assert all(torch.isfinite(value).all() for value in [*wm_logs.values(),*policy_logs.values()])
                outer_updates+=1
            collections = step + 1
            if collections == STEADY_WINDOW[1]:
                torch.cuda.synchronize(device)
                steady_elapsed = time.monotonic() - steady_started
            if collections in observed_steps or args.steps == 21:
                row=dict(collection_steps=collections,local_simulator_steps=local_sim_steps,
                    native_learner_outer_updates=outer_updates,
                    learning_rate=float(optimizers[0].param_groups[0]['lr']),
                    optimizer_steps=dict(runner._native_dp_optimizer_counts))
                assert math.isclose(row['learning_rate'],math.sqrt(2)*cosine_lr(step),rel_tol=1e-6)
                if collections in observed_steps:
                    row['model_state_hashes']={name:model_hash(module) for name,module in modules.items()}
                    observations.append(row)
                print('NATIVE_TWO_GPU_RANK_UPDATE '+json.dumps(dict(rank=rank,**row)),flush=True)
            if args.steps == BONES_QUALITY_COLLECTIONS and collections in {10001, 20001, 30001}:
                peers = [None, None]
                dist.all_gather_object(peers, observations[-1])
                if peers[0] != peers[1]:
                    raise RuntimeError(f"Two-rank learner state diverged at collection {collections}")
                if rank == 0:
                    checkpoint_folder = folder / "checkpoint-series" / f"step-{collections:09d}"
                    checkpoint_folder.mkdir(parents=True, exist_ok=False)
                    manifest = {"collection_steps_per_rank": collections, "global_real_transitions": collections * 8192,
                                "files": save_rank_models(checkpoint_folder, runner, modules, device, torch),
                                "seconds_since_runner_entry": time.monotonic() - started}
                    write(checkpoint_folder / "manifest.json", manifest)
                    intermediate_snapshots.append(manifest)
                    print("NATIVE_TWO_GPU_INTERMEDIATE_SNAPSHOT " + json.dumps(manifest), flush=True)
                dist.barrier(device_ids=[local_rank])
            if collections == STEADY_WINDOW[0]:
                torch.cuda.synchronize(device)
                dist.barrier(device_ids=[local_rank])
                steady_started = time.monotonic()
        assert outer_updates>0 and local_sim_steps==4096*args.steps
        assert all(value>0 for value in runner._native_dp_optimizer_counts.values())
        assert runner._sampler.records==args.steps and runner._sampler.distribution_updates>=1
        if args.steps == 21:
            assert runner._sampler.distribution_updates == 1
        assert all(torch.isfinite(value).all() for module in modules.values() for value in module.state_dict().values())
        final={name:model_hash(module) for name,module in modules.items()}
        final_parameters={name:parameter_hash(module) for name,module in modules.items()}
        assert all(final[name]!=report['initial_model_state_hashes'][name] for name in modules)
        assert all(final_parameters[name]!=report['initial_learned_parameter_hashes'][name] for name in modules)
        checkpoints = save_rank_models(folder, runner, modules, device, torch)
        report.update(status='two_rank_native_participant_completed',local_collection_steps=args.steps,
            local_simulator_steps=local_sim_steps,native_learner_outer_updates=outer_updates,
            optimizer_steps=dict(runner._native_dp_optimizer_counts),sparse_state_observations=observations,
            final_model_state_hashes=final,elapsed_training_seconds=time.monotonic()-started,
            steady_last_800_collections_seconds=steady_elapsed,
            clips=args.expected_clips,
            final_learned_parameter_hashes=final_parameters,all_native_learned_parameters_changed=True,
            sampler_real_records=runner._sampler.records,sampler_distribution_updates=runner._sampler.distribution_updates,
            actor_real_observation_count=int(runner.policy_learner.actor.obs_normalizer.count),
            shared_sampler_value_sha256=tensor_hash(runner._sampler.native.value),
            shared_sampler_count_sha256=tensor_hash(runner._sampler.native.count),
            shared_sampler_effort_sha256=tensor_hash(runner._sampler.native.effort),
            checkpoints=checkpoints,resolved_config_sha256=hashlib.sha256((folder/'resolved-config.json').read_bytes()).hexdigest())
        if args.steps == BONES_QUALITY_COLLECTIONS:
            report["intermediate_snapshots"] = intermediate_snapshots
        write(folder/'result.json',report)
        participants=[None,None];dist.all_gather_object(participants,report)
        assert all(row['status']=='two_rank_native_participant_completed' for row in participants)
        assert participants[0]['initial_model_state_hashes']==participants[1]['initial_model_state_hashes']
        assert participants[0]['final_model_state_hashes']==participants[1]['final_model_state_hashes']
        if args.steps > 21:
            assert participants[0]['sparse_state_observations'] == participants[1]['sparse_state_observations']
        assert participants[0]['initial_real_state_sha256']!=participants[1]['initial_real_state_sha256']
        assert {row['local_rank'] for row in participants}=={0,1}
        for key in ('optimizer_steps','actor_real_observation_count','shared_sampler_value_sha256',
                    'shared_sampler_count_sha256','shared_sampler_effort_sha256','sim_context_state_sha256'):
            assert participants[0][key]==participants[1][key], 'Rank state mismatch: '+key
        # Native actor normalizer uses one latest real observation per step.
        assert all(row['actor_real_observation_count']==8192*args.steps for row in participants)
        if rank==0:
            write(output/'result.json',dict(status='two_rank_native_training_completed',
                world_size=2,backend='nccl',local_collection_steps=args.steps,global_simulator_steps=8192*args.steps,
                initial_and_final_learned_models_identical_across_ranks=True,
                independent_real_simulator_streams_verified=True,normalizer_counts_and_sampler_states_shared=True,
                participants=participants,quality_evaluation_completed=False,
                full_training_state_resume_available=False,paper_results_reproduced=False,
                completed_at_utc=datetime.now(timezone.utc).isoformat()))
        dist.barrier()
    except Exception as error:
        if output.is_dir():
            report.update(status='failed',error=f'{type(error).__name__}: {error}')
            write(output/f'failed-rank-{rank}.json',report)
        raise
    finally:
        if env is not None:env.close()
        dist.destroy_process_group()


if __name__=='__main__':main()
