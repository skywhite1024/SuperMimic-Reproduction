"""Finite true-simulator PPO gate on the canonical40 motions and8192 G1s."""
import argparse
import dataclasses
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import time
import types


def write(path,value):
    path.write_text(json.dumps(value,indent=2,default=str,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def main():
    program_started=time.monotonic()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motions',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--updates',type=int,default=3)
    parser.add_argument('--checkpoint-updates',type=int,nargs='+')
    parser.add_argument('--trace-updates',type=int,nargs='+')
    parser.add_argument('--training-purpose',choices=('integration','finite-training'),default='integration')
    args=parser.parse_args()
    if args.updates<1 or (args.training_purpose=='integration' and args.updates!=3):
        parser.error('Positive update count; integration retains exactly three updates')
    checkpoint_updates=args.checkpoint_updates or list(range(1,args.updates+1))
    trace_updates=args.trace_updates or checkpoint_updates
    for values in (checkpoint_updates,trace_updates):
        if values!=sorted(set(values)) or not all(1<=i<=args.updates for i in values):
            parser.error('Checkpoint/trace schedules must be sorted unique updates within the run')
    if checkpoint_updates[-1]!=args.updates:
        parser.error('Final update must have an immutable checkpoint')
    if args.training_purpose=='finite-training' and args.checkpoint_updates is None:
        parser.error('Longer training requires an explicit sparse checkpoint schedule')
    output=args.output.resolve();motions=args.motions.resolve()
    output.mkdir(parents=True,exist_ok=False)
    os.chdir(output)
    os.environ.update(MUJOCO_GL='disable',WANDB_MODE='offline')
    import numpy as np
    import torch
    from rsl_rl.algorithms import PPO
    from rsl_rl.storage import RolloutStorage
    import supermimic.representations as representations
    from supermimic.models import build_mlp
    from supermimic.representations import create_representation
    from supermimic.runners.supermimic_runner import build_actor
    from supermimic.runners.reset_sampling import ReturnSampler
    from supermimic.scripts.train import TrainConfig
    from supermimic.scripts.train_utils import create_env,setup_device
    from supermimic.sim_context import SimContext
    from supermimic.utils.helpers import unpack_obs_dims
    from supermimic.utils.logger import Logger
    from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg
    from paper_profile import configure_runner
    from paper_ppo_models import NativeMotionPpoActor,NativePpoCritic
    from ppo_sim_runtime import SimulatorPolicyInput,SimulatorPpoLoop
    from ppo_wm_native_runtime import state_sha256
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('Exactly one allocated CUDA GPU required')
    ends=np.load(motions/'motion_ends.npy')
    assert ends.shape==(40,) and int(ends[-1])==441140
    torch.manual_seed(42);np.random.seed(42);random.seed(42)
    representations._compiled_assemble_goal=representations._assemble_goal
    task='SuperMimic-GenCoord-G1'
    native=TrainConfig.from_task(task);configure_runner(native.runner)
    native.env.seed=42;native.env.scene.num_envs=8192
    native.env.commands['motion'].motion_file=str(motions)
    native.env.commands['motion'].sampling_mode='external'
    source_cfg=unitree_g1_tracking_ppo_runner_cfg()
    source_algorithm=dataclasses.asdict(source_cfg.algorithm)
    accepted=set(inspect.signature(PPO.__init__).parameters)-{'self','actor','critic','storage','device'}
    algorithm_kwargs={key:value for key,value in source_algorithm.items() if key in accepted}
    assert source_cfg.num_steps_per_env==24
    config=dict(training_algorithm='PPO-Sim',task=task,seed=42,num_envs=8192,
        outer_updates=args.updates,rollout_steps=24,real_collections=24*args.updates,
        training_purpose=args.training_purpose,checkpoint_updates=checkpoint_updates,
        trace_updates=trace_updates,per_update_reward_done_storage_comparison=True,
        ppo_algorithm=algorithm_kwargs,algorithm_cfg_ignored_fields=sorted(set(source_algorithm)-accepted),
        mjlab_tracking_ppo_source=dataclasses.asdict(source_cfg),
        mjlab_tracking_ppo_source_sha256=hashlib.sha256(Path(inspect.getfile(unitree_g1_tracking_ppo_runner_cfg)).read_bytes()).hexdigest(),
        native_network_architecture=dataclasses.asdict(native.runner),
        environment=dataclasses.asdict(native.env),
        reset_sampling='native ReturnSampler; external motion-command distribution',
        reset_sampler_cfg=dataclasses.asdict(native.runner.return_sampler),
        observation_normalization='Native actor normalizers updated by RSL PPO from real rollout observations after each update',
        goal_backend='author eager goal math',no_world_model_constructed=True,
        world_model_updates=0,imagined_transitions=0,motion_directory=str(motions),
        corpus=dict(clips=40,frames=441140),quality_evaluation_completed=False,
        paper_hyperparameter_and_sampler_equivalence_confirmed=False,
        full_18h_1_4_8_GPU_comparison_completed=False,
        full_training_state_resume_available=False,vendor_files_modified=False)
    write(output/'resolved-config.json',config)
    report=dict(status='starting',updates=[],paper_results_reproduced=False,
        quality_evaluation_completed=False,world_model_updates=0,imagined_transitions=0)
    write(output/'result.json',report)
    env=None
    try:
        device=setup_device('cuda:0')
        env=create_env(native.env,8192,device)
        obs,_=env.reset()
        full_width,goal_width=unpack_obs_dims(obs)
        context=SimContext.from_env(env.unwrapped)
        representation=create_representation(native.runner.representation,full_width,env.num_actions,
            env.unwrapped.step_dt,env.unwrapped.cfg.decimation,
            env.unwrapped.scene['robot'].data.default_joint_pos[0].to(device),actor_obs_mode='proprio').to(device)
        representation.bind(env.unwrapped,context)
        shape=types.SimpleNamespace(representation=representation,num_actions=env.num_actions,num_goals=goal_width)
        native_actor=build_actor(native.runner,shape,device)
        inputs=SimulatorPolicyInput(representation,context,10,device)
        inputs.seed(obs);template=inputs.build(obs,env)
        actor=NativeMotionPpoActor(native_actor,action_dim=env.num_actions,device=device,
            init_std=source_cfg.actor.distribution_cfg['init_std'])
        critic_width=template['critic'].shape[-1]
        critic=NativePpoCritic(build_mlp(critic_width,native.runner.critic_hidden_dims,1).to(device),
            input_dim=critic_width,device=device)
        storage=RolloutStorage('rl',8192,24,template,[env.num_actions],device=device)
        algorithm=PPO(actor,critic,storage,device=device,**algorithm_kwargs)
        motion_cmd=env.unwrapped.command_manager.get_term('motion')
        max_step_reward=env.unwrapped.step_dt*sum(term.weight for term in native.env.rewards.values() if term.weight>0)
        sampler=ReturnSampler(native.runner.return_sampler,motion_cmd.bin_count,
            motion_cmd.motion.time_step_total,8192,max_step_reward,device=device)
        logger=Logger('native-ppo-sim-integration',8192,env.max_episode_length,device)
        loop=SimulatorPpoLoop(env,algorithm,inputs,24,logger=logger,sampler=sampler,
            require_parameter_change=args.training_purpose=='integration')
        dimensions=dict(actor_input=template['actor'].shape[-1],actor_goal=actor.mlp.goal_width,
            history=10,proprio=93,critic_input=critic_width,actions=env.num_actions)
        assert dimensions==dict(actor_input=1570,actor_goal=640,history=10,proprio=93,critic_input=771,actions=29)
        initial=dict(actor=state_sha256(native_actor),critic=state_sha256(critic.native_critic))
        report.update(status='training',dimensions=dimensions,initial_model_state_hashes=initial,
            gpu=torch.cuda.get_device_name(),max_step_reward=max_step_reward)
        write(output/'result.json',report)
        manifest=[]
        started=time.monotonic()
        for update in range(1,args.updates+1):
            row=loop.update(output/'trace'/f'update-{update:03d}.npz' if update in trace_updates else None)
            row['update_complete_seconds_since_program_entry']=time.monotonic()-program_started
            row['update_complete_seconds_since_training_entry']=time.monotonic()-started
            row['episode_statistics']=dict(recent_completed_episodes=len(logger.rewbuffer),
                mean_raw_episode_return=float(np.mean(logger.rewbuffer)) if logger.rewbuffer else None,
                scope='Native logger last100 completed real simulator episodes, not fixed-start full-clip return')
            logger.ep_extras.clear()
            report['updates']=loop.updates
            if update not in checkpoint_updates:
                if update%50==0:
                    write(output/'result.json',report)
                continue
            folder=output/'checkpoints'/f'update-{update:03d}';folder.mkdir(parents=True,exist_ok=False)
            models={'actor.pt':native_actor.state_dict(),'critic.pt':critic.native_critic.state_dict(),
                'ppo.pt':algorithm.save()}
            files=[]
            for name,payload in models.items():
                dest=folder/name;torch.save(payload,dest)
                loaded=torch.load(dest,map_location='cpu',weights_only=True)
                if name in ('actor.pt','critic.pt'):
                    assert set(loaded)==set(payload)
                    assert all(torch.equal(loaded[key],tensor.cpu()) for key,tensor in payload.items())
                else:
                    for key,module in [('actor_state_dict',actor),('critic_state_dict',critic)]:
                        assert all(torch.equal(loaded[key][n],tensor.cpu()) for n,tensor in module.state_dict().items())
                files.append(dict(file=name,path=str(dest),bytes=dest.stat().st_size,
                    sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),model_state_readback_verified=True))
            manifest.append(dict(ppo_update=update,collection_steps=loop.collections,files=files,
                retained_snapshot_complete_seconds_since_program_entry=time.monotonic()-program_started,
                retained_snapshot_complete_seconds_since_training_entry=time.monotonic()-started,
                native_actor_state_sha256=state_sha256(native_actor),
                native_critic_state_sha256=state_sha256(critic.native_critic)))
            report['updates']=loop.updates
            report['checkpoint_sets']=manifest
            write(output/'checkpoint-manifest.json',manifest)
            write(output/'result.json',report)
        assert loop.collections==24*args.updates and loop.simulator_steps==8192*24*args.updates
        assert len(manifest)==len(checkpoint_updates) and len(loop.updates)==args.updates
        assert loop.sampler_updates==loop.collections//native.runner.return_sampler.horizon
        completed_status=('real_G1_simulator_PPO_integration_completed' if args.training_purpose=='integration'
                          else 'real_G1_simulator_PPO_finite_training_completed')
        report.update(status=completed_status,
            ppo_update_calls=args.updates,collection_steps=loop.collections,simulator_steps=loop.simulator_steps,
            elapsed_training_seconds=time.monotonic()-started,
            return_sampler_distribution_updates=loop.sampler_updates,
            final_model_state_hashes=dict(actor=state_sha256(native_actor),critic=state_sha256(critic.native_critic)),
            all_model_and_trace_readbacks_verified=True,
            source_config_sha256=hashlib.sha256((output/'resolved-config.json').read_bytes()).hexdigest(),
            checkpoint_manifest_sha256=hashlib.sha256((output/'checkpoint-manifest.json').read_bytes()).hexdigest())
        write(output/'result.json',report)
        print('PPO_SIM_REAL_G1_TRAINING_PASS '+json.dumps(dict(ppo_updates=args.updates,simulator_steps=loop.simulator_steps,
            imagined_transitions=0,world_model_updates=0)),flush=True)
    except Exception as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}')
        write(output/'result.json',report)
        raise
    finally:
        if env is not None:env.close()


if __name__=='__main__':
    main()
