"""Finite real-G1 PPO-WM integration; not a paper mechanism-comparison result."""
import argparse
import copy
import dataclasses
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import random
import time


def write(path, value):
    path.write_text(json.dumps(value,indent=2,default=str,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motions',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--termination',choices=('released_uint8','diagnostic_float'),required=True)
    args=parser.parse_args()
    output=args.output.resolve(); motions=args.motions.resolve()
    output.mkdir(parents=True,exist_ok=False)
    os.environ.update(WANDB_MODE='offline',MUJOCO_GL='disable')
    os.chdir(output)
    import numpy as np
    import torch
    import wandb
    import supermimic.representations as representations
    import supermimic.learners.wm_learner as wm_module
    import supermimic.runners.ppo_runner as ppo_module
    from supermimic.runners.ppo_runner import PpoRunnerCfg
    from supermimic.scripts.train import TrainConfig
    from supermimic.scripts.train_utils import create_env, setup_device
    from paper_profile import configure_runner, build_adam_optimizers
    from ppo_wm_native_runtime import make_native_ppo_runner, state_sha256
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('Exactly one allocated CUDA GPU required')
    ends=np.load(motions/'motion_ends.npy')
    assert ends.shape==(40,) and int(ends[-1])==441140
    np.random.seed(42); random.seed(42); torch.manual_seed(42)
    representations._compiled_assemble_goal=representations._assemble_goal
    wm_module.build_optimizers=build_adam_optimizers
    native=TrainConfig.from_task('SuperMimic-GenCoord-G1')
    configure_runner(native.runner)
    native.env.seed=42
    native.env.scene.num_envs=4096
    native.env.commands['motion'].motion_file=str(motions)
    native.env.commands['motion'].sampling_mode='uniform'
    cfg=PpoRunnerCfg(max_iterations=3,save_interval=1,log_interval=1,
        num_collect_steps=10,num_steps_per_env=32,batch_size=4096,
        actor_obs='proprio',history_length=10,
        dynamics_hidden_dims=native.runner.dynamics_hidden_dims,
        actor_hidden_dims=native.runner.actor_hidden_dims,
        critic_hidden_dims=native.runner.critic_hidden_dims,
        wm=copy.deepcopy(native.runner.wm),run_name='real-g1-ppo-wm-'+args.termination+'-seed42')
    snapshots=[]
    def local_artifact(files):
        records=[]
        folder=output/'released-checkpoints'/f'outer-{len(snapshots)+1:03d}'
        folder.mkdir(parents=True,exist_ok=False)
        for file in files:
            file=Path(file); dest=folder/file.name
            data=file.read_bytes(); dest.write_bytes(data)
            assert dest.read_bytes()==data
            records.append(dict(path=str(dest),bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
        snapshots.append(records)
    ppo_module.log_model_artifact=local_artifact
    config=dict(task='SuperMimic-GenCoord-G1',training_algorithm='PPO-WM',
        seed=42,num_envs=4096,outer_iterations=3,real_collections_per_outer=10,
        ppo_runner=dataclasses.asdict(cfg),native_network_architecture=dataclasses.asdict(native.runner),
        environment=dataclasses.asdict(native.env),termination_storage=args.termination,
        sampling_mode='uniform',main_method_return_sampler_matched=False,
        io_adapter='Local byte-verified checkpoint copies; offline W&B; online artifact pruning disabled',
        runtime_adaptations='Native motion actor/critic bridge; eager author goal math; WM Adam; explicit PPO termination storage',
        goal_backend='author eager goal math',world_model_optimizer='Adam',
        ppo_hyperparameters='Released PpoRunnerCfg defaults; paper matching unconfirmed',
        motion_directory=str(motions),clips=40,frames=441140,
        exact_paper_equivalence=False,quality_evaluation_completed=False,vendor_modified=False,
        source_files={Path(inspect.getfile(mod)).name:hashlib.sha256(Path(inspect.getfile(mod)).read_bytes()).hexdigest()
            for mod in (ppo_module,wm_module,representations)},
        torch_version=torch.__version__,rsl_version=importlib.metadata.version('rsl-rl-lib'))
    write(output/'resolved-config.json',config)
    device=setup_device('cuda:0')
    env=create_env(native.env,4096,device)
    runner_class=make_native_ppo_runner(native.runner,args.termination,output/'termination-traces')
    runner=runner_class(env=env,cfg=cfg,task_id='SuperMimic-GenCoord-G1',registry_name=None,device=device)
    initial=dict(actor=state_sha256(runner.alg.actor.native_actor),
        critic=state_sha256(runner.alg.critic.native_critic),world_model=state_sha256(runner._dynamics))
    dimensions=dict(actor_input=runner.alg.actor.mlp.input_width,
        goal=runner.alg.actor.mlp.goal_width,history=10,proprio=93,
        critic_input=runner._policy_input_dims()[1],actions=env.num_actions)
    write(output/'initial-model-state.json',dict(state_hashes=initial,dimensions=dimensions))
    started=time.monotonic()
    runner.run()
    elapsed=time.monotonic()-started
    assert len(runner.integration_updates)==len(snapshots)==3
    final=dict(actor=state_sha256(runner.alg.actor.native_actor),
        critic=state_sha256(runner.alg.critic.native_critic),world_model=state_sha256(runner._dynamics))
    assert all(initial[key]!=final[key] for key in initial)
    for model in (runner.alg.actor,runner.alg.critic,runner._dynamics):
        assert all(torch.isfinite(p).all() for p in model.parameters())
    actor_save=next(row['path'] for row in snapshots[-1] if Path(row['path']).name=='actor.pt')
    loaded=torch.load(actor_save,map_location='cpu',weights_only=True)
    assert set(loaded)=={'actor','critic'}
    for name,actual in runner.alg.actor.state_dict().items():
        assert torch.equal(actual.cpu(),loaded['actor'][name])
    for name,actual in runner.alg.critic.state_dict().items():
        assert torch.equal(actual.cpu(),loaded['critic'][name])
    wm_save=next(row['path'] for row in snapshots[-1] if Path(row['path']).name=='world_model.pt')
    loaded_wm=torch.load(wm_save,map_location='cpu',weights_only=True)
    for name,actual in runner._dynamics.state_dict().items():
        assert torch.equal(actual.cpu(),loaded_wm['dynamics'][name])
    for name,actual in runner._sim_context.state_dict().items():
        assert torch.equal(actual.cpu(),loaded_wm['sim_context'][name])
    result=dict(status='real_g1_native_network_ppo_wm_integration_completed',
        termination_mode=args.termination,outer_iterations=3,real_collections=30,
        real_simulator_steps=4096*30,real_native_ppo_updates=len(runner.integration_updates),
        imagined_transitions=sum(row['imagined_steps'] for row in runner.integration_updates),
        world_model_update_calls=3,wm_batches_per_call=10,wm_window_state_slots=8,
        wm_scored_transitions_per_window=7,elapsed_training_seconds=elapsed,
        gpu=torch.cuda.get_device_name(),dimensions=dimensions,
        initial_state_hashes=initial,final_state_hashes=final,
        all_three_native_models_changed_and_finite=True,updates=runner.integration_updates,
        native_released_save_format='world_model.pt bundle + actor.pt with actor/critic RSL state dicts',
        final_actor_and_critic_save_readback_verified=True,snapshots=snapshots,
        final_world_model_and_context_save_readback_verified=True,
        full_training_resume_available=False,
        source_config_sha256=hashlib.sha256((output/'resolved-config.json').read_bytes()).hexdigest(),
        g1_physics_executed=True,quality_evaluation_completed=False,
        paper_results_reproduced=False,matched_return_or_64x_advantage_claimed=False)
    write(output/'result.json',result)
    print('PPO_WM_REAL_G1_INTEGRATION_PASS '+json.dumps({k:result[k] for k in
        ('termination_mode','real_native_ppo_updates','real_simulator_steps','imagined_transitions')}),flush=True)


if __name__=='__main__':
    main()
