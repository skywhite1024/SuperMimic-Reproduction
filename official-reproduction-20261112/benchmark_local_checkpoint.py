"""Local-checkpoint adaptation of the released benchmark, with explicit scopes.

Measures the released full simulator+actor path and WM dynamics-only forward
path separately. Their ratio is a diagnostic of these unequal scopes, not a
claim of a matched physics-only or whole-controller paper speedup. Each size
runs in a separate bounded process supplied by the launcher.
"""

import argparse
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
import traceback


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def timed_repeats(step, synchronize, *, warmup, steps, repeats):
    """Same explicit warmup and synchronized wall-clock accounting per path."""
    for _ in range(warmup):
        step()
    synchronize()
    durations = []
    for _ in range(repeats):
        synchronize()
        started = time.perf_counter()
        for _ in range(steps):
            step()
        synchronize()
        durations.append(time.perf_counter() - started)
    if any(not math.isfinite(value) or value <= 0 for value in durations):
        raise RuntimeError('Invalid synchronized timing interval')
    return durations


def validate_receipts(training, resolved, protocol):
    required = {'joint_names.json','motion_names.json','motion_ends.npy','joint_pos.npy',
        'joint_vel.npy','root_pos_w.npy','root_quat_w.npy','root_lin_vel_w.npy','root_ang_vel_w.npy'}
    if (training['status'] != 'pilot_completed' or training['collection_steps'] != 16001
            or resolved['goal_assembly_backend'] != 'eager'
            or resolved['task'] != 'SuperMimic-GenCoord-G1'
            or resolved['corpus']['clips'] != 40 or resolved['corpus']['frames'] != 441140):
        raise ValueError('Expected the frozen completed16k generalized-coordinate LAFAN actor')
    expected = {Path(v['path']).name:v['sha256'] for v in training['checkpoints']}
    actual = {Path(v['path']).name:v['sha256'] for v in protocol['training']['checkpoints']}
    if (expected != actual or protocol['clips'] != 40 or protocol['task'] != resolved['task']
            or protocol['training']['collection_steps'] != 16001
            or set(protocol['motion_files']) != required):
        raise ValueError('Motion protocol does not bind the same completed40clip model')
    return protocol['motion_files']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--motions', type=Path, required=True)
    parser.add_argument('--motion-protocol', type=Path, required=True)
    parser.add_argument('--num-envs', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--purpose', choices=('integration', 'experiment'), default='experiment')
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.num_envs not in (64,128,256,512,1024,2048,4096,8192,16384,32768,65536):
        parser.error('Use a declared released/paper batch size')
    if args.purpose == 'experiment' and (args.warmup,args.steps,args.repeats) != (20,500,3):
        parser.error('Experiment timing is fixed to20warmups,500steps,3repeats')
    if args.purpose == 'integration' and (args.num_envs > 128 or args.warmup != 2
            or args.steps != 2 or args.repeats != 1):
        parser.error('Integration is restricted to64/128 worlds and2warmups+2steps')
    if args.output.exists():
        raise FileExistsError('Use a fresh benchmark output')
    args.output.mkdir(parents=True)
    os.environ.update(WANDB_MODE='offline', MUJOCO_GL='disable', PYTHONNOUSERSITE='1')
    # Source training used synchronous diagnostics. Timings need the normal
    # asynchronous launch path plus explicit measurement-boundary syncs.
    if os.environ.get('CUDA_LAUNCH_BLOCKING') not in (None,'0'):
        raise RuntimeError('Unset CUDA_LAUNCH_BLOCKING for this isolated timing process')
    record = {'status':'benchmark_case_incomplete','purpose':args.purpose,'num_envs':args.num_envs,
        'warmup_calls_per_path':args.warmup,'timed_calls_per_repeat':args.steps,
        'repeats':args.repeats,'worker_sha256':sha(Path(__file__)),
        'comparison_scope':'full native env+actor versus WM dynamics-only; diagnostic unequal-scope ratio',
        'paper_speedup_reproduced':False,'policy_quality_evaluated':False}
    env = None
    try:
        import torch
        import supermimic.tasks
        import supermimic.representations as representations
        from mjlab.tasks.registry import load_env_cfg
        from mjlab.utils.torch import configure_torch_backends
        from supermimic.representations import extract_actor_goal
        from supermimic.runners.run_config import load_runner_cfg, check_representation_matches
        from supermimic.runners.supermimic_runner import build_actor, build_world_model
        from supermimic.scripts.train_utils import create_env, setup_device
        from supermimic.tasks.task_spec import build_actor_obs_td

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('Exactly one allocated CUDA GPU is required')
        configure_torch_backends()
        device = setup_device('cuda:0')
        training = json.loads((args.run/'result.json').read_text())
        resolved = json.loads((args.run/'resolved-config.json').read_text())
        motion_protocol = json.loads(args.motion_protocol.read_text())
        expected_motions = validate_receipts(training,resolved,motion_protocol)
        files = {}
        for item in training['checkpoints']:
            path = Path(item['path'])
            if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
                raise ValueError('Checkpoint hash/size differs: '+path.name)
            files[path.name] = path
        if set(files) != {'world_model.pt','actor.pt','critic.pt'}:
            raise ValueError('Complete frozen three-model receipt required')
        # Match the exact training motion bytes; do not rely on a historical path.
        packed = [{'file':name,'bytes':(args.motions/name).stat().st_size,
                   'sha256':sha(args.motions/name)} for name in sorted(expected_motions)]
        if {v['file']:v['sha256'] for v in packed} != expected_motions:
            raise ValueError('Motion pack differs from checkpoint training inputs')
        representations._compiled_assemble_goal = representations._assemble_goal
        runner = load_runner_cfg({'runner_cfg':resolved['runner']})
        check_representation_matches(resolved['task'], runner.representation)
        cfg = load_env_cfg(resolved['task'], play=True)
        cfg.seed = 42
        cfg.commands['motion'].motion_file = str(args.motions.resolve())
        env = create_env(cfg,args.num_envs,device)
        wm = build_world_model(runner,env,device)
        wm.load_bundle(torch.load(files['world_model.pt'],map_location=device,weights_only=True))
        wm.dynamics.eval()
        actor = build_actor(runner,wm,device)
        actor.load_state_dict(torch.load(files['actor.pt'],map_location=device,weights_only=True))
        actor.set_slot_mix({'motion':1.0})
        actor.eval()
        sim_ctx = wm.sim_context.read(env.unwrapped)
        rep = wm.representation
        if abs(env.unwrapped.step_dt-.02)>1e-9 or cfg.decimation!=4 or abs(cfg.sim.mujoco.timestep-.005)>1e-9:
            raise ValueError('Expected50Hz control with four5ms MuJoCo substeps')
        record.update(gpu=torch.cuda.get_device_name(0),torch=torch.__version__,
            torch_cuda=torch.version.cuda,tf32_matmul_allowed=torch.backends.cuda.matmul.allow_tf32,
            float32_matmul_precision=torch.get_float32_matmul_precision(),
            control_dt=env.unwrapped.step_dt,physics_dt=cfg.sim.mujoco.timestep,decimation=cfg.decimation,
            run_result_sha256=sha(args.run/'result.json'),resolved_config_sha256=sha(args.run/'resolved-config.json'),
            motion_protocol_sha256=sha(args.motion_protocol),
            checkpoints={name:sha(path) for name,path in files.items()},packed_files=packed,
            dimensions={'full_state':rep.full_state_dim,'rep_state':rep.rep_state_dim,
                'actions':wm.num_actions,'sim_context':wm.sim_context.output_dim,
                'actor_history':runner.history_length},
            goal_assembly_backend='eager',env_configuration=dataclasses.asdict(cfg),
            released_benchmark_changes=['Local frozen result/config instead ofWandB','Explicit motion pack hashes',
                f'Symmetric{args.warmup}warmup calls',f'{args.repeats} synchronized{args.steps}step repeats',
                'Both grad-recording and inference-only WM modes','No failures silently accepted as0SPS'],
            history_protocol='Released benchmark shape-padding: repeat current sensor across history, not sequential control-history quality test')
        free,total = torch.cuda.mem_get_info()
        record['cuda_device_memory_before_timing'] = {'free_bytes':free,'total_bytes':total,
            'scope':'Device-level snapshot, includes native/Warp allocations and other device consumers'}
        torch.cuda.reset_peak_memory_stats()
        # Keep the entire env reset/step sequence in one inference context:
        # contact histories created here must also be reset in this context.
        with torch.inference_mode():
            obs,_ = env.reset()
            sim_evidence = {'finite_pre_and_post':False}
            if not torch.isfinite(obs['full_state']).all():
                raise RuntimeError('Nonfinite initial simulator state')
            def simulator_step():
                nonlocal obs
                state = rep.represent(obs['full_state'])
                sensor = rep.compute_actor_sensor(state,sim_ctx,
                    sim_actor_obs=obs['actor'] if runner.actor_obs=='proprio' else None)
                history = sensor.unsqueeze(1).repeat(1,runner.history_length,1)
                actions = actor(build_actor_obs_td({'motion':extract_actor_goal(obs['goal'])},history))
                obs,_,_,_ = env.step(actions)
            durations = timed_repeats(simulator_step,torch.cuda.synchronize,
                warmup=args.warmup,steps=args.steps,repeats=args.repeats)
            if not torch.isfinite(obs['full_state']).all():
                raise RuntimeError('Nonfinite post-benchmark simulator state')
            sim_evidence.update(finite_pre_and_post=True,seconds_per_repeat=durations,
                steps_per_second=[args.steps*args.num_envs/value for value in durations],
                included=['actor/sensor/history shape padding','envstep:action processing,4physics substeps,control/scene updates',
                    'native rewards,terminations,observations,events and automatic resets'])
            record['simulator_with_policy'] = sim_evidence
            record['simulator_torch_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            record['simulator_torch_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
            # Release the simulator before WM timing as the released script
            # does; otherwise the largest WM batch inherits its VRAM footprint.
            env.close()
            env = None
            # Match the released WM workload: fixed random inputs each call,
            # outputs are not fed back. This is forward throughput, not an
            # accuracy test or an autoregressive trajectory benchmark.
        del actor,sim_ctx
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        free,total = torch.cuda.mem_get_info()
        record['cuda_device_memory_after_simulator_close'] = {'free_bytes':free,'total_bytes':total}
        random = torch.Generator(device=device).manual_seed(42)
        state = torch.randn((args.num_envs,rep.rep_state_dim),device=device,generator=random)
        action = torch.randn((args.num_envs,wm.num_actions),device=device,generator=random)
        aux = torch.randn((args.num_envs,7),device=device,generator=random)
        context = torch.randn((args.num_envs,wm.sim_context.output_dim),device=device,generator=random)
        modes = []
        for inference in (False,True):
            with torch.inference_mode(inference):
                # Tensors above are ordinary tensors, so autograd-enabled mode
                # never consumes inference tensors saved for parameter gradients.
                output = None
                def model_step():
                    nonlocal output
                    output = wm.dynamics(state,action,aux,context)
                durations = timed_repeats(model_step,torch.cuda.synchronize,
                    warmup=args.warmup,steps=args.steps,repeats=args.repeats)
                if not isinstance(output,tuple) or len(output)!=2 or not all(torch.isfinite(x).all() for x in output):
                    raise RuntimeError('Invalid/nonfinite WM forward output')
                modes.append({'inference_mode':inference,'autograd_enabled':torch.is_grad_enabled(),
                    'seconds_per_repeat':durations,
                    'steps_per_second':[args.steps*args.num_envs/value for value in durations],
                    'included':'Dynamics forward:normalizers,action transform,MLP,state integration; noactor,reward,FKgoals,history,observations or backward pass',
                    'inputs':'Seed42 fixed random batch, reused eachcall as released benchmark; physical validity not claimed'})
        record.update(status='benchmark_case_completed',world_model_modes=modes,
            wm_torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            wm_torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            simulator_median_sps=statistics.median(sim_evidence['steps_per_second']),
            diagnostic_wm_to_sim_scope_ratios={str(m['inference_mode']):statistics.median(m['steps_per_second'])
                /statistics.median(sim_evidence['steps_per_second']) for m in modes},
            differentiation_speed_or_full_control_speedup_claimed=False)
    except Exception as error:
        record.update(status='benchmark_case_failed',error=f'{type(error).__name__}: {error}')
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as close_error:
                record['close_error'] = f'{type(close_error).__name__}: {close_error}'
        (args.output/'result.json').write_text(json.dumps(record,indent=2,default=str,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
        print('LOCAL_BENCHMARK_CASE '+json.dumps({k:v for k,v in record.items() if k not in
            {'env_configuration','packed_files','checkpoints'}}),flush=True)


if __name__=='__main__':
    main()
