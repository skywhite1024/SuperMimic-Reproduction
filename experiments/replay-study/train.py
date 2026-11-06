"""Finite 4-ratio x 3-seed continued-learning study; engineering != main results."""
import argparse
from dataclasses import fields
import hashlib
import json
import random
import signal
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from study_runtime import ROOT, STUDY, make_env, plan, references, sha, verify_inputs
from sampling import FixedCollection, StudyReplay
from unisupermimic.rl.algos.supermimic.learner import SuperMimicConfig, SuperMimicLearner


def digest(module):
    h = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        h.update(key.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['engineering', 'pilot', 'main'], default='pilot')
    parser.add_argument('--old-fraction', type=float, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--steps', type=int)
    parser.add_argument('--max-seconds', type=int)
    args = parser.parse_args()
    spec = plan()
    assert args.old_fraction in spec['old_replay_fractions']
    assert args.seed in ([spec['pilot']['seed']] if args.phase == 'pilot' else spec['training_seeds'])
    if args.phase == 'main':
        assert spec['main_budget_frozen'], 'Freeze a measured common budget after the GPU pilot'
        assert args.device == 'cuda', 'Main comparisons require the same GPU backend'
        assert args.steps is None and args.max_seconds is None, 'Main runs must share the frozen budget'
    parent_path = (args.parent or ROOT/spec['parent_path']).resolve()
    assert sha(parent_path) == spec['parent_sha256']
    parent = torch.load(parent_path, map_location='cpu', weights_only=True)
    assert parent['iteration'] == 16000 and parent['pi']['obs_norm.mean'].numel() == 99
    engineering = args.phase == 'engineering'
    phase = {'steps': 4, 'wm_warmup': 1, 'max_seconds': 600} if engineering else spec['pilot' if args.phase == 'pilot' else 'provisional_main']
    steps = args.steps if args.steps is not None else phase['steps']
    maximum = args.max_seconds if args.max_seconds is not None else phase['max_seconds']
    assert 0 < steps <= phase['steps'] and 0 < maximum <= phase['max_seconds']
    assert steps > phase['wm_warmup']
    output = STUDY/'runs'/args.label
    assert output.resolve().is_relative_to((STUDY/'runs').resolve()) and not output.exists(), 'Use a fresh label'
    torch.set_num_threads(2 if args.device == 'cpu' else 4)
    if args.device == 'cuda':
        assert torch.cuda.device_count() == 1 and 'H100' in torch.cuda.get_device_name(0)
    config = {f.name: parent['args'][f.name] for f in fields(SuperMimicConfig) if f.name in parent['args']}
    for key in ['wm_hidden', 'policy_hidden', 'critic_hidden', 'test_hidden_dims']:
        if config.get(key): config[key] = tuple(config[key])
    common = dict(spec['common_training'])
    count = common.pop('num_envs')
    if engineering:
        count = 128
        common.update(buffer_steps=80, wm_batches=1, wm_batch=8, pi_batch=8,
                      critic_epochs=1, critic_minibatches=1, ckpt_every=1000, log_every=1)
    config.update(common, iterations=16000+steps, wm_only_iters=16000+phase['wm_warmup'],
                  seed=args.seed, amp='none' if args.device == 'cpu' else 'bf16')
    files, new_names = references()
    data_hashes = verify_inputs(files)
    collection = FixedCollection(new_names, spec['collection_old_fraction'])
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    started = time.perf_counter()
    env = make_env(files, args.device, count, args.seed, collection)
    learner = None
    deadline_hit = False
    try:
        learner = SuperMimicLearner(env, SuperMimicConfig(**config), output, device=args.device,
                                   backend='mujoco' if args.device == 'cpu' else 'mjwarp')
        learner.load(parent_path)
        for norm in [learner.pi.obs_norm, learner.pi.goal_norm, learner.wm.in_norm, learner.wm.out_norm, learner.critic.norm]:
            norm.max_count = 0
        replay = StudyReplay(learner.buffer, learner.mdp, learner.cfg.horizon, new_names, args.old_fraction, engineering)
        replay.install(learner)
        variant = {'study': spec['study'], 'phase': args.phase, 'old_replay_fraction': args.old_fraction,
                   'seed': args.seed, 'observation_dim': 99, 'reference_variant': 'raw', 'normalizers_fixed': True,
                   'parent_sha256': spec['parent_sha256'], 'plan_sha256': sha(STUDY/'plan.json')}
        native_checkpoint = learner._checkpoint
        learner._checkpoint = lambda include_optimizers: {**native_checkpoint(include_optimizers), 'variant': variant}
        before = {name: digest(getattr(learner, name)) for name in ['wm', 'pi', 'critic']}
        initial_steps = learner.sim_steps
        source = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        (output/'run.json').write_text(json.dumps({'variant': variant, 'source_commit': source, 'data_sha256': data_hashes,
            'source_dirty': bool(dirty), 'torch_version': torch.__version__, 'compute_device': args.device,
            'cuda_device_name': torch.cuda.get_device_name(0) if args.device == 'cuda' else None,
            'source_sha256': {p.name: sha(p) for p in [STUDY/'train.py', STUDY/'sampling.py', STUDY/'study_runtime.py']},
            'num_envs': count, 'planned_updates': steps, 'maximum_seconds': maximum}, indent=2), encoding='utf-8')
        native_collect = learner.collect
        def timed_collect(n):
            nonlocal deadline_hit
            if time.perf_counter()-started > maximum:
                deadline_hit = True
                raise TimeoutError('Finite program deadline reached; preserve partial run')
            return native_collect(n)
        learner.collect = timed_collect
        def interrupted(signum, frame):
            raise InterruptedError(f'Received signal {signum}')
        signal.signal(signal.SIGTERM, interrupted)
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        learner.train()
        destination = learner.save(output/'final-full.pt')
        saved = torch.load(destination, map_location='cpu', weights_only=True)
        assert all(bool(torch.isfinite(value).all()) for name in ['wm','pi','critic'] for value in saved[name].values())
        after = {name: digest(getattr(learner, name)) for name in before}
        assert all(before[name] != after[name] for name in before)
        assert all(torch.equal(value.detach().cpu(), saved[name][key]) for name in before for key, value in getattr(learner, name).state_dict().items())
        norms = all(torch.equal(parent[name][key], saved[name][key]) for name in before for key in parent[name]
                    if '_norm.' in key or key.startswith('norm.'))
        assert norms
        receipt = replay.receipt()
        for key in ['wm_starts', 'pi_consumed']:
            row = receipt['observed_sampling'][key]
            assert row['count'] > 0
            observed_old = 1-row['new_fraction']
            tolerance = max(.02, 6*(args.old_fraction*(1-args.old_fraction)/row['count'])**.5)
            assert abs(observed_old-args.old_fraction) <= tolerance
        result = {'status': 'real_updates_and_checkpoint_readback_verified', 'variant': variant,
                  'scientific_main_result': args.phase == 'main', 'device': args.device,
                  'new_updates': learner.iteration-16000, 'new_sim_transitions': learner.sim_steps-initial_steps,
                  'wm_updates': steps*config['wm_batches'], 'policy_updates': steps-phase['wm_warmup'],
                  'wall_seconds': time.perf_counter()-started, 'normalizer_tensors_unchanged': norms,
                  'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated() if args.device == 'cuda' else None,
                  'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved() if args.device == 'cuda' else None,
                  'before': before, 'after': after, 'collection': collection.receipt(), 'replay': receipt,
                  'checkpoint_sha256': sha(destination), 'physical_acceptance_inferred': False}
        (output/'proof.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print('REPLAY_STUDY_VERIFIED '+json.dumps({k: result[k] for k in ['status','variant','new_updates','wall_seconds']}), flush=True)
    except BaseException as error:
        if learner is not None:
            learner.save(output/'interrupted-full.pt')
        output.mkdir(parents=True, exist_ok=True)
        (output/'failure.json').write_text(json.dumps({'error': str(error), 'new_updates': learner.iteration-16000 if learner is not None else 0,
            'deadline_hit': deadline_hit, 'wall_seconds': time.perf_counter()-started}, indent=2), encoding='utf-8')
        raise
    finally:
        env.close(); env.backend.close()


if __name__ == '__main__':
    main()
