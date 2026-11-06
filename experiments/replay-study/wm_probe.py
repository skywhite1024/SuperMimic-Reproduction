"""Fixed evaluation-only dynamics windows; never inserted into a training buffer."""
import argparse
from dataclasses import fields
import json
from pathlib import Path
import numpy as np
import torch
from study_runtime import ROOT, STUDY, plan, references, sha, verify_inputs


def collect(parent, device):
    from runtime import make_env
    from unisupermimic.rl.algos.supermimic.geometry import unflatten_state
    from unisupermimic.rl.contracts.supermimic import SUPERMIMIC_STATE_KEY, SUPERMIMIC_TRANSITION_KEY
    from unisupermimic.tasks.motion_tracking.g1.supermimic.evaluate import load_policy
    output = STUDY/'runs'/f'frozen-probe-{device}'
    assert not output.exists(), 'Preserve the existing frozen probe'
    assert sha(parent) == plan()['parent_sha256']
    saved = torch.load(parent, map_location='cpu', weights_only=True)
    files, new_names = references()
    motion_hashes = verify_inputs(files)
    env = make_env(files, device=device, observation_dim=99, evaluation=True, num_envs=220)
    try:
        mdp = env.supermimic_mdp; motion = mdp.motion
        clips = torch.arange(55, device=device).repeat_interleave(4)
        phases = torch.tensor([0., .25, .5, .75], device=device).repeat(55)
        assert bool((motion.clip_len > 17).all())
        frames = motion.clip_start[clips] + (phases*(motion.clip_len[clips]-17)).long()
        _, info = env.reset_to_frames(frames)
        current = info[SUPERMIMIC_STATE_KEY]
        policy = load_policy(env, saved, device=device)
        def forbidden(*args, **kwargs): raise RuntimeError('Probe forbids resets after its initial phase states')
        env.reset = forbidden
        rows = {k: [] for k in ['s','s1','a','xi']}; history = None
        with torch.inference_mode():
            for _ in range(16):
                obs = mdp.obs(unflatten_state(current['s']), current['a_prev'], current['xi'], current['f'])
                history = obs[:,None].repeat(1,saved['args']['hist'],1) if history is None else torch.cat((history[:,1:],obs[:,None]),1)
                action = policy(history, mdp.goal(current['f'],saved['args']['goal_frames']))
                state = env.step(action)
                assert not bool(state.terminated.any() or state.truncated.any())
                transition = state.info[SUPERMIMIC_TRANSITION_KEY]
                for key in rows: rows[key].append(transition[key].detach().cpu().clone())
                current = state.info[SUPERMIMIC_STATE_KEY]
        packed = {}
        for key, values in rows.items():
            a = torch.stack(values)
            packed[key] = torch.cat([a[i:i+8].transpose(0,1) for i in [0,8]],0).numpy()
        packed['clip_id'] = clips.cpu().repeat(2).numpy()
        packed['new_group'] = np.array([motion.names[i].removesuffix('.csv') in new_names for i in packed['clip_id']])
        assert all(np.isfinite(value).all() for value in packed.values())
        output.mkdir(parents=True)
        np.savez_compressed(output/'windows.npz', **packed)
        metadata = {'backend': 'mujoco' if device == 'cpu' else 'mjwarp', 'device':device,
                    'window_count':len(packed['clip_id']), 'horizon':8, 'phases':[0.,.25,.5,.75],
                    'windows_per_clip':8, 'parent_sha256':sha(parent), 'data_sha256':sha(output/'windows.npz'),
                    'motion_names':list(motion.names), 'purpose':'evaluation-only within-library baseline-policy transitions; never used for updates',
                    'motion_sha256':motion_hashes,
                    'no_post_initialization_resets':True, 'not_unseen_motion_generalization':True}
        (output/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        print('FROZEN_WM_PROBE_CREATED '+json.dumps(metadata),flush=True)
    finally:
        env.close(); env.backend.close()


def evaluate(probe, checkpoints, destination):
    from sampling import FixedCollection
    from study_runtime import make_env
    from unisupermimic.rl.algos.supermimic.learner import SuperMimicConfig, SuperMimicLearner
    data = np.load(probe, allow_pickle=False)
    metadata = json.loads(probe.with_name('metadata.json').read_text(encoding='utf-8'))
    assert sha(probe) == metadata['data_sha256']
    assert not destination.exists(), 'Use a fresh diagnostics filename'
    files, new_names = references()
    verify_inputs(files)
    env = make_env(files, 'cpu', 128, 0, FixedCollection(new_names))
    results = []
    try:
        for index, checkpoint in enumerate(checkpoints):
            saved = torch.load(checkpoint,map_location='cpu',weights_only=True)
            config = {f.name:saved['args'][f.name] for f in fields(SuperMimicConfig) if f.name in saved['args']}
            config.update(compile=False, amp='none', buffer_steps=80)
            learner = SuperMimicLearner(env,SuperMimicConfig(**config),STUDY/'runs/probe-loads'/destination.stem/str(index),device='cpu',backend='mujoco')
            learner.load(checkpoint)
            for norm in [learner.wm.in_norm,learner.wm.out_norm]: norm.max_count=0
            groups = {}
            with torch.inference_mode():
                for name, selector in [('old',~data['new_group']),('additional',data['new_group'])]:
                    ids = np.flatnonzero(selector); total = 0.
                    for start in range(0,len(ids),16):
                        chosen = ids[start:start+16]
                        batch = {key:torch.from_numpy(data[key][chosen]) for key in ['s','s1','a','xi']}
                        prediction = learner._wm_rollout(batch['s'],batch['a'],batch['xi'])
                        loss, _ = learner._wm_err(prediction,batch['s1'],torch.ones(len(chosen),8))
                        assert bool(torch.isfinite(loss)), 'Nonfinite probe prediction error'
                        total += float(loss)*len(chosen)/8
                    groups[name] = {'windows':len(ids),'native_normalized_multistep_mse':total/len(ids)}
            results.append({'checkpoint':checkpoint.relative_to(ROOT).as_posix() if checkpoint.is_relative_to(ROOT) else checkpoint.name,
                            'checkpoint_sha256':sha(checkpoint),'variant':saved.get('variant'), 'groups':groups})
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_text(json.dumps({'probe':metadata,'prediction_compute_device':'cpu','results':results},indent=2),encoding='utf-8')
        print('WM_PROBE_EVALUATED '+json.dumps({'checkpoints':len(results),'probe_backend':metadata['backend']}),flush=True)
    finally:
        env.close(); env.backend.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collect',action='store_true');p.add_argument('--parent',type=Path)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--probe',type=Path);p.add_argument('--checkpoints',type=Path,nargs='+');p.add_argument('--output',type=Path)
    a=p.parse_args();torch.set_num_threads(2)
    if a.collect: collect((a.parent or ROOT/plan()['parent_path']).resolve(),a.device)
    else:
        assert a.probe and a.checkpoints and a.output
        evaluate(a.probe.resolve(),[p.resolve() for p in a.checkpoints],a.output.resolve())


if __name__=='__main__': main()
