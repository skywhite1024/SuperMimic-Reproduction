"""Instrument a separate deterministic rollout; never changes acceptance metrics."""
from __future__ import annotations
import argparse
import gzip
import csv
import json
from pathlib import Path
import torch
from audit_common import file_hash, write_json
from unisupermimic.tasks.motion_tracking.g1.supermimic.evaluate import make_eval_env, load_policy
from unisupermimic.rl.algos.supermimic.geometry import unflatten_state, rot_angle_sq
from unisupermimic.rl.contracts.supermimic import SUPERMIMIC_STATE_KEY, SUPERMIMIC_TRANSITION_KEY

CLIPS = ['dance2_subject4', 'fallAndGetUp1_subject5', 'fallAndGetUp2_subject3']
FIELDS = ['alive', 'root_z', 'ref_z', 'height_error_m', 'tilt_component_error',
          'root_position_error_m', 'root_orientation_error_rad', 'yaw_error_rad',
          'joint_position_rmse_rad', 'action_rms', 'action_delta_rms', 'height_gate', 'tilt_gate', 'terminated', 'done']


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(0)
    saved = torch.load(args.checkpoint, map_location='cuda:0', weights_only=True)
    expected = json.loads(args.evaluation.read_text())
    assert file_hash(args.checkpoint) == expected['checkpoint_sha256']
    env = make_eval_env(args.checkpoint.parent, backend='mjwarp')
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        mdp = env.supermimic_mdp
        motion = mdp.motion
        policy = load_policy(env, saved)
        assert len(motion.names) == env.num_envs == 40
        indices = torch.tensor([motion.names.index(n) for n in CLIPS], device=env.device)
        _, reset = env.reset_to_frames(motion.clip_start.clone())
        current = reset[SUPERMIMIC_STATE_KEY]
        returns = torch.zeros(40, device=env.device)
        steps = torch.zeros(40, dtype=torch.int64, device=env.device)
        finished = torch.zeros(40, dtype=torch.bool, device=env.device)
        failed = torch.zeros_like(finished)
        history = None
        trace = []
        hist = int(saved['args']['hist'])
        while not bool(finished.all()):
            obs = mdp.obs(unflatten_state(current['s']), current['a_prev'], current['xi'], current['f'])
            history = obs.unsqueeze(1).repeat(1, hist, 1) if history is None else torch.cat((history[:, 1:], obs.unsqueeze(1)), dim=1)
            action = policy(history, mdp.goal(current['f'], int(saved['args']['goal_frames'])))
            state = env.step(action)
            tr = state.info[SUPERMIMIC_TRANSITION_KEY]
            alive = ~finished
            returns += torch.where(alive, tr['r'], torch.zeros_like(tr['r']))
            steps += alive
            failed |= alive & tr['term']
            finished |= alive & tr['done']
            nxt = unflatten_state(tr['s1'][indices])
            position, rotation, _, _, q, _ = nxt
            frames = tr['f'][indices] + 1
            ref_p, ref_R = motion.root_p[frames], motion.root_R[frames]
            height = (position[:, 2] - ref_p[:, 2]).abs()
            tilt = (rotation[:, 2, 2] - ref_R[:, 2, 2]).abs()
            yaw_delta = torch.atan2(rotation[:, 1, 0], rotation[:, 0, 0]) - torch.atan2(ref_R[:, 1, 0], ref_R[:, 0, 0])
            trace.append(torch.stack([alive[indices].float(), position[:, 2], ref_p[:, 2], height, tilt,
                torch.linalg.vector_norm(position - ref_p, dim=-1), torch.sqrt(rot_angle_sq(rotation, ref_R).clamp_min(0)),
                torch.atan2(torch.sin(yaw_delta), torch.cos(yaw_delta)), ((q - motion.q[frames])**2).mean(-1).sqrt(),
                (action[indices]**2).mean(-1).sqrt(), ((action[indices] - tr['a_prev'][indices])**2).mean(-1).sqrt(),
                (height > mdp.cfg.term_height).float(), (tilt > mdp.cfg.term_tilt).float(),
                tr['term'][indices].float(), tr['done'][indices].float()], dim=-1).detach())
            current = state.info[SUPERMIMIC_STATE_KEY]
        values = torch.stack(trace).cpu().numpy()
        normalized = returns / ((motion.clip_len - 1).float() * 0.1)
        summaries = {}
        for index, name in enumerate(motion.names):
            row = expected['per_clip'][name]
            assert int(steps[index]) == row['steps'] and bool(failed[index]) == row['failed'], name
            assert abs(float(normalized[index]) - row['norm_return']) < 1e-5, name
        for j, name in enumerate(CLIPS):
            a = values[:, j]
            a = a[a[:, 0] > 0.5]
            assert ((a[:, 13] > 0.5) == ((a[:, 11] > 0.5) | (a[:, 12] > 0.5))).all(), 'Recorded gates do not explain termination'
            with gzip.open(args.output / (name + '.csv.gz'), 'wt', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['step', 'time_seconds', *FIELDS])
                for i, row in enumerate(a): writer.writerow([i + 1, (i + 1) * 0.02, *row.tolist()])
            final = dict(zip(FIELDS, map(float, a[-1])))
            summaries[name] = {'steps': len(a), 'seconds': len(a)*0.02,
                'failed': expected['per_clip'][name]['failed'], 'last_transition': final,
                'max_height_error_m': float(a[:, 3].max()), 'max_tilt_component_error': float(a[:, 4].max()),
                'mean_joint_position_rmse_rad': float(a[:, 8].mean()),
                'height_gate_threshold_m': mdp.cfg.term_height, 'tilt_component_gate_threshold': mdp.cfg.term_tilt}
        write_json(args.output / 'summary.json', {'checkpoint_sha256': file_hash(args.checkpoint),
            'same_all40_rollout_metrics_reproduced': True, 'clips': summaries,
            'boundary': 'Joint-position tracking error and action magnitude/rate are distinct;diagnostics do not alter metrics or termination.'})
        print('CONTINUATION_DIAGNOSTICS ' + json.dumps({n: {'steps': v['steps'], 'failed': v['failed'], 'last_height': v['last_transition']['height_error_m'], 'last_tilt': v['last_transition']['tilt_component_error']} for n, v in summaries.items()}), flush=True)
    finally:
        env.close(); env.backend.close()


if __name__ == '__main__': main()
