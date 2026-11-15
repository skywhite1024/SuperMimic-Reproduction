"""Audit recovered pose windows against original receipts and canonical motions.

Full-trajectory metrics remain cloud readback evidence. Only sampled arrays are
independently recomputed here. Rendering performs forward kinematics, not physics.
"""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8', newline='\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('recovered', 'verified-results', 'motions', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    assert not args.output.exists(), args.output
    spec = json.loads(Path(__file__).with_name('lafan-locomotion18-spec.json').read_bytes())
    assert {name: digest(args.motions / name) for name in spec['source_motion_files']} == spec['source_motion_files']
    names = json.loads((args.motions / 'motion_names.json').read_bytes())
    joints = json.loads((args.motions / 'joint_names.json').read_bytes())
    ends = np.load(args.motions / 'motion_ends.npy', allow_pickle=False)
    starts = np.r_[0, ends[:-1]]
    reference_arrays = [np.load(args.motions / (name + '.npy'), mmap_mode='r', allow_pickle=False)
                        for name in ('root_pos_w', 'root_quat_w', 'joint_pos',
                                     'root_lin_vel_w', 'root_ang_vel_w', 'joint_vel')]
    ledger = json.loads((args.recovered / 'receipt-ledger.json').read_bytes())['receipts']
    args.output.mkdir(parents=True)
    cases = []
    render_samples = []
    total_frames = 0
    for path in sorted(args.recovered.glob('ppo-wm-*.json')):
        original = next(row for row in ledger if Path(row['path']).name == path.name)
        assert digest(path) == original['sha256'] and path.stat().st_size == original['bytes']
        payload = json.loads(path.read_bytes())
        case = payload['case']
        assert payload['native_joint_names'] == joints
        source = args.verified_results / case / 'eval-002048'
        assert digest(source / 'result.json') == payload['source_result_sha256']
        assert digest(source / 'audit/summary.json') == payload['source_audit_sha256']
        audit = json.loads((source / 'audit/summary.json').read_bytes())
        assert audit['ledger']['sha256'] == payload['source_ledger_sha256']
        records = json.loads((source / 'result.json').read_bytes())['records']
        assert len(records) == len(payload['all18_metrics']) == 18
        assert [r['index'] for r in records] == list(range(18))
        score = lambda r: r['reward_sum'] / (r['reference_frames'] * .1)
        successes = sorted((r for r in records if r['success']), key=score)
        failures = sorted((r for r in records if not r['success']), key=lambda r: r['index'])
        selected = [r['index'] for r in [successes[0], successes[-1], failures[0]]]
        assert selected == payload['selected_indices']
        for row, metric in zip(records, payload['all18_metrics']):
            assert (row['index'], row['name'], row['success'], row['steps']) == (
                metric['index'], metric['name'], metric['native_success'], metric['steps'])
            assert abs(score(row) - metric['native_normalized_full_motion_return']) < 1e-14
        raw = base64.b64decode(payload['npz_base64'], validate=True)
        assert len(raw) == payload['npz_bytes']
        assert hashlib.sha256(raw).hexdigest() == payload['npz_sha256']
        (args.output / (case + '-samples.npz')).write_bytes(raw)
        windows = []
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            expected_keys = {str(i) + '_' + k for i in selected for k in
                             ('frame', 'state', 'reference', 'actions', 'reward', 'root_error', 'joint_rms')}
            assert set(archive.files) == expected_keys
            for i in selected:
                sample = {k: archive[str(i) + '_' + k] for k in
                          ('frame', 'state', 'reference', 'actions', 'reward', 'root_error', 'joint_rms')}
                row = records[i]
                n = row['steps']
                frame = sample['frame']
                expected = np.unique(np.r_[np.arange(min(n, 300)),
                    np.arange(max(0, n // 2 - 150), min(n, n // 2 + 150)),
                    np.arange(max(0, n - 300), n)]).astype(np.int64)
                assert np.array_equal(frame, expected)
                count = len(frame)
                assert sample['state'].shape == sample['reference'].shape == (count, 71)
                assert sample['actions'].shape == (count, 29)
                assert all(sample[k].shape == (count,) for k in ('reward', 'root_error', 'joint_rms'))
                assert all(np.isfinite(a).all() for a in sample.values())
                assert frame.dtype == np.int64
                assert all(sample[k].dtype == np.float32 for k in sample if k != 'frame')
                canonical_index = names.index(row['name'])
                assert ends[canonical_index] - starts[canonical_index] == row['reference_frames']
                target = np.concatenate([array[int(starts[canonical_index]) + frame]
                                         for array in reference_arrays], axis=1)
                assert np.array_equal(target, sample['reference'])
                state = sample['state']
                assert np.allclose(np.linalg.norm(state[:, 3:7], axis=1), 1, atol=2e-6, rtol=0)
                root = np.linalg.norm(state[:, :3].astype(np.float64) - target[:, :3], axis=1)
                joint = np.sqrt(np.mean((state[:, 7:36].astype(np.float64) - target[:, 7:36])**2, axis=1))
                assert np.allclose(root, sample['root_error'], atol=3e-6, rtol=2e-7)
                assert np.allclose(joint, sample['joint_rms'], atol=2e-7, rtol=2e-7)
                assert np.allclose(state[0], target[0], atol=3e-6, rtol=0)
                metric = payload['all18_metrics'][i]
                assert count == metric['sampled_frames']
                windows.append(dict(index=i, name=row['name'], native_success=row['success'],
                    sampled_frames=count, original_steps=n,
                    sampled_root_error_max_recompute_difference=float(np.max(np.abs(root-sample['root_error']))),
                    sampled_joint_rms_max_recompute_difference=float(np.max(np.abs(joint-sample['joint_rms']))),
                    sampled_max_world_root_distance=float(root.max()),
                    sampled_mean_joint_rms_degrees=float(np.rad2deg(joint.mean()))))
                total_frames += count
                render_samples.append((case, row, sample, metric))
        cases.append(dict(case=case, source_original_sha256=digest(path), npz_sha256=payload['npz_sha256'],
                          all18_metrics=payload['all18_metrics'], sampled_windows=windows))
    report = dict(status='recovered_sample_arrays_independently_verified_against_canonical_reference',
        cases=cases, sampled_frames=total_frames, local_full_trajectory_arrays_verified=False,
        cloud_full_trajectory_metrics_recomputed_locally=False, all54_endpoint_returns_recomputed=True,
        all_sampled_reference_values_bitwise_equal_to_canonical_pack=True,
        sample_windows_are_not_continuous_across_gaps=True, simulation_reexecuted=False,
        native_success_is_not_tracking_quality_certification=True, paper_results_reproduced=False)
    write(args.output / 'audit.json', report)
    if args.render:
        render(args.output, joints, render_samples)
    print(json.dumps(dict(status=report['status'], cases=len(cases), sampled_frames=total_frames,
                          rendering_requested=args.render), indent=2), flush=True)


def render(output, joints, samples):
    import mujoco
    import imageio_ffmpeg
    from PIL import Image, ImageDraw, ImageFont
    from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec, G1_XML
    spec = get_spec()
    spec.worldbody.add_geom(name='render_floor', type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=[50, 50, .1], rgba=[.94, .95, .97, 1])
    spec.worldbody.add_light(pos=[0, 0, 4], dir=[0, 0, -1], diffuse=[.8, .8, .8])
    model = spec.compile()
    data = mujoco.MjData(model)
    addresses = np.array([model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
                          for name in joints])
    assert set(addresses) == set(range(7, 36))
    camera = mujoco.MjvCamera()
    camera.azimuth = 135; camera.elevation = -15; camera.distance = 2.5
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 17)
    small = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 14)
    videos = []
    sheet = Image.new('RGB', (1664, 1720), '#101820')
    ImageDraw.Draw(sheet).text((12, 8), 'Saved PPO-WM poses: policy LEFT / reference RIGHT; independent pelvis-following cameras', font=font, fill='white')
    with mujoco.Renderer(model, height=240, width=416) as renderer:
        for ordinal, (case, row, sample, metric) in enumerate(samples):
            frames = sample['frame']; n = row['steps']
            segments = [np.arange(0, min(100, n), 2),
                        np.arange(max(0, n//2-50), min(n, n//2+50), 2),
                        np.arange(max(0, n-100), n, 2)]
            video = output / (case + '-' + str(row['index']).zfill(2) + '.mp4')
            writer = imageio_ffmpeg.write_frames(str(video), size=(832, 336), fps=25,
                codec='libx264', pix_fmt_in='rgb24', pix_fmt_out='yuv420p',
                output_params=['-crf', '23', '-preset', 'fast'])
            writer.send(None)
            count = 0; selected_frames = []
            try:
                for label, segment in zip(('FIRST window', 'MIDDLE window: time jump', 'LAST window: time jump'), segments):
                    for frame in segment:
                        k = int(np.searchsorted(frames, frame)); assert frames[k] == frame
                        state, target = sample['state'][k], sample['reference'][k]
                        canvas = Image.new('RGB', (832, 336), '#101820')
                        for x, pose in [(0, state), (416, target)]:
                            data.qpos[:7] = pose[:7]; data.qpos[addresses] = pose[7:36]
                            data.qvel[:] = 0; mujoco.mj_forward(model, data)
                            camera.lookat[:] = [float(pose[0]), float(pose[1]), .75]
                            renderer.update_scene(data, camera=camera)
                            canvas.paste(Image.fromarray(renderer.render().copy()), (x, 60))
                        draw = ImageDraw.Draw(canvas)
                        draw.text((8, 4), case + ' | ' + row['name'] + ' | native ' + ('SUCCESS' if row['success'] else 'FAILURE'), font=font, fill='white')
                        draw.text((8, 27), f'{label} | t={frame*.02:.2f}s | independent pelvis-following cameras', font=small, fill='#b5d3ef')
                        draw.text((8, 310), f'World root distance {sample["root_error"][k]:.3f}m | joint RMS {np.rad2deg(sample["joint_rms"][k]):.1f}deg | saved poses; no physics rerun', font=small, fill='white')
                        writer.send(np.asarray(canvas)); count += 1; selected_frames.append(int(frame))
                        if frame == segment[len(segment)//2] and label.startswith('MIDDLE'):
                            canvas.save(output / (case + '-' + str(row['index']).zfill(2) + '-preview.png'))
                            tile = canvas.resize((832, 336))
                            sheet.paste(tile, ((ordinal % 2)*832, 40+(ordinal//2)*336))
                assert count == 150
            finally:
                writer.close()
            decoded = imageio_ffmpeg.read_frames(str(video), pix_fmt='rgb24')
            meta = next(decoded); decoded_count = sum(1 for _ in decoded)
            assert decoded_count == count and meta['size'] == (832, 336) and meta['fps'] == 25
            videos.append(dict(file=video.name, sha256=digest(video), frames=count, fps=25,
                simulated_frame_indices=selected_frames, duration_seconds=6,
                source_case=case, source_index=row['index'], full_clip_native_success=row['success']))
            print('POSE_SAMPLE_RENDERED ' + json.dumps(dict(file=video.name, frames=count)), flush=True)
    sheet.save(output/'sample-overview.png')
    assets = [G1_XML, *sorted((G1_XML.parent/'assets').iterdir())]
    write(output/'rendering.json', dict(status='nine_sample_window_videos_rendered_and_decoded',
        videos=videos, joint_qpos_addresses=addresses.tolist(), physics_reexecuted=False,
        sample_window_concatenation_has_explicit_time_jump_labels=True,
        world_XY_drift_not_removed_from_numeric_errors=True,
        camera='Each panel follows its own pelvis; same distance/azimuth/elevation.',
        original_G1_visual_asset_SHA256={str(p.relative_to(G1_XML.parent)): digest(p) for p in assets if p.is_file()},
        overview_sha256=digest(output/'sample-overview.png'), no_new_quality_pass_claimed=True))


if __name__ == '__main__':
    main()
