"""Freeze seeded 120-second references before physical execution, with 29 segments."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from runtime import ROOT, UNI

FPS = 50


def resample(file):
    raw = np.loadtxt(file, delimiter=",")
    t = np.arange(len(raw)) / 30
    target = np.arange(int(np.floor(t[-1] * FPS)) + 1) / FPS
    out = np.empty((len(target), 36))
    out[:, :3] = np.stack([np.interp(target, t, raw[:, j]) for j in range(3)], 1)
    out[:, 3:7] = Slerp(t, Rotation.from_quat(raw[:, 3:7]))(target).as_quat()
    out[:, 7:] = np.stack([np.interp(target, t, raw[:, j]) for j in range(7, 36)], 1)
    return out


def compose(seed, blend_seconds=.6, segments=29, velocity_matching=False):
    rng = np.random.default_rng(seed)
    directory = UNI / "src/unisupermimic/assets/motions/g1/lafan1_supermimic"
    families = {name: sorted(directory.glob(name + "*.csv")) for name in ["walk", "run", "dance", "jumps"]}
    blend = round(blend_seconds * FPS)
    total = 120 * FPS + 1
    durations = rng.uniform(3, 6, size=segments)
    durations = np.rint(durations / durations.sum() * (total + (segments - 1) * blend)).astype(int)
    durations[-1] += total + (segments - 1) * blend - durations.sum()
    result = None
    rows = []
    for i, length in enumerate(durations):
        family = ["walk", "run", "dance", "jumps"][i % 4]
        file = families[family][int(rng.integers(len(families[family])))]
        src = resample(file)
        if length >= len(src):
            raise ValueError("Source clip is too short for the frozen segment")
        if result is None:
            offset = 0
        else:
            # Reference-only pose/velocity matching. No simulated robot state is available here.
            candidates = np.arange(0, len(src) - length, 5)
            anchor = result[-blend]
            score = np.mean((src[candidates, 7:] - anchor[7:]) ** 2, axis=1)
            score += .2 * (src[candidates, 2] - anchor[2]) ** 2
            if velocity_matching:
                src_qv = np.gradient(src[:,7:],1/FPS,axis=0)
                ref_qv = (result[-blend+1,7:]-result[-blend-1,7:])*(FPS/2)
                score += .02 * np.mean((src_qv[candidates]-ref_qv)**2,axis=1)
                src_v=np.gradient(src[:,:2],1/FPS,axis=0)[candidates]
                src_R=Rotation.from_quat(src[candidates,3:7]).as_matrix()
                yaw=np.arctan2(src_R[:,1,0],src_R[:,0,0])
                src_local=np.stack([np.cos(yaw)*src_v[:,0]+np.sin(yaw)*src_v[:,1],-np.sin(yaw)*src_v[:,0]+np.cos(yaw)*src_v[:,1]],1)
                ref_R=Rotation.from_quat(anchor[3:7]).as_matrix();ref_yaw=np.arctan2(ref_R[1,0],ref_R[0,0])
                ref_v=(result[-blend+1,:2]-result[-blend-1,:2])*(FPS/2)
                ref_local=np.array([np.cos(ref_yaw)*ref_v[0]+np.sin(ref_yaw)*ref_v[1],-np.sin(ref_yaw)*ref_v[0]+np.cos(ref_yaw)*ref_v[1]])
                score += .25*np.mean((src_local-ref_local)**2,axis=1)
            offset = int(candidates[np.argmin(score)])
        incoming = src[offset:offset + length].copy()
        incoming_rotation = Rotation.from_quat(incoming[:, 3:7])
        if result is None:
            incoming[:, :2] -= incoming[0, :2]
            incoming[:, 3:7] = incoming_rotation.as_quat()
            result = incoming
            boundary = 0
        else:
            boundary = len(result) - blend
            anchor = result[boundary]
            yaw0 = np.arctan2(incoming_rotation.as_matrix()[0, 1, 0], incoming_rotation.as_matrix()[0, 0, 0])
            r_anchor = Rotation.from_quat(anchor[3:7]).as_matrix()
            yaw1 = np.arctan2(r_anchor[1, 0], r_anchor[0, 0])
            turn = Rotation.from_euler("z", yaw1 - yaw0)
            incoming[:, :3] = turn.apply(incoming[:, :3] - incoming[0, :3]) + np.array([anchor[0], anchor[1], incoming[0, 2]])
            incoming[:, 3:7] = (turn * incoming_rotation).as_quat()
            t = np.linspace(0, 1, blend)
            weight = (6 * t**5 - 15 * t**4 + 10 * t**3)[:, None]
            merged = result[-blend:] * (1 - weight) + incoming[:blend] * weight
            first, second = result[-blend:, 3:7], incoming[:blend, 3:7]
            dot = np.sum(first * second, axis=1, keepdims=True)
            second = np.where(dot < 0, -second, second)
            dot = np.clip(np.abs(dot), 0, 1)
            theta = np.arccos(dot)
            den = np.sin(theta)
            qa = np.sin((1 - weight) * theta) / np.maximum(den, 1e-10)
            qb = np.sin(weight * theta) / np.maximum(den, 1e-10)
            quat = np.where(den < 1e-6, (1 - weight) * first + weight * second, qa * first + qb * second)
            merged[:, 3:7] = quat / np.linalg.norm(quat, axis=1, keepdims=True)
            result = np.concatenate([result[:-blend], merged, incoming[blend:]])
        rows.append({"segment": i + 1, "family": family, "source": file.name,
                     "source_sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                     "source_start_frame_50hz": offset, "frames": int(length),
                     "start_frame": int(boundary), "blend_end_frame": int(boundary + (blend if i else 0)),
                     "end_frame": len(result) - 1})
    assert result.shape == (total, 36) and np.isfinite(result).all()
    # An explicit .5-second terminal hold supplies future goal frames beyond t=120.
    result = np.concatenate([result, np.repeat(result[-1:], 25, axis=0)])
    destination = ROOT / ("references-velocity-matched" if velocity_matching else "references") / f"seed-{seed}"
    destination.mkdir(parents=True, exist_ok=True)
    csv = destination / "sequence.csv"
    np.savetxt(csv, result, delimiter=",", fmt="%.9g")
    metadata = {"seed": seed, "seconds": 120, "fps": FPS, "segments": rows,
                "reference_csv_sha256": hashlib.sha256(csv.read_bytes()).hexdigest(),
                "blend_seconds": blend_seconds, "blend": "quintic position/joint overlap + shortest-arc quaternion SLERP",
                "alignment": "one rigid world-XY/yaw transform per incoming segment, determined from the outgoing reference overlap; no online robot-based placement",
                "window_selection": "minimum joint-pose MSE plus .2 times root-height squared difference, 5-frame source stride" + ("; plus .02 joint-velocity MSE and .25 heading-local root-XY-velocity MSE" if velocity_matching else ""),
                "velocity": "computed by the unchanged upstream MotionLib from the fully composed 50 Hz reference",
                "future_windows": "one continuous clip; windows cross all internal segment boundaries",
                "terminal_hold_frames": 25, "source_scope": "existing public G1/LAFAN1 library; no flip coverage asserted"}
    (destination / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return str(csv)


if __name__ == "__main__":
    print(json.dumps([compose(seed) for seed in [0, 1, 2]]))
