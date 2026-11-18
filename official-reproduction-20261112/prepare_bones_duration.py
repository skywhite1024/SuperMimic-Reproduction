"""Build reproducible nested BONES duration packs and a common evaluation sample."""

import hashlib
import json
from pathlib import Path

import numpy as np

from push_recovery_contract import sha256, write_json

FIELDS = {"root_pos_w": 3, "root_quat_w": 4, "root_lin_vel_w": 3,
          "root_ang_vel_w": 3, "joint_pos": 29, "joint_vel": 29}
HOUR = 50 * 3600
CASES = ("1h", "7h", "40h", "full")


def read_identity(raw, filter_report, mask_path):
    names_file = raw / "motion_names.json"
    ends_file = raw / "motion_ends.npy"
    if (sha256(names_file) != filter_report["input_names_sha256"]
            or sha256(ends_file) != filter_report["input_motion_ends_sha256"]
            or sha256(mask_path) != filter_report["feasible_mask_sha256"]):
        raise ValueError("The raw motion order, clip boundaries or filter mask changed")
    names = json.loads(names_file.read_bytes())
    ends = np.load(ends_file, allow_pickle=False)
    lengths = np.diff(ends, prepend=0)
    if (len(names) != len(ends) or len(set(names)) != len(names) or (lengths <= 0).any()
            or len(names) != filter_report["raw_clips"]):
        raise ValueError("Invalid BONES corpus identity")
    bits = np.frombuffer(mask_path.read_bytes(), dtype=np.uint8)
    keep = np.unpackbits(bits, count=len(names), bitorder="little").astype(bool)
    if keep.sum() != filter_report["feasible_clips"] or lengths[keep].sum() != filter_report["feasible_frames"]:
        raise ValueError("Filter mask differs from original feasibility counts")
    return names, ends, lengths, np.flatnonzero(keep)


def choose_subsets(lengths, keep):
    order = np.random.Generator(np.random.PCG64(42)).permutation(keep)
    cumulative = np.cumsum(lengths[order])
    result = {}
    for hours in (1, 7, 40):
        count = np.searchsorted(cumulative, hours * HOUR) + 1
        if count > len(order):
            raise ValueError("There is insufficient data for the requested duration")
        result[f"{hours}h"] = np.sort(order[:count])
    result["full"] = np.sort(keep)
    # Uniform without replacement inside the shared 1-hour membership; no score is used.
    eligible = result["1h"][lengths[result["1h"]] >= 9]
    result["common64"] = np.sort(np.random.Generator(np.random.PCG64(20261118)).choice(
        eligible, 64, replace=False))
    return result


def contiguous_spans(indices, starts, ends):
    boundaries = np.r_[0, np.flatnonzero(np.diff(indices) > 1) + 1, len(indices)]
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        yield int(starts[indices[a]]), int(ends[indices[b - 1]])


def copy_pack(raw, output, names, ends, lengths, case, indices, filter_sha):
    output.mkdir(parents=True, exist_ok=False)
    selected = lengths[indices]
    target_ends = np.cumsum(selected, dtype=np.int64)
    starts = ends - lengths
    total = int(target_ends[-1])
    spans = tuple(contiguous_spans(indices, starts, ends))
    np.save(output / "motion_ends.npy", target_ends)
    write_json(output / "motion_names.json", [names[i] for i in indices])
    (output / "joint_names.json").write_bytes((raw / "joint_names.json").read_bytes())
    for field, width in FIELDS.items():
        filepath = f"{field}.npy"
        source = np.load(raw / filepath, mmap_mode="r", allow_pickle=False)
        if source.dtype != np.float32 or source.shape != (ends[-1], width):
            raise ValueError(f"Unexpected shape or dtype for {filepath}")
        target = np.lib.format.open_memmap(output / filepath, mode="w+", dtype=np.float32,
                                           shape=(total, width))
        cursor = 0
        for start, end in spans:
            block_len = end - start
            target[cursor:cursor + block_len] = source[start:end]
            cursor += block_len
        if cursor != total:
            raise ValueError(f"Unexpected frame count while materializing {case}: {filepath}")
        target.flush()
        del target
        # Check the edges of each selected block against the mmap source.
        readback = np.load(output / filepath, mmap_mode="r", allow_pickle=False)
        cursor = 0
        for start, end in spans:
            block_len = end - start
            if not (np.array_equal(source[start], readback[cursor])
                    and np.array_equal(source[end-1], readback[cursor+block_len-1])):
                raise RuntimeError(f"Readback mismatch in {case}: {filepath}")
            cursor += block_len
        del readback
    receipt = {
        "status": "bones_proxy_duration_pack_verified", "case": case,
        "clips": len(indices), "frames": total, "hours_at_50hz": total / HOUR,
        "paper_exact_membership": False, "filter_report_sha256": filter_sha,
        "membership_sha256": hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest(),
        "nested_seed": 42, "probe_seed": 20261118, "sampling": "nested whole-clip subsets",
        "common_probe_seen_in_training": True,
        "files": {path.name: sha256(path) for path in sorted(output.iterdir())},
    }
    write_json(output.with_name(output.name + "-duration-pack.json"), receipt)
    print(f"BONES_DURATION_PACK {case} clips={len(indices)} frames={total} hours={total / HOUR:.6f}",
          flush=True)
    return receipt
