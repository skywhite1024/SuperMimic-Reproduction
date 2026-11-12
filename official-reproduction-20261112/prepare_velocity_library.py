"""Build a byte-preserving walking/running candidate library from packed sources.

Membership is determined only by the frozen name expression, never returns or
successes. This is a local candidate definition, not the authors' unpublished
walking/running membership. Every parent file is rehashed before selection,
and every selected numeric element is compared against its parent after writing
the output. A historical protocol does not establish current source bytes.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np


NAME_RULE = r"(?:^|[_\-\s])(?:walk(?:ing)?|run(?:ning)?|jog(?:ging)?)(?=[_\-\s\d]|$)"
FIELDS = {"joint_pos.npy": 29, "joint_vel.npy": 29,
          "root_ang_vel_w.npy": 3, "root_lin_vel_w.npy": 3,
          "root_pos_w.npy": 3, "root_quat_w.npy": 4}
PACKED_FILES = set(FIELDS) | {"joint_names.json", "motion_names.json", "motion_ends.npy"}
CHUNK = 32768


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8", newline="\n")


def membership(label, folder, protocol):
    metadata = json.loads(protocol.read_text(encoding="utf-8"))
    hashes = metadata["motion_files"]
    if set(hashes) != PACKED_FILES:
        raise ValueError("Parent protocol must identify all nine packed files")
    for name in sorted(PACKED_FILES):
        if sha256(folder / name) != hashes[name]:
            raise ValueError("Parent packed-file hash mismatch: " + label + "/" + name)
    names = json.loads((folder / "motion_names.json").read_text(encoding="utf-8"))
    joints = json.loads((folder / "joint_names.json").read_text(encoding="utf-8"))
    ends = np.load(folder / "motion_ends.npy", allow_pickle=False)
    if (len(set(names)) != len(names) or ends.dtype != np.dtype("int64")
            or ends.shape != (len(names),) or not np.all(np.diff(ends, prepend=0) > 0)):
        raise ValueError("Invalid unique nonempty source clips: " + label)
    rows = []
    pattern = re.compile(NAME_RULE, re.IGNORECASE)
    for index, name in enumerate(names):
        if pattern.search(name):
            rows.append({"source": label, "source_index": index, "source_name": name,
                         "source_start": 0 if index == 0 else int(ends[index - 1]),
                         "source_end": int(ends[index]), "name": label + "__" + name})
    if not rows:
        raise ValueError("No candidate clips: " + label)
    receipt = {"label": label, "input_clips": len(names), "input_frames": int(ends[-1]),
               "selected_clips": len(rows),
               "selected_frames": sum(r["source_end"] - r["source_start"] for r in rows),
               "parent_protocol_sha256": sha256(protocol), "parent_packed_sha256": hashes,
               "parent_numeric_files_rehashed_this_run": True,
               "membership_files_rehashed_this_run": True}
    return rows, joints, int(ends[-1]), receipt


def build(sources, output):
    started = time.monotonic()
    if output.exists():
        raise FileExistsError("A fresh output directory is required")
    rows, receipts, source_frames, joint_names = [], [], {}, None
    for label, (folder, protocol) in sources.items():
        selected, joints, frames, receipt = membership(label, folder, protocol)
        if joint_names is not None and joints != joint_names:
            raise ValueError("Source joint ordering differs")
        joint_names = joints
        rows.extend(selected)
        source_frames[label] = frames
        receipts.append(receipt)
    if len(set(r["name"] for r in rows)) != len(rows):
        raise ValueError("Output clip names collide")
    cursor = 0
    for row in rows:
        row["output_start"] = cursor
        cursor += row["source_end"] - row["source_start"]
        row["output_end"] = cursor
    for index, row in enumerate(rows):
        row["output_index"] = index
    output.mkdir(parents=True, exist_ok=False)
    packed = output / "packed"
    packed.mkdir()
    write_json(packed / "joint_names.json", joint_names)
    write_json(packed / "motion_names.json", [r["name"] for r in rows])
    np.save(packed / "motion_ends.npy", np.array([r["output_end"] for r in rows], dtype=np.int64))
    comparisons = []
    for name, width in FIELDS.items():
        target = np.lib.format.open_memmap(packed / name, mode="w+", dtype=np.float32,
                                          shape=(cursor, width))
        for label, (folder, _) in sources.items():
            source = np.load(folder / name, mmap_mode="r", allow_pickle=False)
            if source.dtype != np.dtype("float32") or source.shape != (source_frames[label], width):
                raise ValueError("Source array contract differs: " + label + "/" + name)
            for row in rows:
                if row["source"] != label:
                    continue
                for first in range(row["source_start"], row["source_end"], CHUNK):
                    last = min(first + CHUNK, row["source_end"])
                    values = source[first:last]
                    if not np.isfinite(values).all():
                        raise ValueError("Nonfinite selected source values: " + name)
                    out = row["output_start"] + first - row["source_start"]
                    target[out:out + last - first] = values
            del source
        target.flush()
        del target
        # Reopen independently; comparing bytes also preserves signed zero.
        target = np.load(packed / name, mmap_mode="r", allow_pickle=False)
        compared = 0
        for label, (folder, _) in sources.items():
            source = np.load(folder / name, mmap_mode="r", allow_pickle=False)
            for row in rows:
                if row["source"] != label:
                    continue
                for first in range(row["source_start"], row["source_end"], CHUNK):
                    last = min(first + CHUNK, row["source_end"])
                    out = row["output_start"] + first - row["source_start"]
                    if source[first:last].tobytes() != target[out:out + last - first].tobytes():
                        raise ValueError("Selected output bytes differ: " + name)
                    compared += last - first
            del source
        del target
        if compared != cursor:
            raise RuntimeError("Incomplete array comparison: " + name)
        comparisons.append({"file": name, "frames_compared": compared,
                            "all_selected_bytes_equal": True})
        print("VELOCITY_LIBRARY_FIELD_VERIFIED " + name, flush=True)
    with (output / "membership.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    read_names = json.loads((packed / "motion_names.json").read_text(encoding="utf-8"))
    read_ends = np.load(packed / "motion_ends.npy", allow_pickle=False)
    if read_names != [r["name"] for r in rows] or read_ends.tolist() != [r["output_end"] for r in rows]:
        raise RuntimeError("Output membership readback differs")
    receipt = {"status": "verified_walking_running_library",
        "kind": "local_name_candidate_v1", "verification_scope": "membership rule and selected data bytes only",
        "created_at": datetime.now(timezone.utc).isoformat(), "fps": 50,
        "clips": len(rows), "frames": cursor, "duration_hours": cursor / 50 / 3600,
        "selection_rule": {"regex": NAME_RULE, "case_insensitive": True,
                           "selection_order": "source order bones then lafan40; original clip order within each",
                           "outcome_or_return_filter_applied": False, "clips_shortened": False},
        "author_membership_confirmed": False, "semantic_motion_quality_verified": False,
        "paper_results_reproduced": False, "parent_sources": receipts,
        "member_ledger": {"file": "membership.csv", "bytes": (output / "membership.csv").stat().st_size,
                          "sha256": sha256(output / "membership.csv")},
        "selected_array_comparisons": comparisons,
        "packed_files": [{"file": name, "bytes": (packed / name).stat().st_size,
                          "sha256": sha256(packed / name)} for name in sorted(PACKED_FILES)],
        "builder_sha256": sha256(Path(__file__)), "elapsed_seconds": time.monotonic() - started}
    write_json(output / "library-manifest.json", receipt)
    print("VELOCITY_LIBRARY_COMPLETED " + json.dumps(receipt), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bones", type=Path, required=True)
    parser.add_argument("--bones-protocol", type=Path, required=True)
    parser.add_argument("--lafan40", type=Path, required=True)
    parser.add_argument("--lafan40-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build({"bones": (args.bones.resolve(), args.bones_protocol.resolve()),
           "lafan40": (args.lafan40.resolve(), args.lafan40_protocol.resolve())}, args.output.resolve())


if __name__ == "__main__":
    main()
