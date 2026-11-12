"""Write the author's packed-motion layout with bounded host RAM.

The native packer loads every NPZ before concatenation. This writer creates the
same sorted-name arrays using disk-backed slices, so the 14+ GB corpus does not
exhaust a 32 GB Windows machine. It does not resample or alter motion values.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from mjlab.scripts.pack_motions import FIELDS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    rows = [
        json.loads(line) for line in (args.source / "conversion-manifest.jsonl").open()
    ]
    if any(row.get("status", "converted") != "converted" for row in rows):
        raise ValueError("Resolve failed conversions before packing the full corpus")
    rows.sort(key=lambda row: row["name"])
    if args.limit is not None:
        rows = rows[: args.limit]
    args.output.mkdir(parents=True, exist_ok=False)
    ends = np.cumsum([row["frames"] for row in rows], dtype=np.int64)
    np.save(args.output / "motion_ends.npy", ends)
    (args.output / "motion_names.json").write_text(
        json.dumps([row["name"] for row in rows]), encoding="utf-8"
    )
    with np.load(args.source / "npz" / (rows[0]["name"] + ".npz")) as first:
        joints = first["joint_names"].tolist()
        arrays = {
            field: np.lib.format.open_memmap(
                args.output / (field + ".npy"),
                mode="w+",
                dtype=first[field].dtype,
                shape=(int(ends[-1]), *first[field].shape[1:]),
            )
            for field in FIELDS
        }
    (args.output / "joint_names.json").write_text(json.dumps(joints), encoding="utf-8")
    begin = 0
    for index, row in enumerate(rows):
        target = args.source / "npz" / (row["name"] + ".npz")
        with np.load(target) as clip:
            if clip["joint_names"].tolist() != joints or clip["fps"].item() != 50:
                raise ValueError(f"Joint/FPS mismatch: {target}")
            end = int(ends[index])
            for field in FIELDS:
                values = clip[field]
                if values.dtype != arrays[field].dtype or len(values) != end - begin:
                    raise ValueError(f"Shape/dtype mismatch: {target} {field}")
                arrays[field][begin:end] = values
        begin = end
        if (index + 1) % 10000 == 0:
            print(
                json.dumps({"packed_clips": index + 1, "total": len(rows)}), flush=True
            )
    for array in arrays.values():
        array.flush()
    files = []
    for path in sorted(args.output.iterdir()):
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        files.append(
            {"file": path.name, "bytes": path.stat().st_size, "sha256": digest}
        )
    result = {
        "status": "packed",
        "clips": len(rows),
        "frames": int(ends[-1]),
        "fps": 50,
        "files": files,
        "name_filter_applied": False,
        "feasibility_filter_applied": False,
    }
    (args.output.parent / (args.output.name + "-verification.json")).write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({k: v for k, v in result.items() if k != "files"}), flush=True)


if __name__ == "__main__":
    main()
