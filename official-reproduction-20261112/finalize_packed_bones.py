"""Independently read back previously extracted packed motions and finalize them."""

import argparse
import hashlib
import json
import time
from pathlib import Path


FIELDS = {
    "joint_names.json", "motion_names.json", "motion_ends.npy", "joint_pos.npy",
    "joint_vel.npy", "root_pos_w.npy", "root_quat_w.npy", "root_lin_vel_w.npy",
    "root_ang_vel_w.npy",
}


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--manifest-sha256", required=True)
    p.add_argument("--receipt", type=Path, required=True)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    started = time.monotonic()
    if sha256(args.manifest) != args.manifest_sha256:
        raise RuntimeError("Manifest SHA256 mismatch")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = {row["file"]: row for row in manifest["files"]}
    if set(records) != FIELDS or len(manifest["files"]) != len(FIELDS):
        raise RuntimeError("Unexpected manifest fields")
    if {p.name for p in args.source.iterdir()} != FIELDS:
        raise RuntimeError("Extracted directory must contain exactly the nine expected files")
    for name, record in records.items():
        path = args.source / name
        if (not path.is_file() or path.is_symlink() or path.stat().st_size != record["bytes"]
                or sha256(path) != record["sha256"]):
            raise RuntimeError("Readback mismatch: " + name)
        print("READBACK_PASS " + name, flush=True)
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.source.rename(args.output)
    receipt = {
        "status": "packed_files_readback_verified", "clips": manifest["clips"],
        "frames": manifest["frames"], "fps": manifest["fps"], "files": manifest["files"],
        "manifest_sha256": args.manifest_sha256,
        "output": str(args.output or args.source), "seconds": time.monotonic() - started,
        "paper_filters_applied": False,
    }
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print("BONES_READBACK_PASS " + json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
