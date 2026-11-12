"""Verify and stage the nine packed motion files without changing their bytes."""

import argparse
import hashlib
import json
import tarfile
import time
from pathlib import Path, PurePosixPath


FIELDS = {
    "joint_names.json", "motion_names.json", "motion_ends.npy", "joint_pos.npy",
    "joint_vel.npy", "root_pos_w.npy", "root_quat_w.npy", "root_lin_vel_w.npy",
    "root_ang_vel_w.npy",
}


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    assert sha256(args.manifest) == args.manifest_sha256
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = {row["file"]: row for row in manifest["files"]}
    assert set(records) == FIELDS and len(manifest["files"]) == len(FIELDS)
    print("Verifying archive SHA256", flush=True)
    assert sha256(args.archive) == args.archive_sha256
    print("Archive SHA256 passed", flush=True)
    output = args.output.resolve()
    partial = output.with_name(output.name + ".partial")
    if output.exists():
        raise FileExistsError(output)
    partial.mkdir(parents=True, exist_ok=True)
    seen = set()
    manifest_member_seen = False
    with tarfile.open(args.archive, "r|gz") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            assert not path.is_absolute() and ".." not in path.parts
            if member.isdir():
                continue
            name = path.name
            # The complete transfer archive also carries its manifest outside
            # the packed directory. Verify it against the expected manifest hash.
            if member.name == "packed-raw-verification.json":
                assert member.isfile() and not manifest_member_seen
                assert member.size == args.manifest.stat().st_size
                source = archive.extractfile(member)
                assert source is not None
                with source:
                    assert hashlib.sha256(source.read()).hexdigest() == args.manifest_sha256
                manifest_member_seen = True
                print("Embedded manifest SHA256 passed", flush=True)
                continue
            assert member.isfile() and name in FIELDS and name not in seen
            seen.add(name)
            record = records[name]
            assert member.size == record["bytes"]
            target = partial / name
            if target.is_file() and target.stat().st_size == member.size:
                if sha256(target) == record["sha256"]:
                    print("Reused verified " + name, flush=True)
                    continue
            temporary = partial / (name + ".writing")
            digest = hashlib.sha256()
            source = archive.extractfile(member)
            assert source is not None
            with source, temporary.open("wb") as destination:
                while block := source.read(8 * 1024 * 1024):
                    destination.write(block)
                    digest.update(block)
            assert temporary.stat().st_size == member.size
            assert digest.hexdigest() == record["sha256"]
            temporary.replace(target)
            print("Extracted " + name, flush=True)
    assert seen == FIELDS
    # Read the persisted files again, independently of the extraction buffers.
    for name, record in records.items():
        assert sha256(partial / name) == record["sha256"]
    partial.rename(output)
    receipt = {
        "status": "cloud_packed_files_verified", "clips": manifest["clips"],
        "frames": manifest["frames"], "fps": manifest["fps"],
        "archive_sha256": args.archive_sha256, "files": manifest["files"],
        "seconds": round(time.monotonic() - started, 1),
        "output": str(output), "paper_filters_applied": False,
    }
    output.with_name(output.name + "-verification.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print("BONES_CLOUD_STAGE_PASS " + json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
