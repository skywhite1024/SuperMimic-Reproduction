"""Extract only raw BONES CSV motions from a previously hash-verified archive."""

import argparse
import hashlib
import json
import shutil
import tarfile
import time
from collections import Counter
from pathlib import Path, PurePosixPath

from download_bones import SHA256, SIZE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    archive = args.archive.resolve()
    receipt = json.loads((archive.parent / "range-progress.json").read_text())
    if receipt["status"] != "verified" or receipt["expected_sha256"] != SHA256:
        raise RuntimeError("VerifiedArchiveReceiptRequired")
    if archive.stat().st_size != SIZE:
        raise RuntimeError("ArchiveSizeChanged")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    count, byte_count = 0, 0
    other_types = Counter()
    start = time.monotonic()

    def record(status):
        state = {
            "status": status,
            "archive_sha256": SHA256,
            "csv_files": count,
            "csv_bytes": byte_count,
            "other_archive_files_by_suffix": dict(other_types),
            "elapsed_seconds": round(time.monotonic() - start, 1),
            "name_filter_applied": False,
            "feasibility_filter_applied": False,
        }
        (output / "extraction-status.json").write_text(
            json.dumps(state, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(state), flush=True)

    record("extracting")
    try:
        with (
            (output / "raw-csv-manifest.jsonl").open("x", encoding="utf-8") as manifest,
            tarfile.open(archive, "r|gz") as bundle,
        ):
            for member in bundle:
                name = PurePosixPath(member.name)
                if not member.isfile():
                    if not member.isdir():
                        raise RuntimeError("UnexpectedArchiveMemberType")
                    continue
                if name.parts[:2] != ("g1", "csv") or name.suffix.lower() != ".csv":
                    other_types[name.suffix] += 1
                    continue
                target = output.joinpath(*name.parts).resolve()
                if not target.is_relative_to(output):
                    raise RuntimeError("UnsafeArchivePath")
                if target.exists():
                    raise RuntimeError("DuplicateArchiveMember")
                if count % 1000 == 0 and shutil.disk_usage(output).free < 5 * 1024**3:
                    raise RuntimeError("InsufficientFreeSpace")
                target.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                written = 0
                source = bundle.extractfile(member)
                if source is None:
                    raise RuntimeError("MissingArchiveMemberData")
                with source, target.open("xb") as destination:
                    while block := source.read(1024**2):
                        destination.write(block)
                        digest.update(block)
                        written += len(block)
                if written != member.size:
                    raise RuntimeError("ExtractedMemberSizeMismatch")
                manifest.write(
                    json.dumps(
                        {
                            "path": member.name,
                            "bytes": written,
                            "sha256": digest.hexdigest(),
                        }
                    )
                    + "\n"
                )
                count += 1
                byte_count += written
                if count % 2000 == 0:
                    manifest.flush()
                    record("extracting")
        record("extracted")
    except BaseException:
        record("interrupted")
        raise


if __name__ == "__main__":
    main()
