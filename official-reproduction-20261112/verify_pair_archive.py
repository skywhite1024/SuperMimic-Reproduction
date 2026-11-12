"""Verify and extract the bounded BONES pair evidence archive to a fresh folder.

Checks transfer bytes, every declared archive member, and the six CPU-loaded
model files. This does not verify excluded trajectory files or motion quality.
"""

import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def expected_files():
    files = {f"evaluation-{label}/{name}.json" for label in ("1k", "16k")
             for name in ("result", "protocol", "accounting")}
    for label in ("1k", "16k"):
        files.update(f"training-{label}/{name}.json" for name in ("result", "resolved-config"))
        files.update(f"training-{label}/models/{name}.pt" for name in ("world_model", "actor", "critic"))
    files.update(("comparison/summary.json", "comparison/paired-motions.csv"))
    for label in ("1k", "16k"):
        files.update((f"trajectories-{label}/summary.json", f"trajectories-{label}/files.csv"))
    return files


def verify_archive(archive, receipt, output):
    if receipt["status"] != "archive_created_on_afs":
        raise ValueError("A completed server archive receipt is required")
    if archive.stat().st_size != receipt["bytes"] or digest(archive) != receipt["sha256"]:
        raise ValueError("Downloaded archive differs from server size/SHA256")
    required = expected_files()
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        names = [member.name for member in members]
        if (len(names) != len(set(names)) or set(names) != required | {"package-manifest.json"}
                or len(names) != receipt["members"] or not all(member.isfile() for member in members)):
            raise ValueError("Unexpected, duplicated or non-regular archive member")
        member_map = {member.name: member for member in members}
        if member_map["package-manifest.json"].size > 1024 * 1024:
            raise ValueError("Unexpectedly large package manifest")
        manifest_bytes = bundle.extractfile(member_map["package-manifest.json"]).read()
        manifest = json.loads(manifest_bytes)
        if manifest["status"] != "complete_pair_evidence_ready":
            raise ValueError("Package is not complete")
        entries = {entry["file"]: entry for entry in manifest["files"]}
        if len(entries) != len(manifest["files"]) or set(entries) != required:
            raise ValueError("Manifest membership differs from expected evidence")
        for name, entry in entries.items():
            if member_map[name].size != entry["bytes"] or entry["bytes"] <= 0:
                raise ValueError(f"Invalid member size: {name}")
        output.mkdir(parents=True, exist_ok=False)
        for name, entry in entries.items():
            target = output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".partial")
            checksum = hashlib.sha256()
            written = 0
            with bundle.extractfile(member_map[name]) as source, partial.open("xb") as destination:
                while block := source.read(1024 * 1024):
                    destination.write(block)
                    checksum.update(block)
                    written += len(block)
            if written != entry["bytes"] or checksum.hexdigest() != entry["sha256"]:
                raise ValueError(f"Extracted member checksum failed: {name}")
            partial.rename(target)
        (output / "package-manifest.json").write_bytes(manifest_bytes)

    import torch

    def check_tensors(value):
        if isinstance(value, torch.Tensor):
            if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all().item():
                raise ValueError("Nonfinite model tensor")
            return 1
        if isinstance(value, dict):
            return sum(check_tensors(child) for child in value.values())
        if isinstance(value, (list, tuple)):
            return sum(check_tensors(child) for child in value)
        return 0

    model_checks = []
    for label, steps in (("1k", 1001), ("16k", 16001)):
        training = json.loads((output / f"training-{label}/result.json").read_text())
        if training["status"] != "pilot_completed" or training["collection_steps"] != steps:
            raise ValueError("Unexpected training receipt")
        checkpoints = {Path(item["path"]).name: item for item in training["checkpoints"]}
        if len(checkpoints) != 3 or len(training["checkpoints"]) != 3:
            raise ValueError("Incomplete checkpoint receipt")
        for name in ("world_model", "actor", "critic"):
            relative = f"training-{label}/models/{name}.pt"
            entry = entries[relative]
            original = checkpoints[f"{name}.pt"]
            if any(entry[key] != original[key] for key in ("bytes", "sha256")):
                raise ValueError("Packaged model differs from original training receipt")
            tensor_count = check_tensors(torch.load(output / relative, map_location="cpu", weights_only=True))
            if tensor_count < 1:
                raise ValueError("No model tensors")
            model_checks.append({"file": relative, "tensors": tensor_count, "finite": True,
                                 "sha256": entry["sha256"]})
    result = {"status": "local_pair_archive_verified", "archive_sha256": receipt["sha256"],
              "archive_bytes": receipt["bytes"], "verified_members": len(members),
              "models": model_checks, "excluded_trajectory_arrays_checked_locally": False,
              "paper_results_reproduced": False}
    (output / "local-verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify_archive(args.archive, json.loads(args.receipt.read_text()), args.output)
    print("LOCAL_PAIR_ARCHIVE_PASS " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
