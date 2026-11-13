"""Select verified native weights from an actually completed parent training run."""

import hashlib
import json
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def select_completed_checkpoint(run, collection_step=None):
    run = Path(run).resolve()
    result_path = run / "result.json"
    training = json.loads(result_path.read_bytes())
    if training["status"] != "pilot_completed":
        raise ValueError("An actually completed parent training run is required")
    selection = {"parent_result_sha256": digest(result_path),
                 "parent_collection_steps": training["collection_steps"],
                 "evaluated_collection_steps": training["collection_steps"],
                 "kind": "final_models", "files": []}
    if collection_step is None:
        records = training["checkpoints"]
        paths = [Path(r["path"]) for r in records]
    else:
        if not 1 <= collection_step <= training["collection_steps"]:
            raise ValueError("Snapshot step outside completed parent run")
        rows = [r for r in training.get("retained_checkpoint_series", [])
                if r["collection_steps"] == collection_step]
        if len(rows) != 1:
            raise ValueError("Exactly one retained snapshot is required")
        folder = run / "checkpoint-series" / f"step-{collection_step:09d}"
        manifest_path = folder / "manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        core_fields = ("status", "collection_steps", "files", "full_training_resume_available", "scope")
        if any(manifest[key] != rows[0][key] for key in core_fields):
            raise ValueError("Snapshot manifest differs from parent receipt")
        if manifest["status"] != "model_snapshot_preserved":
            raise ValueError("Snapshot was not completely preserved")
        records = manifest["files"]
        paths = [folder / r["file"] for r in records]
        selection.update(kind="retained_snapshot", evaluated_collection_steps=collection_step,
                         manifest_sha256=digest(manifest_path))
    required = {"world_model.pt", "actor.pt", "critic.pt"}
    if len(paths) != 3 or {p.name for p in paths} != required:
        raise ValueError("Exactly the three native models are required")
    checkpoints = {}
    for path, record in zip(paths, records, strict=True):
        if collection_step is not None and record["file"] != path.name:
            raise ValueError("Snapshot filenames must be basenames")
        if path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
            raise ValueError("Checkpoint bytes or hash mismatch: " + path.name)
        checkpoints[path.name] = path
        selection["files"].append({"path": str(path), "bytes": record["bytes"],
                                   "sha256": record["sha256"]})
    return training, checkpoints, selection
