"""Select verified native weights from an actually completed parent training run."""

import hashlib
import json
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def select_completed_checkpoint(run, collection_step=None, *, two_rank=False):
    run = Path(run).resolve()
    result_path = run / "result.json"
    training = json.loads(result_path.read_bytes())
    if two_rank:
        if collection_step is not None or run.name != "rank-0":
            raise ValueError("Select the final verified rank-0 weights for two-rank evaluation")
        group_path = run.parent / "result.json"
        driver_path = run.parent.parent / "result.json"
        group = json.loads(group_path.read_bytes())
        driver = json.loads(driver_path.read_bytes())
        if (training["status"] != "two_rank_native_participant_completed"
                or group["status"] != "two_rank_native_training_completed"
                or group["world_size"] != 2 or group["backend"] != "nccl"
                or driver["status"] != "two_GPU_native_gate_completed"
                or driver["parent_result_sha256"] != digest(group_path)
                or len(group["participants"]) != 2 or group["participants"][0] != training
                or training["resolved_config_sha256"] != digest(run / "resolved-config.json")):
            raise ValueError("A completed and bound two-rank parent is required")
        other = group["participants"][1]
        if (training["rank"] != 0 or other["rank"] != 1
                or training["final_model_state_hashes"] != other["final_model_state_hashes"]
                or training["local_collection_steps"] != other["local_collection_steps"]
                or group["global_simulator_steps"] != training["local_simulator_steps"] + other["local_simulator_steps"]):
            raise ValueError("Two-rank weights or actual simulator counts disagree")
        collections = training["local_collection_steps"]
    elif training["status"] != "pilot_completed":
        raise ValueError("An actually completed parent training run is required")
    else:
        collections = training["collection_steps"]
    selection = {"parent_result_sha256": digest(result_path),
                 "parent_collection_steps": collections,
                 "evaluated_collection_steps": collections,
                 "kind": "final_models", "files": []}
    if two_rank:
        selection.update(kind="two_rank_final_models", group_result_sha256=digest(group_path),
                         verified_driver_sha256=digest(driver_path),
                         global_real_simulator_steps=group["global_simulator_steps"])
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
    required = ({"world_model.pt", "actor.pt"} if training.get("critic_enabled") is False
                else {"world_model.pt", "actor.pt", "critic.pt"})
    if len(paths) != len(required) or {p.name for p in paths} != required:
        raise ValueError("Exactly the declared native models are required")
    checkpoints = {}
    for path, record in zip(paths, records, strict=True):
        if two_rank and (path.resolve().parent != run or not record["native_model_readback_verified"]):
            raise ValueError("A verified checkpoint inside the rank-0 folder is required")
        if collection_step is not None and record["file"] != path.name:
            raise ValueError("Snapshot filenames must be basenames")
        if path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
            raise ValueError("Checkpoint bytes or hash mismatch: " + path.name)
        checkpoints[path.name] = path
        selection["files"].append({"path": str(path), "bytes": record["bytes"],
                                   "sha256": record["sha256"]})
    return training, checkpoints, selection
