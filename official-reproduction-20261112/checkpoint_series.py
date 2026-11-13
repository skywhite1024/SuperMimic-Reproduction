"""Keep immutable model snapshots for finite long runs, without claiming resume."""

import hashlib
import json
from pathlib import Path
import shutil


def preserve_checkpoint(files, output, collection_steps, *, without_critic=False):
    """Copy the three native model files before the next native save overwrites them.

    Called by the runner's single checkpoint worker. The runner waits for that
    worker before overwriting source files. A manifest is written last; an
    interrupted copy remains present but is not a completed snapshot.
    """
    import torch

    files = [Path(path) for path in files]
    required = {"world_model.pt", "actor.pt"} if without_critic else {"world_model.pt", "actor.pt", "critic.pt"}
    models = [path for path in files if path.name in required]
    if len(models) != len(required) or {path.name for path in models} != required:
        raise ValueError("A complete declared native model set is required")
    if without_critic and any(p.name == "critic.pt" for p in files):
        raise ValueError("No-critic snapshots must not contain a critic")
    if collection_steps < 1:
        raise ValueError("A positive completed collection-step count is required")
    destination = Path(output) / f"step-{collection_steps:09d}"
    destination.mkdir(parents=True, exist_ok=False)

    def check_finite(value):
        if isinstance(value, torch.Tensor):
            if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all().item():
                raise ValueError("Nonfinite intermediate checkpoint")
        elif isinstance(value, dict):
            for child in value.values():
                check_finite(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                check_finite(child)

    receipts = []
    for source in models:
        target = destination / source.name
        # Hard links are unsafe: the native next save truncates its source inode.
        shutil.copyfile(source, target)
        check_finite(torch.load(target, map_location="cpu", weights_only=True))
        with target.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        receipts.append({"file": target.name, "bytes": target.stat().st_size, "sha256": digest})
    receipt = {"status": "model_snapshot_preserved", "collection_steps": collection_steps,
        "files": receipts, "full_training_resume_available": False,
        "scope": "Native model weights and registered buffers only; excludes optimizer, replay, sampler, simulator and RNG state"}
    (destination / "manifest.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8", newline="\n")
    return receipt
