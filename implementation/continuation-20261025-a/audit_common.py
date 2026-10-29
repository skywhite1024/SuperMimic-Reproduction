"""Read-only state checks shared by the controlled continuation entrypoints."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def state_hash(value):
    """Hash tensor contents and structure without serialization/device metadata."""
    import torch
    h = hashlib.sha256()
    def visit(x):
        if isinstance(x, torch.Tensor):
            y = x.detach().cpu().contiguous()
            assert bool(torch.isfinite(y).all()), 'Nonfinite checkpoint tensor'
            h.update(f'tensor:{y.dtype}:{tuple(y.shape)}:'.encode())
            h.update(y.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x, dict):
            h.update(b'dict:')
            for key in sorted(x, key=lambda k: (type(k).__name__, str(k))):
                visit(key); visit(x[key])
        elif isinstance(x, (list, tuple)):
            h.update(f'{type(x).__name__}:{len(x)}:'.encode())
            for part in x: visit(part)
        else:
            h.update(f'{type(x).__name__}:{x!r};'.encode())
    visit(value)
    return h.hexdigest()


def optimizer_hashes(learner):
    return {name: state_hash(getattr(learner, attr).state_dict()) for name, attr in
            [('opt_wm', 'opt_wm'), ('opt_pi', 'opt_pi'), ('opt_v', 'opt_v')]}


def lr_at(iteration, lr=2e-4, decay=250000):
    return lr * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * iteration / decay))) if iteration < decay else lr * 0.03


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)
