"""Record imports and a finite CUDA computation without reading credentials."""
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import mjlab
import mujoco
import supermimic
import torch
import warp

result = {
    "observed_at": datetime.now(timezone.utc).isoformat(),
    "python": platform.python_version(),
    "platform": platform.platform(),
    "packages": {
        name: importlib.metadata.version(name)
        for name in ("torch", "mujoco", "mujoco-warp", "warp-lang", "mjlab", "supermimic", "rsl-rl-lib")
    },
    "source": {"mjlab": str(mjlab.__file__), "supermimic": str(supermimic.__file__)},
    "cuda_available": torch.cuda.is_available(),
    "cuda_build": torch.version.cuda,
    "devices": [],
    "does_not_verify_training_or_paper_results": True,
}
if result["cuda_available"]:
    for index in range(torch.cuda.device_count()):
        x = torch.arange(1024, device=f"cuda:{index}", dtype=torch.float32)
        assert (x * 2).sum().item() == 1047552.0
        result["devices"].append({"index": index, "name": torch.cuda.get_device_name(index)})
output = Path(sys.argv[1])
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
print(json.dumps(result, indent=2))
