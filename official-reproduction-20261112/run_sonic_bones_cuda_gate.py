"""Verify actual SONIC CUDA inference on the fixed 64-motion BONES probe."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import urllib.request

from push_recovery_contract import sha256, write_json
from run_lafan_learning_curve import execute

WHEEL = "onnxruntime_gpu-1.32.0-cp313-cp313-manylinux_2_28_x86_64.whl"
WHEEL_URL = ("https://files.pythonhosted.org/packages/a9/b6/"
             "759cc7838e5640d30e365e3b8466987172526faabf1779f0bf776f566b1e/" + WHEEL)
WHEEL_HASH = "d3cec70e7a7dcfb085e0e6458210211541a33ffbf89c6a6e4e3d6adf7262cd6d"
WHEEL_BYTES = 251389345
FLAT = "flatbuffers-25.12.19-py2.py3-none-any.whl"
FLAT_HASH = "7634f50c427838bb021c2d66a3d1168e9d199b0607e6329399f04846d42e20b4"


def install(root, output):
    wheel_cache = root / "data/sonic-inference-wheels"
    wheel = wheel_cache / WHEEL
    if not wheel.is_file():
        with tempfile.NamedTemporaryFile(dir=wheel_cache, prefix=".ort-gpu-", delete=False) as target:
            part = Path(target.name)
            with urllib.request.urlopen(WHEEL_URL, timeout=45) as remote:
                size = 0
                while chunk := remote.read(1 << 20):
                    size += len(chunk)
                    if size > WHEEL_BYTES:
                        raise RuntimeError("ONNX Runtime transfer exceeds the manifest size")
                    target.write(chunk)
        if part.stat().st_size != WHEEL_BYTES or sha256(part) != WHEEL_HASH:
            raise RuntimeError("ONNX Runtime GPU wheel digest does not match PyPI")
        part.replace(wheel)
    flat = wheel_cache / FLAT
    if (wheel.stat().st_size != WHEEL_BYTES or sha256(wheel) != WHEEL_HASH
            or sha256(flat) != FLAT_HASH):
        raise RuntimeError("An inference wheel failed verification")
    env = output / "onnx-gpu-venv"
    subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", str(env)],
                   check=True, timeout=180)
    python = env / "bin/python"
    subprocess.run([str(python), "-m", "pip", "install", "--disable-pip-version-check",
                    "--no-index", "--no-deps", str(flat), str(wheel)], check=True, timeout=300)
    return python


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parent
    motions = root / "runs/bones-duration-1118c/motions/common64"
    original = root / "data/sonic-default-6733128"
    batched = root / "runs/sonic-lafan40-1116a/models-batched"
    manifest = json.loads((motions.with_name(motions.name + "-duration-pack.json")).read_bytes())
    files = {p.name: sha256(p) for p in sorted(motions.iterdir()) if p.is_file()}
    if manifest["case"] != "common64" or manifest["files"] != files or manifest["frames"] != 20397:
        raise RuntimeError("Unexpected common64 membership")
    report = {"status": "started", "wheel": WHEEL, "wheel_sha256": WHEEL_HASH,
              "ONNX_runtime_version": "1.32.0", "original_SONIC_revision": "6733128a3d8a523b1418b06bca3cdf61c8b0987f",
              "script_commit_scope": "CUDA inference gate on common64", "paper_benchmark_equivalence": False,
              "motion_files": files}
    write_json(output / "driver.json", report)
    ev = output / "evaluation"
    try:
        python = install(root, output)
        # Keep this environment scoped to the gate. Use the same compiled model
        # and native audited evaluation code with the CUDA inference option.
        os.environ.update(PYTHONNOUSERSITE="1", WANDB_MODE="offline", MUJOCO_GL="disable")
        probe = (
            "import json, torch, onnxruntime as ort;"
            "assert torch.cuda.device_count()==1;"
            "print('SONIC_ORT_ENV '+json.dumps(dict(torch=str(torch.__version__),"
            "cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),"
            "ort=ort.__version__,providers=ort.get_available_providers())),flush=True);"
            "assert 'CUDAExecutionProvider' in ort.get_available_providers()"
        )
        subprocess.run([str(python), "-c", probe], check=True, timeout=120)
        evaluate = [str(python), str(source / "evaluate_sonic_corpus_v2.py"),
                    "--models", str(original), "--batched", str(batched),
                    "--contract", str(source / "sonic-g1-control-contract.json"),
                    "--motions", str(motions), "--output", str(ev), "--expected-clips", "64",
                    "--num-envs", "64", "--onnx-provider", "cuda", "--order", "index"]
        with (output / "evaluation.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(["timeout", "--signal=TERM", "--kill-after=30s", "1100s", *evaluate],
                                    stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            print((output / "evaluation.log").read_text(encoding="utf-8", errors="replace")[-12000:],
                  flush=True)
            raise RuntimeError("SONIC CUDA evaluation failed")
        result = json.loads((ev / "result.json").read_bytes())
        assert result["ONNX_provider"] == "CUDAExecutionProvider"
        for graph in ("encoder", "decoder"):
            assert result["ONNX_runtime_compute_profiles"][graph]["cuda_compute_ops"]
        execute("audit_corpus_result.py", ["--result", ev / "result.json", "--protocol", ev / "protocol.json",
            "--motions", motions, "--expected-clips", 64, "--require-trajectories",
            "--output", ev / "accounting.json"], output / "accounting.log", 120)
        execute("audit_saved_initial_states_v2.py", ["--evaluation", ev, "--motions", motions,
            "--expected-clips", 64, "--output", ev / "audit"], output / "audit.log", 120)
        report.update(status="completed", original_comparison_and_cuda_kernels_verified=True,
                      result_sha256=sha256(ev / "result.json"))
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "driver.json", report)
        names = [output / "driver.json"]
        names.extend(ev / name for name in ("result.json", "protocol.json", "accounting.json",
                     "audit/summary.json", "audit/initial-states.json"))
        subprocess.run([sys.executable, str(source / "emit_json_receipts.py"),
                        *map(str, (path for path in names if path.is_file()))], check=True, timeout=60)


if __name__ == "__main__":
    main()
