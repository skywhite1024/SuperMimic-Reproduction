"""Verify actual SONIC CUDA inference on the fixed 64-motion BONES probe."""

import argparse
import http.client
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import urllib.request

from prepare_sonic_onnx_batch import VARIANT_HASHES
from push_recovery_contract import sha256, write_json
from run_lafan_learning_curve import execute

ORT_VERSION = "1.26.0"
# PyPI's 1.26 release targets CUDA 12.8 and cuDNN 9. The SenseCore worker
# image provides CUDA 12.9, and the pinned torch build is from CUDA 12.x.
WHEEL = f"onnxruntime_gpu-{ORT_VERSION}-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
WHEEL_URL = ("https://files.pythonhosted.org/packages/dd/97/"
             "fe8979f44b9275654b42f7bb556e30789b71a1b22998c83b540df2b1b774/" + WHEEL)
WHEEL_MIRROR = WHEEL_URL.replace("https://files.pythonhosted.org/", "https://mirrors.aliyun.com/pypi/")
WHEEL_HASH = "cfda2fad535595bfc3e570eb588092717711dcb2957656d814695e0c9ceb1508"
WHEEL_BYTES = 276974871
FLAT = "flatbuffers-25.12.19-py2.py3-none-any.whl"
FLAT_HASH = "7634f50c427838bb021c2d66a3d1168e9d199b0607e6329399f04846d42e20b4"
SONIC_REVISION = "6733128a3d8a523b1418b06bca3cdf61c8b0987f"
VARIANT_RELEASE_PATHS = {"default": "sonic-default-6733128", "v1.1": "sonic-v1_1-6733128"}
V11_SIZES = {"model_encoder.onnx": 50117796, "model_decoder.onnx": 149778638,
             "observation_config.yaml": 2475}


def fetch_v11_release(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for name, expected_hash in VARIANT_HASHES["v1.1"].items():
        path, expected_size = directory / name, V11_SIZES[name]
        if path.is_file():
            if path.stat().st_size != expected_size or sha256(path) != expected_hash:
                raise ValueError(f"Cached SONIC v1.1 artifact differs from the pinned release: {name}")
            print(f"SONIC_V11_MODEL_CACHE_VERIFIED name={name} bytes={expected_size}", flush=True)
            continue
        for endpoint in ("https://huggingface.co", "https://hf-mirror.com"):
            url = f"{endpoint}/nvidia/GEAR-SONIC/resolve/{SONIC_REVISION}/sonic_v1_1/{name}"
            part = None
            started = time.monotonic()
            print(f"SONIC_V11_TRANSFER_START name={name} source={endpoint}", flush=True)
            try:
                with tempfile.NamedTemporaryFile(dir=directory, prefix=".sonic-v11-", delete=False) as target:
                    part = Path(target.name)
                    request = urllib.request.Request(url, headers={"User-Agent": "SuperMimic-Reproduction/1.0"})
                    with urllib.request.urlopen(request, timeout=35) as remote:
                        size = 0
                        while chunk := remote.read(1 << 20):
                            size += len(chunk)
                            if size > expected_size or time.monotonic() - started > 420:
                                raise ValueError("SONIC v1.1 transfer exceeded pinned size or time limit")
                            target.write(chunk)
                if size != expected_size or sha256(part) != expected_hash:
                    raise ValueError("SONIC v1.1 transfer differs from the pinned upstream bytes")
                part.replace(path)
                print(f"SONIC_V11_TRANSFER_VERIFIED name={name} source={endpoint} bytes={size}", flush=True)
                break
            except (OSError, ValueError, http.client.HTTPException) as error:
                print(f"SONIC_V11_TRANSFER_FAILED name={name} source={endpoint} error={type(error).__name__}",
                      flush=True)
            finally:
                if part is not None:
                    part.unlink(missing_ok=True)
        else:
            raise RuntimeError(f"Public SONIC v1.1 transfer failed: {name}")


def fetch_wheel(wheel):
    chunk_size = 1 << 20
    for source, url in (("aliyun", WHEEL_MIRROR), ("pypi", WHEEL_URL)):
        part = None
        started = time.monotonic()
        print(f"SONIC_ORT_WHEEL_TRANSFER_START source={source} total_bytes={WHEEL_BYTES}", flush=True)
        try:
            with tempfile.NamedTemporaryFile(dir=wheel.parent, prefix=".ort-gpu-", delete=False) as target:
                part = Path(target.name)
                with urllib.request.urlopen(url, timeout=25) as remote:
                    size = 0
                    while chunk := remote.read(chunk_size):
                        if time.monotonic() - started > 300:
                            raise TimeoutError("ONNX Runtime wheel exceeded the five-minute transfer window")
                        size += len(chunk)
                        if size > WHEEL_BYTES:
                            raise ValueError("ONNX Runtime wheel exceeds the manifest size")
                        target.write(chunk)
                        if size % (32 * chunk_size) == 0:
                            print(f"SONIC_ORT_WHEEL_TRANSFER_PROGRESS source={source} bytes={size}", flush=True)
            if size != WHEEL_BYTES or sha256(part) != WHEEL_HASH:
                raise ValueError("ONNX Runtime GPU wheel digest differs from PyPI")
            part.replace(wheel)
            print(f"SONIC_ORT_WHEEL_TRANSFER_VERIFIED source={source} bytes={size}", flush=True)
            return
        except (OSError, ValueError, http.client.HTTPException) as error:
            print(f"SONIC_ORT_WHEEL_TRANSFER_FAILED source={source} error={type(error).__name__}", flush=True)
        finally:
            if part is not None:
                part.unlink(missing_ok=True)
    raise RuntimeError("Both ONNX Runtime wheel transports failed")


def install(root, output):
    wheel_cache = root / "data/sonic-inference-wheels"
    wheel = wheel_cache / WHEEL
    if not wheel.is_file():
        fetch_wheel(wheel)
    flat = wheel_cache / FLAT
    if (wheel.stat().st_size != WHEEL_BYTES or sha256(wheel) != WHEEL_HASH
            or sha256(flat) != FLAT_HASH):
        raise RuntimeError("An inference wheel failed verification")
    print("SONIC_ORT_INPUT_WHEELS_VERIFIED", flush=True)
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
    parser.add_argument("--variant", choices=VARIANT_HASHES, default="default")
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parent
    motions = root / "runs/bones-duration-1118c/motions/common64"
    original = root / "data" / VARIANT_RELEASE_PATHS[args.variant]
    batched = (output / "models-batched" if args.variant == "v1.1"
               else root / "runs/sonic-lafan40-1116a/models-batched")
    manifest = json.loads((motions.with_name(motions.name + "-duration-pack.json")).read_bytes())
    files = {p.name: sha256(p) for p in sorted(motions.iterdir()) if p.is_file()}
    if manifest["case"] != "common64" or manifest["files"] != files or manifest["frames"] != 20397:
        raise RuntimeError("Unexpected common64 membership")
    report = {"status": "started", "wheel": WHEEL, "wheel_sha256": WHEEL_HASH,
              "ONNX_runtime_version": ORT_VERSION, "original_SONIC_revision": SONIC_REVISION,
              "SONIC_variant": args.variant,
              "script_commit_scope": "CUDA inference gate on common64", "paper_benchmark_equivalence": False,
              "motion_files": files}
    write_json(output / "driver.json", report)
    ev = output / "evaluation"
    try:
        if args.variant == "v1.1":
            fetch_v11_release(original)
            execute("prepare_sonic_onnx_batch.py",
                    ["--models", original, "--output", batched, "--variant", args.variant],
                    output / "batch-derivation.log", 600,
                    python=root / "conda/sonic-inference-cpu/bin/python")
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
                    "--num-envs", "64", "--onnx-provider", "cuda", "--order", "index",
                    "--sonic-variant", args.variant]
        with (output / "evaluation.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(["timeout", "--signal=TERM", "--kill-after=30s", "1100s", *evaluate],
                                    stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            print((output / "evaluation.log").read_text(encoding="utf-8", errors="replace")[-12000:],
                  flush=True)
            raise RuntimeError("SONIC CUDA evaluation failed")
        result = json.loads((ev / "result.json").read_bytes())
        assert result["ONNX_provider"] == "CUDAExecutionProvider"
        assert result["SONIC_variant"] == args.variant
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
        if args.variant == "v1.1":
            names.append(batched / "validation.json")
        names.extend(ev / name for name in ("result.json", "protocol.json", "accounting.json",
                     "audit/summary.json", "audit/initial-states.json"))
        subprocess.run([sys.executable, str(source / "emit_json_receipts.py"),
                        *map(str, (path for path in names if path.is_file()))], check=True, timeout=60)


if __name__ == "__main__":
    main()
