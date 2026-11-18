"""Evaluate the released SONIC actor on the full feasible BONES proxy."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from push_recovery_contract import sha256, write_json
from run_lafan_learning_curve import execute
from run_sonic_bones_cuda_gate import ORT_VERSION, VARIANT_RELEASE_PATHS, WHEEL_HASH


CLIPS = 129543
FRAMES = 47374271
SCRIPTS = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_bytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--membership-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, gate, output = args.root.resolve(), args.gate.resolve(), args.output.resolve()
    driver_file = gate / "driver.json"
    gate_result = gate / "evaluation/result.json"
    gate_driver = read(driver_file)
    gate_eval = read(gate_result)
    variant = gate_driver.get("SONIC_variant", "default")
    if (gate_driver["status"] != "completed" or gate_driver["ONNX_runtime_version"] != ORT_VERSION
            or gate_driver["wheel_sha256"] != WHEEL_HASH or gate_driver["result_sha256"] != sha256(gate_result)
            or not gate_driver["original_comparison_and_cuda_kernels_verified"]
            or variant not in VARIANT_RELEASE_PATHS or gate_eval.get("SONIC_variant", "default") != variant
            or gate_eval["ONNX_provider"] != "CUDAExecutionProvider"):
        raise ValueError("A completed and verified SONIC CUDA gate is required")
    import onnxruntime as ort

    if ort.__version__ != ORT_VERSION or "CUDAExecutionProvider" not in ort.get_available_providers():
        raise ValueError("Use the verified CUDA inference environment from the common64 gate")
    motions = root / "runs/bones-duration-1118c/motions/full"
    manifest_file = motions.with_name("full-duration-pack.json")
    pack = read(manifest_file)
    if (pack["case"], pack["clips"], pack["frames"], pack["membership_sha256"]) != (
            "full", CLIPS, FRAMES, args.membership_sha256):
        raise ValueError("The verified complete BONES proxy pack is required")
    if {path.name: sha256(path) for path in sorted(motions.iterdir()) if path.is_file()} != pack["files"]:
        raise ValueError("The packed motion bytes changed")
    output.mkdir(parents=True, exist_ok=False)
    evaluation = output / "evaluation"
    result_file = evaluation / "result.json"
    protocol_file = evaluation / "protocol.json"
    accounting_file = evaluation / "accounting.json"
    state_file = evaluation / "audit/initial-states.json"
    trajectory_file = evaluation / "audit/summary.json"
    report_file = output / "result.json"
    report = {"status": "evaluating", "clips": CLIPS, "reference_frames": FRAMES,
              "num_envs": 1024, "membership_sha256": args.membership_sha256,
              "gate_driver_sha256": sha256(driver_file), "gate_result_sha256": sha256(gate_result),
              "original_SONIC_revision": gate_driver["original_SONIC_revision"], "SONIC_variant": variant,
              "author_exact_membership": False}
    write_json(report_file, report)
    try:
        batched = gate / "models-batched" if variant == "v1.1" else root / "runs/sonic-lafan40-1116a/models-batched"
        execute("evaluate_sonic_corpus_v2.py", ["--models", root / "data" / VARIANT_RELEASE_PATHS[variant],
            "--batched", batched, "--sonic-variant", variant,
            "--contract", SCRIPTS / "sonic-g1-control-contract.json", "--motions", motions,
            "--output", evaluation, "--expected-clips", CLIPS, "--num-envs", 1024,
            "--onnx-provider", "cuda", "--order", "length", "--seed", 42],
            output / "evaluation.log", 21600, stream_progress=("BATCH_COMPLETED ", "OFFICIAL_CORPUS_EVAL_PASS "))
        execute("audit_corpus_result.py", ["--result", result_file, "--protocol", protocol_file,
            "--motions", motions, "--expected-clips", CLIPS, "--require-trajectories",
            "--output", accounting_file], output / "accounting.log", 1500)
        execute("audit_saved_initial_states_v2.py", ["--evaluation", evaluation, "--motions", motions,
            "--expected-clips", CLIPS, "--output", evaluation / "audit"], output / "audit.log", 3300)
        protocol, counts, states = map(read, (protocol_file, accounting_file, state_file))
        result = read(result_file)
        if (protocol["motion_files"] != pack["files"] or counts["clips"] != CLIPS
                or counts["reference_frames"] != FRAMES or not states["all_saved_initial_states_verified"]
                or counts["result_sha256"] != sha256(result_file)
                or result["SONIC_variant"] != variant or protocol["SONIC_variant"] != variant
                or result["ONNX_provider"] != "CUDAExecutionProvider"):
            raise RuntimeError("The full evaluation failed its membership or artifact checks")
        report.update(status="complete", successes=counts["successes"], failures=counts["failures"],
                      mean_normalized_return=counts["mean_normalized_full_motion_return"],
                      evaluation_seconds=result["elapsed_seconds"], per_motion_result_sha256=counts["result_sha256"],
                      profiles=result["ONNX_runtime_compute_profiles"])
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(report_file, report)
        files = (manifest_file, report_file, protocol_file, accounting_file, state_file, trajectory_file)
        # The large per-motion original is retained on AFS for an independent readback.
        subprocess.run([sys.executable, str(SCRIPTS / "emit_json_receipts.py"), "--max-bytes", "8000000",
                        *map(str, (path for path in files if path.is_file()))], check=True, timeout=180)
    print("SONIC_BONES_FULL_COMPLETE " + json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
