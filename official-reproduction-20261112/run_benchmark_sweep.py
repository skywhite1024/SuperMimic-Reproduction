"""Run every declared throughput size in a separate bounded Linux process.

Requires the completed short GPU integration of the identical worker. Writes
the complete case ledger before starting, then preserves each exit, missing
receipt and failed validation. A finished sweep may contain failed sizes.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time


SIZES = (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)
BINDINGS = ("worker_sha256", "run_result_sha256", "resolved_config_sha256",
            "motion_protocol_sha256", "checkpoints", "packed_files", "dimensions",
            "gpu", "torch", "torch_cuda", "control_dt", "physics_dt", "decimation",
            "cuda_matmul_fp32_precision", "tf32_matmul_allowed", "goal_assembly_backend")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def validate_measurement(record, *, size, purpose):
    warmup, steps, repeats = (2, 2, 1) if purpose == "integration" else (20, 500, 3)
    if (record["status"] != "benchmark_case_completed" or record["purpose"] != purpose
            or record["num_envs"] != size
            or (record["warmup_calls_per_path"], record["timed_calls_per_repeat"], record["repeats"])
                != (warmup, steps, repeats)):
        raise ValueError("Incomplete or wrong-purpose benchmark case")
    if (record["paper_speedup_reproduced"] or record["policy_quality_evaluated"]
            or record["differentiation_speed_or_full_control_speedup_claimed"]
            or not record["simulator_with_policy"]["finite_pre_and_post"]
            or not record["tf32_matmul_allowed"] or record["decimation"] != 4
            or record["control_dt"] != .02 or record["physics_dt"] != .005):
        raise ValueError("Invalid timing scope, finite-state or physics metadata")
    modes = record["world_model_modes"]
    if ([mode["inference_mode"] for mode in modes] != [False, True]
            or [mode["autograd_enabled"] for mode in modes] != [True, False]):
        raise ValueError("Both explicit WM gradient modes are required")
    for path in [record["simulator_with_policy"], *modes]:
        durations, rates = path["seconds_per_repeat"], path["steps_per_second"]
        if len(durations) != repeats or len(rates) != repeats:
            raise ValueError("Missing timing repeat")
        if any(not math.isfinite(value) or value <= 0 for value in durations + rates):
            raise ValueError("Nonpositive/nonfinite timing is not a valid throughput")
        if any(not math.isclose(rate, steps * size / duration, rel_tol=1e-12)
               for rate, duration in zip(rates, durations, strict=True)):
            raise ValueError("Throughput does not recompute from duration and control steps")
    sim = statistics.median(record["simulator_with_policy"]["steps_per_second"])
    if not math.isclose(record["simulator_median_sps"], sim, rel_tol=1e-12):
        raise ValueError("Incorrect simulator median")
    for mode in modes:
        ratio = statistics.median(mode["steps_per_second"]) / sim
        if not math.isclose(record["diagnostic_wm_to_sim_scope_ratios"][str(mode["inference_mode"])],
                            ratio, rel_tol=1e-12):
            raise ValueError("Incorrect unequal-scope diagnostic ratio")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("integration", "run", "motions", "motion-protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != "linux":
        raise RuntimeError("The production sweep uses Linux GNU timeout process supervision")
    worker = Path(__file__).with_name("benchmark_local_checkpoint.py")
    integration = json.loads(args.integration.read_bytes())
    validate_measurement(integration, size=64, purpose="integration")
    if integration["worker_sha256"] != sha(worker):
        raise ValueError("The actual GPU integration used a different benchmark worker")
    if args.output.exists():
        raise FileExistsError("A fresh finite sweep folder is required")
    args.output.mkdir(parents=True)
    state = {
        "status": "benchmark_sweep_running", "coordinator_sha256": sha(Path(__file__)),
        "integration": {"path": str(args.integration), "sha256": sha(args.integration)},
        "worker_sha256": sha(worker), "declared_sizes": list(SIZES),
        "per_case_timeout_seconds": 900, "kill_after_seconds": 30,
        "timing": {"warmup_per_path": 20, "steps_per_repeat": 500, "repeats": 3},
        "comparison_scope": integration["comparison_scope"],
        "paper_speedup_reproduced": False, "policy_quality_evaluated": False,
        "all_sizes_valid": False,
        "cases": [{"num_envs": size, "status": "not_started"} for size in SIZES],
    }
    summary_path = args.output / "summary.json"
    save(summary_path, state)
    for case in state["cases"]:
        size = case["num_envs"]
        folder = args.output / f"batch-{size}"
        log = args.output / f"batch-{size}.log"
        case.update(status="running", result_path=str(folder / "result.json"), log_path=str(log))
        save(summary_path, state)
        command = ["timeout", "--signal=TERM", "--kill-after=30s", "900s", sys.executable,
                   str(worker), "--run", str(args.run), "--motions", str(args.motions),
                   "--motion-protocol", str(args.motion_protocol), "--num-envs", str(size),
                   "--output", str(folder), "--purpose", "experiment",
                   "--warmup", "20", "--steps", "500", "--repeats", "3"]
        started = time.monotonic()
        with log.open("wb") as stream:
            rc = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False).returncode
        case.update(exit_code=rc, wall_seconds=time.monotonic() - started,
                    log_bytes=log.stat().st_size, log_sha256=sha(log))
        result = folder / "result.json"
        try:
            if result.exists():
                case.update(result_bytes=result.stat().st_size, result_sha256=sha(result))
                record = json.loads(result.read_bytes())
                case["worker_status"] = record.get("status")
            else:
                raise ValueError("Worker result receipt is missing")
            if rc != 0:
                raise ValueError(f"Worker exited with status {rc}; not a valid measurement")
            validate_measurement(record, size=size, purpose="experiment")
            if any(record[key] != integration[key] for key in BINDINGS):
                raise ValueError("Model/data/runtime metadata differs from the completed integration")
            case.update(status="valid_measurement", simulator_median_sps=record["simulator_median_sps"],
                        world_model_modes=record["world_model_modes"],
                        diagnostic_unequal_scope_ratios=record["diagnostic_wm_to_sim_scope_ratios"])
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            case.update(status="failed_or_incomplete", error=f"{type(error).__name__}: {error}")
        save(summary_path, state)
        print("LOCAL_BENCHMARK_SWEEP_CASE " + json.dumps(case, separators=(",", ":")), flush=True)
    valid = sum(case["status"] == "valid_measurement" for case in state["cases"])
    state.update(status="benchmark_sweep_completed" if valid == len(SIZES)
                 else "benchmark_sweep_finished_with_failed_sizes",
                 all_sizes_attempted=True, all_sizes_valid=valid == len(SIZES),
                 valid_sizes=valid, failed_sizes=len(SIZES) - valid)
    save(summary_path, state)
    print("LOCAL_BENCHMARK_SWEEP_FINISHED " + json.dumps({
        "status": state["status"], "valid_sizes": valid, "failed_sizes": len(SIZES) - valid}), flush=True)


if __name__ == "__main__":
    main()
