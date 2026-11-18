"""Run a measured PPO-Sim clock study with fixed LAFAN40 evaluation."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from ppo_sim_checkpoint import digest, select_ppo_sim_checkpoint
from run_lafan_learning_curve import execute
from run_ppo_sim_learning_curve import evaluate, write

SCRIPTS = Path(__file__).resolve().parent
HOUR = 3600
HOURS = (1, 2, 4, 8, 12, 16, 18)


def verify_distributed_ranks(root, leader_result, world_size):
    statuses = []
    for rank in range(world_size):
        folder = root / f"rank-{rank}"
        rank_result = leader_result if rank == 0 else json.loads((folder / "result.json").read_bytes())
        config_path = folder / "resolved-config.json"
        cfg = json.loads(config_path.read_bytes())
        assert rank_result["status"] == leader_result["status"]
        assert rank_result["source_config_sha256"] == digest(config_path)
        assert cfg["rank"] == rank_result["rank"] == rank and cfg["rank_seed"] == 42 + rank
        assert rank_result["world_size"] == world_size == cfg["world_size"]
        assert rank_result["backend"] == "nccl" and rank_result["local_cuda_device"] == f"cuda:{rank}"
        assert rank_result["gpu"] == "NVIDIA H100 80GB HBM3"
        assert rank_result["ppo_update_calls"] == leader_result["ppo_update_calls"]
        assert rank_result["simulator_steps"] == leader_result["simulator_steps"]
        for key in ("initial_model_state_hashes", "final_model_state_hashes", "shared_snapshots"):
            assert rank_result[key] == leader_result[key]
        statuses.append({"rank": rank, "local_device": rank_result["local_cuda_device"],
                         "local_real_transitions": rank_result["simulator_steps"],
                         "initial_observation_sha256": rank_result["initial_real_observation_sha256"],
                         "result_sha256": digest(folder / "result.json"),
                         "config_sha256": digest(config_path)})
    assert len({row["initial_observation_sha256"] for row in statuses}) == world_size
    for snapshot in leader_result["shared_snapshots"]:
        assert snapshot["normalizer_observations"] == snapshot["update"] * 8192 * 24 * world_size
    assert [row["update"] for row in leader_result["shared_snapshots"]] == [
        row["ppo_update"] for row in leader_result["checkpoint_sets"]]
    return statuses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timing-smoke", action="store_true")
    parser.add_argument("--world-size", choices=(1, 2, 4), type=int, default=1)
    args = parser.parse_args()
    targets = [20, 60] if args.timing_smoke else [hour * HOUR for hour in HOURS]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    group_root = output / "training"
    distributed = args.world_size > 1
    training = group_root / "rank-0" if distributed else group_root
    receipt = output / "result.json"
    report = {"status": "training", "checkpoint_training_seconds": targets, "timing_smoke": args.timing_smoke,
              "training_budget_seconds": targets[-1], "seed": 42, "num_envs": 8192, "rollout_steps": 24,
              "no_world_model_constructed": True, "imagined_transitions": 0, "evaluations": [],
              "evaluation_corpus": "canonical LAFAN40, 441140 reference frames", "paper_results_reproduced": False}
    write(receipt, report)
    receipts = []
    try:
        launcher = ["-m", "torch.distributed.run", "--standalone", "--nnodes=1",
                    f"--nproc-per-node={args.world_size}", "--max-restarts=0"] if distributed else ()
        execute("run_ppo_sim_integration.py", ["--motions", args.motions.resolve(), "--output", group_root,
            "--training-purpose", "finite-training", "--updates", 65536, "--checkpoint-seconds", *targets,
            "--trace-updates", 1, "--world-size", args.world_size], output / "training.log", targets[-1] + 1200,
            stream_progress=("PPO_SIM_TIMED_SNAPSHOT ", "PPO_SIM_REAL_G1_TRAINING_PASS "),
            python_options=launcher)
        result, config, _, _ = select_ppo_sim_checkpoint(training)
        assert config["checkpoint_seconds"] == targets and result["elapsed_training_seconds"] >= targets[-1]
        rows = result["updates"]
        manifest = result["checkpoint_sets"]
        assert [target for item in manifest for target in item["crossed_training_clock_seconds"]] == targets
        for item in manifest:
            update = item["ppo_update"]
            after = rows[update - 1]["update_complete_seconds_since_training_entry"]
            before = rows[update - 2]["update_complete_seconds_since_training_entry"] if update > 1 else 0
            assert all(before < seconds <= after for seconds in item["crossed_training_clock_seconds"])
        assert manifest[-1]["ppo_update"] == result["ppo_update_calls"]
        report.update(status="evaluating", actual_update_calls=result["ppo_update_calls"],
                      training_result_sha256=digest(training / "result.json"),
                      timed_snapshots_verified=True, elapsed_training_seconds=result["elapsed_training_seconds"],
                      real_simulator_steps=result["simulator_steps"])
        if distributed:
            statuses = verify_distributed_ranks(group_root, result, args.world_size)
            report.update(world_size=args.world_size, backend="nccl", envs_per_gpu=8192,
                          real_simulator_steps=sum(row["local_real_transitions"] for row in statuses),
                          rank_receipts=statuses, independent_simulation_streams_verified=True,
                          shared_actor_critic_normalizer_and_sampler_verified=True)
        write(receipt, report)
        evaluation_sets = manifest[-1:] if args.timing_smoke else manifest
        for item in evaluation_sets:
            update = item["ppo_update"]
            try:
                row = evaluate(training, update, args.motions, output / f"eval-{update:06d}", receipts)
            except Exception as error:
                row = {"status": "failed", "ppo_update": update, "error": f"{type(error).__name__}: {error}"}
            row.update(crossed_training_clock_seconds=item["crossed_training_clock_seconds"],
                       retained_seconds=item["retained_snapshot_complete_seconds_since_training_entry"])
            report["evaluations"].append(row)
            write(receipt, report)
        report["status"] = "complete" if all(row["status"] == "complete" for row in report["evaluations"]) else "incomplete"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write(receipt, report)
        files = [receipt, training / "resolved-config.json", training / "checkpoint-manifest.json", *receipts]
        if args.timing_smoke:
            files.append(training / "result.json")
        if distributed:
            files.extend(group_root / f"rank-{rank}" / name
                         for rank in range(1, args.world_size)
                         for name in ("resolved-config.json", "result.json"))
        subprocess.run([sys.executable, str(SCRIPTS / "emit_json_receipts.py"), "--max-bytes", "64000000",
                        *map(str, (item for item in files if item.is_file()))], check=True, timeout=180)
    if report["status"] != "complete":
        raise RuntimeError("Incomplete timing study; verified snapshots are retained")
    print("PPO_SIM_TIMED_PIPELINE_COMPLETED " + json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
