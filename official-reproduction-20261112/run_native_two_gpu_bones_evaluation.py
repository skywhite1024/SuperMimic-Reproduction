"""Evaluate the completed two-rank native actor on the verified full BONES pack."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from push_recovery_contract import sha256, write_json
from run_lafan_learning_curve import execute
from run_native_two_gpu_gate import BONES_CLIPS, BONES_QUALITY_COLLECTIONS
from select_completed_checkpoint import select_completed_checkpoint


SCRIPTS = Path(__file__).resolve().parent
FRAMES = 47374271


def read(path):
    return json.loads(path.read_bytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("training", "motions", "output"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--training-driver-sha256", required=True)
    parser.add_argument("--membership-sha256", required=True)
    args = parser.parse_args()
    train, motions, output = (path.resolve() for path in (args.training, args.motions, args.output))
    _, _, selection = select_completed_checkpoint(train, two_rank=True)
    transitions = BONES_QUALITY_COLLECTIONS * 8192
    if (selection["verified_driver_sha256"] != args.training_driver_sha256
            or selection["evaluated_collection_steps"] != BONES_QUALITY_COLLECTIONS
            or selection["global_real_simulator_steps"] != transitions):
        raise ValueError("The verified two-rank training result must match the selected checkpoint")
    manifest = motions.with_name("full-duration-pack.json")
    pack = read(manifest)
    hashes = {file.name: sha256(file) for file in sorted(motions.iterdir()) if file.is_file()}
    if (pack["case"] != "full" or pack["clips"] != BONES_CLIPS or pack["frames"] != FRAMES
            or pack["membership_sha256"] != args.membership_sha256 or pack["files"] != hashes):
        raise ValueError("The audited full BONES membership and bytes are required")

    output.mkdir(parents=True, exist_ok=False)
    ev = output / "evaluation"
    result, protocol, accounting = [ev / name for name in ("result.json", "protocol.json", "accounting.json")]
    trajectory, initial = ev / "audit/summary.json", ev / "audit/initial-states.json"
    receipt = output / "result.json"
    report = {"status": "evaluating", "training_driver_sha256": args.training_driver_sha256,
              "training_rank0_result_sha256": selection["parent_result_sha256"],
              "training_group_sha256": selection["group_result_sha256"],
              "global_real_transitions": transitions, "training_ranks": 2,
              "training_collections_per_rank": BONES_QUALITY_COLLECTIONS, "clips": BONES_CLIPS,
              "reference_frames": FRAMES, "membership_sha256": args.membership_sha256,
              "author_exact_membership": False}
    write_json(receipt, report)
    try:
        execute("evaluate_motion_corpus_v2.py", ["--run", train, "--training-algorithm", "first-order-two-rank",
            "--motions", motions, "--output", ev, "--expected-clips", BONES_CLIPS,
            "--num-envs", 1024, "--order", "length", "--seed", 42], output / "evaluation.log", 7200,
            stream_progress=("BATCH_COMPLETED ", "OFFICIAL_CORPUS_EVAL_PASS "))
        execute("audit_corpus_result.py", ["--result", result, "--protocol", protocol, "--motions", motions,
            "--expected-clips", BONES_CLIPS, "--require-trajectories", "--output", accounting],
            output / "accounting.log", 1500)
        execute("audit_saved_initial_states_v2.py", ["--evaluation", ev, "--motions", motions,
            "--expected-clips", BONES_CLIPS, "--output", ev / "audit"], output / "audit.log", 3300)
        protocol_report, counts, initial_report = map(read, (protocol, accounting, initial))
        assert protocol_report["motion_files"] == hashes and protocol_report["manual_reset_version"] == 2
        verified_selection = dict(selection, resolved_config_sha256=sha256(train / "resolved-config.json"))
        assert protocol_report["checkpoint_selection"] == verified_selection
        assert counts["result_sha256"] == sha256(result) and counts["reference_frames"] == FRAMES
        assert initial_report["all_saved_initial_states_verified"]
        report.update(status="complete", successes=counts["successes"], failures=counts["failures"],
                      mean_normalized_return=counts["mean_normalized_full_motion_return"],
                      per_motion_result_sha256=counts["result_sha256"])
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(receipt, report)
        files = (manifest, receipt, protocol, accounting, initial, trajectory)
        subprocess.run([sys.executable, str(SCRIPTS / "emit_json_receipts.py"), "--max-bytes", "4000000",
                        *map(str, (file for file in files if file.is_file()))], timeout=180, check=True)
    print("BONES_NATIVE_2GPU_EVAL_COMPLETE " + json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
