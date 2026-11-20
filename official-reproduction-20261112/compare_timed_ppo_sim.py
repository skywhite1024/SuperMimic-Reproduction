"""Compare completed one- and four-H100 PPO-Sim runs at their saved times."""

import argparse
import json
import math
from pathlib import Path

from compare_bones_policy_results import load_policy
from push_recovery_contract import sha256, write_json
from run_ppo_sim_timed_job import HOUR, HOURS


PROTOCOL_FIELDS = (
    "motion_files", "num_envs", "seed", "manual_reset_version", "batch_order", "batch_reset",
    "termination_terms", "physics", "native_joint_limit_clipping_on_reset",
)


def read(path):
    return json.loads(path.read_bytes())


def load_series(root, world_size):
    """Recover the scores and rollout counts for a verified timing study."""
    proof = read(root / "validation.json")
    assert proof["status"] == "timed_ppo_sim_training_and_native_eval_verified"
    assert proof.get("world_size", 1) == world_size and not proof["timing_smoke"]
    assert proof["checkpoints"] == [hour * HOUR for hour in HOURS]
    if world_size > 1:
        assert [rank["rank"] for rank in proof["rank_receipts"]] == list(range(world_size))
        assert sum(rank["local_real_transitions"] for rank in proof["rank_receipts"]) == proof["real_simulator_steps"]
        assert proof["global_actor_observation_count"] == proof["real_simulator_steps"]
    training = root / ("training" if world_size == 1 else "training/rank-0")
    manifest = read(training / "checkpoint-manifest.json")
    summary = read(root / "result.json")
    assert summary["status"] == "complete" and summary["imagined_transitions"] == 0
    assert len(manifest) == len(proof["scores"]) == len(HOURS)

    points = []
    for hour, snapshot, verified in zip(HOURS, manifest, proof["scores"], strict=True):
        update = snapshot["ppo_update"]
        assert update == verified["ppo_update"] and snapshot["crossed_training_clock_seconds"] == [hour * HOUR]
        folder = root / f"eval-{update:06d}" / "evaluation"
        identities, scores, completed, metrics = load_policy(folder / "result.json")
        assert len(identities) == 40 and sum(frames for _, frames in identities) == 441_140
        assert math.isclose(metrics["mean_normalized_full_motion_return"],
                            verified["mean_normalized_full_motion_return"], rel_tol=1e-13)
        assert metrics["successes"] == verified["successes"]
        protocol = read(folder / "protocol.json")
        elapsed = snapshot["retained_snapshot_complete_seconds_since_training_entry"]
        assert elapsed >= hour * HOUR and elapsed <= proof["clock_seconds"]
        info = dict(metrics, ppo_update=update, saved_at_training_seconds=elapsed,
                    real_simulator_transitions=update * summary["num_envs"] * summary["rollout_steps"] * world_size)
        points.append((identities, {key: protocol[key] for key in PROTOCOL_FIELDS}, scores, completed, info))
    assert points[-1][-1]["real_simulator_transitions"] == proof["real_simulator_steps"]
    return points, sha256(root / "validation.json")


def plot(points, path):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import MultipleLocator

    fig, (score_axis, step_axis) = plt.subplots(2, 1, figsize=(7.8, 6.5), sharex=True, layout="constrained")
    for key, label, color, marker in (
        ("one_h100", "1 H100", "#266282", "o"), ("four_h100", "4 H100s", "#ad4f26", "s"),
    ):
        timing = [row[key]["saved_at_training_seconds"] / HOUR for row in points]
        returns = [row[key]["mean_normalized_full_motion_return"] for row in points]
        billions = [row[key]["real_simulator_transitions"] / 1e9 for row in points]
        score_axis.plot(timing, returns, label=label, color=color, marker=marker, linewidth=2)
        step_axis.plot(timing, billions, color=color, marker=marker, linewidth=2)
    score_axis.axhline(0.82, color="#525866", linestyle="--", linewidth=1, label="0.82 reference")
    score_axis.set(ylabel="Mean normalized tracking return", ylim=(0, 1), title="PPO-Sim fixed-start LAFAN40 evaluation")
    score_axis.yaxis.set_major_locator(MultipleLocator(0.2))
    score_axis.legend(loc="lower right", frameon=False)
    step_axis.set(xlabel="Training time (hours)", ylabel="Real simulator transitions (billions)",
                  xticks=HOURS, ylim=(0, None))
    for axis in (score_axis, step_axis):
        axis.grid(axis="y", color="#e0e4e8")
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=180, format="png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("single", "four", "output"):
        parser.add_argument("--" + option, type=Path, required=True)
    args = parser.parse_args()
    single, single_digest = load_series(args.single, 1)
    four, four_digest = load_series(args.four, 4)
    reference_ids, reference_protocol = single[0][:2]
    points = []
    for hour, left, right in zip(HOURS, single, four, strict=True):
        left_ids, left_protocol, left_scores, left_done, left_info = left
        right_ids, right_protocol, right_scores, right_done, right_info = right
        assert left_ids == right_ids == reference_ids and left_protocol == right_protocol == reference_protocol
        both = left_done & right_done
        difference = right_scores - left_scores
        points.append({
            "target_training_hours": hour, "one_h100": left_info, "four_h100": right_info,
            "four_to_one_real_transition_ratio": right_info["real_simulator_transitions"] / left_info["real_simulator_transitions"],
            "mean_paired_return_four_minus_one": float(difference.mean()),
            "mean_paired_return_four_minus_one_on_jointly_completed": float(difference[both].mean()) if both.any() else None,
            "completion": {"both": int(both.sum()), "single_only": int((left_done & ~right_done).sum()),
                           "four_only": int((~left_done & right_done).sum()), "neither": int((~left_done & ~right_done).sum())},
        })
    args.output.mkdir(parents=True, exist_ok=True)
    figure = args.output / "comparison.png"
    temporary_figure = args.output / "comparison.png.tmp"
    plot(points, temporary_figure)
    temporary_figure.replace(figure)
    write_json(args.output / "comparison.json", {
        "clips": len(reference_ids), "reference_frames": 441_140, "exact_ordered_identities_match": True,
        "evaluation_protocol_fields_match": list(PROTOCOL_FIELDS),
        "validation_sha256": {"one_h100": single_digest, "four_h100": four_digest},
        "figure_sha256": sha256(figure), "points": points,
    })


if __name__ == "__main__":
    main()
