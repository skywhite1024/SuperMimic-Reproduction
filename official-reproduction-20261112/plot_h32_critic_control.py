"""Plot fixed-start LAFAN40 scores for the verified equal-horizon ablation."""

import argparse
import json
from pathlib import Path

from push_recovery_contract import sha256, write_json


TARGET = .82


def read(path):
    return json.loads(path.read_bytes())


def series(verified, timing):
    snapshots = {row["collection_steps"]: row["retained_snapshot_complete"]
                 for row in read(timing)["checkpoints"]}
    public_keys = ("collection_steps", "real_transitions", "clips", "successes",
                   "mean_normalized_full_motion_return")
    return [{**{key: row[key] for key in public_keys},
             "hours_since_runner_entry": snapshots[row["collection_steps"]]["seconds_since_runner_run_entry"] / 3600}
            for row in verified]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("validation", "critic-timing", "no-critic-timing", "output"):
        parser.add_argument("--" + arg, required=True, type=Path)
    args = parser.parse_args()
    verified = read(args.validation)
    assert verified["status"] == "full40_H32_critic_free_originals_verified"
    matched = {"With critic": series(verified["critic_curve"], args.critic_timing),
               "Without critic": series(verified["no_critic_curve"], args.no_critic_timing)}
    assert len(matched["With critic"]) == len(matched["Without critic"]) == 9
    assert [row["collection_steps"] for row in matched["With critic"]] == [
        row["collection_steps"] for row in matched["Without critic"]]
    args.output.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), sharey=True, layout="constrained")
    for (label, rows), color in zip(matched.items(), ("#2166A5", "#BD581D"), strict=True):
        returns = [row["mean_normalized_full_motion_return"] for row in rows]
        axes[0].plot([row["real_transitions"] / 1e6 for row in rows], returns, "o-", label=label, color=color)
        axes[1].plot([row["hours_since_runner_entry"] for row in rows], returns, "o-", label=label, color=color)
    axes[0].set(xlabel="Real simulator transitions (millions)", ylabel="Mean normalized full-motion return")
    axes[1].set(xlabel="Training time since runner entry (hours)")
    for ax in axes:
        ax.axhline(TARGET, color="#666666", lw=1, linestyle="--", label="0.82 return")
        ax.set_ylim(-.025, 1.02)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.19)
    axes[0].legend(loc="lower right", frameon=False)
    fig.suptitle("LAFAN40 fixed-start evaluation at H = 32")
    image = args.output / "h32-critic-control.png"
    fig.savefig(image, dpi=180)
    plt.close(fig)
    report = {"status": "equal_horizon_fixed_start_curve_comparison", "seed": 42, "horizon": 32,
              "originals_verified": verified["original_JSONs"], "evaluation_return_target": TARGET,
              "curves": matched,
              "first_sampled_at_or_above_target": {label: next((row for row in rows
                  if row["mean_normalized_full_motion_return"] >= TARGET), None) for label, rows in matched.items()},
              "input_sha256": {"validation": sha256(args.validation), "critic_timing": sha256(args.critic_timing),
                               "no_critic_timing": sha256(args.no_critic_timing)}, "plot_sha256": sha256(image)}
    write_json(args.output / "curve-comparison.json", report)


if __name__ == "__main__":
    main()
