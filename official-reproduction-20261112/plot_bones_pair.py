"""Plot the complete raw BONES comparison from a verified local evidence package."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.pair_root / "comparison/summary.json").read_text())
    with (args.pair_root / "comparison/paired-motions.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == summary["clips"] == 142220
    assert sorted(int(row["index"]) for row in rows) == list(range(142220))
    returns = np.array([[float(row["reference_normalized_return"]),
                         float(row["candidate_normalized_return"])] for row in rows])
    assert np.isfinite(returns).all()
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.2))
    colors = ["#526c8f", "#14816e"]
    labels = ["1,001 steps", "16,001 steps"]
    rates = [100 * summary[key]["successes"] / len(rows) for key in ("reference", "candidate")]
    bars = axes[0, 0].bar(labels, rates, color=colors, width=0.5)
    axes[0, 0].bar_label(bars, labels=[f"{value:.2f}%" for value in rates], padding=5)
    axes[0, 0].set(ylim=(0, 105), ylabel="Complete clips (%)", title="A  Native full-clip success")
    x = np.arange(2)
    for i, (key, label) in enumerate(zip(("reference", "candidate"), labels)):
        values = [summary[key][name] for name in ("mean_normalized_full_motion_return",
                                                "frame_weighted_normalized_full_motion_return")]
        bars = axes[0, 1].bar(x + (i - 0.5) * 0.32, values, 0.32, color=colors[i], label=label)
        axes[0, 1].bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
    axes[0, 1].set(xticks=x, xticklabels=["Equal clip weight", "Reference-frame weight"],
                   ylim=(0, 1.02), ylabel="Normalized full-reference return", title="B  All failures included")
    axes[0, 1].legend(frameon=False, loc="upper left", fontsize=8)
    lower = min(0, float(returns.min()))
    upper = max(1, float(returns.max()))
    hb = axes[1, 0].hexbin(returns[:, 0], returns[:, 1], gridsize=55, bins="log", mincnt=1, cmap="viridis")
    axes[1, 0].plot([lower, upper], [lower, upper], color="#a54949", linestyle="--", linewidth=1)
    axes[1, 0].set(xlabel="1,001-step return", ylabel="16,001-step return",
                   title="C  Paired returns for every clip", xlim=(lower, upper), ylim=(lower, upper))
    fig.colorbar(hb, ax=axes[1, 0], label="Clips per hexagon", pad=0.025)
    names = ["both_success", "recovered_success", "regressed_failure", "both_failure"]
    values = [summary["success_transitions"][name] for name in names]
    assert sum(values) == len(rows)
    bars = axes[1, 1].barh(["Both success", "Failure to success", "Success to failure", "Both failure"],
                           values, color=["#14816e", "#6aab8b", "#b25b53", "#8c8b89"])
    axes[1, 1].bar_label(bars, labels=[f"{value:,}" for value in values], padding=4)
    axes[1, 1].invert_yaxis()
    axes[1, 1].set(xlabel="Clips", xlim=(0, max(values) * 1.22), title="D  Success transitions")
    fig.suptitle("Raw BONES: 1k versus 16k training", fontsize=18, x=0.05, ha="left")
    fig.text(0.05, 0.927, "142,220 training clips | seed 42 | same frozen evaluation protocol | eager goal assembly", fontsize=10)
    fig.text(0.05, 0.023, "Single runs; paper filters absent. Neither determinism, strict motion quality nor held-out generalization is established.", fontsize=9)
    fig.tight_layout(rect=(0.02, 0.05, 1, 0.91), h_pad=2.2, w_pad=2.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
