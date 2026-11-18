"""Render the audited 33-condition push grids."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np

from push_recovery_contract import POSES, pulse_grid


def render(directory, output):
    colors = ["#bc4b43", "#edc881", "#268373"]
    classes = ["Fell", "Survived, did not settle", "Settled"]
    palette = ListedColormap(colors)
    norm = BoundaryNorm([-.5, .5, 1.5, 2.5], palette.N)
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 5.7), sharex=True, sharey=True)
    fig.subplots_adjust(left=.10, right=.98, bottom=.18, top=.82, hspace=.60, wspace=.08)
    fig.suptitle("G1 balance after a horizontal velocity step", y=.966, fontsize=15, fontweight="bold")
    labels, _ = pulse_grid()
    for i, pose in enumerate(POSES):
        for j, (policy, title) in enumerate((("critic_h32", "Critic, H = 32"),
                                             ("no_critic_h64", "No critic, H = 64"))):
            ax = axes[i, j]
            data = json.loads((directory / policy / f"{pose}.json").read_bytes())
            if [r["case"] for r in data["records"]] != labels:
                raise ValueError("Unexpected order in audited push grid")
            rows = data["records"][1:]
            grid = np.array([0 if r["fell"] else 2 if r["recovery_seconds"] is not None
                             else 1 for r in rows]).reshape(4, 8)
            ax.imshow(grid, cmap=palette, norm=norm, aspect="auto")
            for k, record in enumerate(rows):
                if (settled := record["recovery_seconds"]) is not None:
                    y, x = divmod(k, 8)
                    ax.text(x, y, f"{settled:.1f}", ha="center", va="center", color="white", fontsize=8)
            label = "Constructed T-pose" if i == 0 else "G1 HOME stance"
            gate = "passed" if data["zero_push_eligible"] else "failed"
            ax.set_title(f"{label} · {title}\nZero-push stability {gate}", loc="left", fontsize=9.5)
            ax.set_yticks(range(4), ["+X", "−X", "+Y", "−Y"])
            ax.set_xticks(range(8), [f"{value:.1f}" for value in np.arange(.5, 4.1, .5)])
            ax.set_xticks(np.arange(-.5, 8, 1), minor=True)
            ax.set_yticks(np.arange(-.5, 4, 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=1)
            ax.tick_params(which="both", length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
    fig.text(.54, .095, "Added pelvis velocity (m/s); numbers are settling times (s)",
             ha="center", fontsize=10)
    fig.supylabel("Push direction in the world frame", x=.018, fontsize=10)
    fig.legend(handles=[Patch(facecolor=c, label=label) for c, label in zip(colors, classes, strict=True)],
               loc="lower center", bbox_to_anchor=(.54, .015), ncols=3, frameon=False, fontsize=9)
    fig.savefig(output, dpi=165, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    render(args.directory, args.output)
