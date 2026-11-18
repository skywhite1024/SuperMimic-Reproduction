"""Plot the verified BONES duration endpoints on their common training probe."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator, MultipleLocator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("validation", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    verified = json.loads(args.validation.read_bytes())
    if (verified["status"], verified["kind"], verified["original_JSONs"]) != (
            "all_originals_and_native_scores_verified", "duration", 87):
        raise ValueError("Expected the completed four-case verification receipt")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 12, "axes.titleweight": "bold",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#9ca9b6", "text.color": "#14283a",
                         "axes.labelcolor": "#14283a"})
    fig, (returns, errors) = plt.subplots(1, 2, figsize=(11, 4.8))
    fig.subplots_adjust(left=.085, right=.97, top=.70, bottom=.23, wspace=.34)
    fig.text(.085, .94, "BONES duration comparison", weight="bold", fontsize=18)
    fig.text(.085, .875, "Same 64 training motions at evaluation · one seed · 4,096 environments per case",
             color="#415970", fontsize=11)

    labels = ("1 h", "7 h", "40 h", "263.19 h")
    colors = ("#007b6e", "#5752a1", "#9c5819", "#285d99")
    markers = ("o", "^", "D", "s")
    cases = [verified["cases"][key] for key in ("1h", "7h", "40h", "full")]
    handles = []
    for label, color, mark, case in zip(labels, colors, markers, cases, strict=True):
        points = case["points"]
        x = [p["real_transitions"] / 1e6 for p in points]
        y = [p["mean_normalized_full_motion_return"] for p in points]
        handles.extend(returns.plot(x, y, marker=mark, markersize=6, linewidth=1.9,
                                    color=color, label=label))
        error = case["forecast"]["eighth_transition_normalized_state_mse"]
        errors.scatter(case["data_hours"], error, color=color, marker=mark, s=50, zorder=3)
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.068, .835),
               frameon=False, ncols=4, handlelength=2.3, columnspacing=1.7)
    returns.set(title="Motion tracking", xlabel="Real simulator transitions (millions)",
                ylabel="Mean normalized return", ylim=(.70, .935))
    returns.set_xticks([16.388096, 32.772096, 65.540096], ["16.4", "32.8", "65.5"])
    returns.axhline(.90, color="#617283", linewidth=.9, dashes=(3, 3), zorder=1)
    returns.text(17, .9025, "Local 0.90 threshold", fontsize=8, color="#526679")
    returns.yaxis.set_major_locator(MultipleLocator(.05))

    errors.set(title="Eight-transition state prediction", xlabel="Training data (hours, log scale)",
               ylabel="Normalized state MSE at step 8", xscale="log", xlim=(.7, 390), ylim=(0, .025))
    errors.xaxis.set_major_locator(FixedLocator([c["data_hours"] for c in cases]))
    errors.xaxis.set_major_formatter(FixedFormatter(["1", "7", "40", "263.19"]))
    errors.yaxis.set_major_locator(MultipleLocator(.005))
    for ax in (returns, errors):
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e0e6eb", linewidth=.8)
        ax.tick_params(length=0, pad=6)
        ax.title.set_ha("left")
        ax.title.set_position((0, 1.03))
    fig.text(.085, .061, "Forecasts use each case's fresh policy rollouts at 65.54M training transitions.",
             fontsize=9, color="#415970")
    fig.savefig(args.output, dpi=160, facecolor="white")


if __name__ == "__main__":
    main()
