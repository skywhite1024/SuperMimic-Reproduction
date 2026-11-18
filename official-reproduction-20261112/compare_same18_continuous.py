"""Compare verified continuous first-order and finite PPO-WM points on the same motions."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from push_recovery_contract import sha256, write_json


def read(path):
    return json.loads(path.read_bytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True, help="Verified original 1118b receipt directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    older = Path(__file__).resolve().parent
    previous = read(older / "finite-same18-comparison-1116.json")
    native_validation = args.native / "validation.json"
    verified = read(native_validation)
    assert (verified["status"], verified["kind"], verified["original_JSONs"]) == (
        "all_originals_and_native_scores_verified", "same18", 51)
    same18 = verified["cases"]["same18"]
    native_dir = args.native / "runs/lafan-locomotion18-control-1118b/main"
    protocol = read(native_dir / "eval-000064001/protocol.json")
    for point in same18["points"]:
        step = point["collection_steps"]
        item = read(native_dir / f"eval-{step:09d}/protocol.json")
        assert item["motion_files"] == same18["motion_files"] == protocol["motion_files"]
        assert item["termination_terms"] == protocol["termination_terms"]
    curves = [{"label": "First-order, continuous", "points": [
        {"real_steps": p["real_transitions"], "imagined_steps": p["configured_imagined_training_transitions"],
         "score": p["mean_normalized_full_motion_return"], "successes": p["successes"],
         "collection_steps": p["collection_steps"]} for p in same18["points"]]}]
    for row in previous["PPO_WM"]:
        folder = older / "ppo-wm-batch-learning-1115" / row["case"]
        assert sha256(folder / "validation.json") == row["validation_sha256"]
        for point in row["points"]:
            item = read(folder / f"eval-{point['ppo_update']:06d}/protocol.json")
            assert item["motion_files"] == protocol["motion_files"]
            assert item["termination_terms"] == protocol["termination_terms"]
            assert item["manual_reset_version"] == 2
        curves.append({"label": "PPO-WM " + row["label"], "points": [
            {"real_steps": p["real_simulator_steps"], "imagined_steps": p["cumulative_imagined_steps"],
             "score": p["mean_normalized_full_motion_return"], "successes": p["successes"],
             "ppo_update": p["ppo_update"]} for p in row["points"]]})
    previous_by_step = {row["collection_steps"]: row for row in previous["first_order"]}
    repeated_points = [dict(collection_steps=p["collection_steps"], continuous_score=p["score"],
                           prior_interrupted_parent_score=previous_by_step[p["collection_steps"]][
                               "mean_normalized_full_motion_return"])
                       for p in curves[0]["points"] if p["collection_steps"] in previous_by_step]
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "comparison.json", {
        "status": "complete_continuous_and_finite_same18_curves_compared", "motions": 18,
        "reference_frames": 220352, "first_order_run_sha256": sha256(native_validation),
        "previous_comparison_sha256": sha256(older / "finite-same18-comparison-1116.json"),
        "first_sample_at_local_return_082": same18["first_observed_at_or_above_082"],
        "points_at_matching_first_order_collection_counts": repeated_points,
        "identical_packed_motion_and_termination_rules": True, "curves": curves,
        "paper_training_return_or_budget_matched": False,
    })

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), sharey=True)
    fig.subplots_adjust(left=.075, right=.976, top=.66, bottom=.20, wspace=.16)
    fig.text(.075, .94, "First-order and PPO-WM tracking on the same 18 LAFAN motions",
             weight="bold", fontsize=15)
    fig.text(.075, .877, "Mean full-motion evaluation return · one seed · each point uses the same 220,352 reference frames",
             color="#425971", fontsize=10)
    colors = ["#174659", "#0d67a1", "#916223", "#8062a2", "#588342"]
    markers = ["o", "s", "D", "^", "v"]
    for curve, color, marker in zip(curves, colors, markers, strict=True):
        for axis, key, scale in zip(axes, ("real_steps", "imagined_steps"), (1e6, 1e9), strict=True):
            axis.plot([p[key] / scale for p in curve["points"]], [p["score"] for p in curve["points"]],
                      label=curve["label"], color=color, marker=marker, lw=2, markersize=4.5)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_axisbelow(True)
        axis.set_ylim(-.015, .965)
        axis.grid(axis="y", alpha=.23)
        axis.axhline(.82, color="#627587", linestyle="--", linewidth=.9)
    axes[0].set(xlabel="Real simulator transitions (millions)", ylabel="Mean normalized return")
    axes[1].set(xlabel="Counted imagined training transitions (billions)")
    axes[1].text(3.23, .775, "Local 0.82 threshold", fontsize=8, color="#526679", ha="right")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.066, .826), ncols=3,
               frameon=False, fontsize=9, columnspacing=1.5)
    fig.text(.075, .048, "The PPO-WM runs used a finite local budget. The paper reports a different training-return measure.",
             fontsize=9, color="#425971")
    fig.savefig(output / "same18-comparison.png", dpi=160, facecolor="white")


if __name__ == "__main__":
    main()
