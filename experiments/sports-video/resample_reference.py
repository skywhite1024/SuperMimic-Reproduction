"""Explicit source-PTS reference resampling, shared BEFORE P0/P1 assignment.

Usage::

    python resample_reference.py raw.csv --contract raw.contract.json \
        --target-fps 30 --output-dir shared_30hz

Requires per-frame timing linked to source PTS and a source-video hash. Position
and limited robot DOFs are interpolated linearly (DOFs are NOT unwrapped).
Quaternions use shortest-path SLERP. Nothing is extrapolated or time-stretched.

The first source sample is retained. If the last source sample is not on the
target CFR grid, the output stops at the preceding grid sample: less than one
target interval is cropped and explicitly reported. Appending the original end
would break CFR; repeating a last frame would invent duration. Neither is done.
Coverage is measured from first to last motion-sample PTS, not container duration
including the final frame's display interval.

Freeze this one preprocessing result for P0, then apply XY correction to get P1.
E* remains independently reviewed/frozen under the same time-grid policy. This
program accepts no training-branch-specific option, and rejects an input already
marked as root_xy_only. It does not recover lost fast-motion details, repair a
GVHMR model's incorrect time assumptions, or establish physical trackability.
"""

from __future__ import annotations

import argparse
import copy
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

import reference_tools as rt


def _rate(value: str | float) -> Fraction:
    try:
        rate = Fraction(str(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise rt.ContractError("target_fps must be a positive finite rate, e.g. 30 or 30000/1001") from exc
    if rate <= 0 or not np.isfinite(float(rate)):
        raise rt.ContractError("target_fps must be positive and finite")
    return rate


def _has_xy_processing(contract: dict) -> bool:
    records = [contract.get("processing", {})] + contract.get("processing_history", [])
    return any(isinstance(item, dict) and item.get("operation") == "root_xy_only" for item in records)


def resample_reference(
    motion: np.ndarray,
    source_times: np.ndarray,
    contract: dict,
    *,
    target_fps: str | float = 30,
    max_source_gap_s: float = 0.1,
    max_gap_factor: float = 2.5,
) -> tuple[np.ndarray, dict, dict, dict]:
    """Return motion, sidecar, explicit per-frame map, and audit report.

    Raises ContractError before export for unknown units/timing, large gaps,
    source audit failure, or a source shorter than one target interval. The
    source data and metadata are never changed in place. Metadata evidence is
    propagated, not upgraded to verified by interpolation.
    """
    rt.validate_contract(contract)
    if contract["time"]["provenance"] not in {"source_pts", "resampled_source_pts"}:
        raise rt.ContractError("source time must be linked to real PTS; unknown/export_declared_cfr is insufficient")
    if not contract["time"].get("source_video_sha256"):
        raise rt.ContractError("source_video_sha256 is required for source timing provenance")
    if contract["time"]["provenance"] == "resampled_source_pts" and not contract["time"].get("resampling_evidence"):
        raise rt.ContractError("previous resampling requires its evidence chain")
    if _has_xy_processing(contract):
        raise rt.ContractError("resample the shared original P0 reference BEFORE root XY correction; do not retime P1 independently")
    if not all(np.isfinite(v) and v > 0 for v in (max_source_gap_s, max_gap_factor)):
        raise rt.ContractError("gap limits must be finite and positive")
    source_times = np.asarray(source_times, dtype=np.float64)
    motion = np.asarray(motion, dtype=np.float64)
    source_audit = rt.audit_reference(motion, source_times, contract)
    if source_audit["status"] != "passed":
        raise rt.ContractError("source reference rejected: " + ", ".join(source_audit["issues"]))
    # The standalone API, as well as CLI loading, must check the supplied clock
    # against sidecar data rather than trusting two inconsistent time arrays.
    time = contract["time"]
    if "timestamps_s" in time and not np.array_equal(np.asarray(time["timestamps_s"], dtype=float), source_times):
        raise rt.ContractError("supplied source_times disagree with sidecar timestamps_s")
    if "motion_fps" in time and not np.allclose(np.diff(source_times), 1 / time["motion_fps"], rtol=1e-4, atol=1e-6):
        raise rt.ContractError("source motion_fps conflicts with per-frame PTS")
    rate = _rate(target_fps)
    fps, step = float(rate), float(1 / rate)
    dt = np.diff(source_times)
    median_dt = float(np.median(dt))
    limit = min(max_source_gap_s, max_gap_factor * median_dt)
    bad_gaps = np.flatnonzero(dt > limit + 1e-9)
    if len(bad_gaps):
        details = [{"left_frame": int(i), "right_frame": int(i + 1), "gap_s": float(dt[i])} for i in bad_gaps[:20]]
        raise rt.ContractError("source gaps exceed interpolation bound " + json.dumps({"limit_s": limit, "gaps": details}))
    duration = float(source_times[-1] - source_times[0])
    intervals_float = duration * fps
    nearest = round(intervals_float)
    # Account only for floating-point endpoint representation, not a real
    # fractional target interval. No arbitrary 'close enough' frame is appended.
    arithmetic_tolerance = 16 * np.finfo(float).eps * max(1.0, abs(intervals_float))
    if abs(intervals_float - nearest) <= arithmetic_tolerance:
        intervals = nearest
    else:
        intervals = int(np.floor(intervals_float))
    if intervals < 1:
        raise rt.ContractError("source is shorter than one target sample interval; cannot export two CFR samples")
    relative_target = np.arange(intervals + 1, dtype=np.float64) * rate.denominator / rate.numerator
    target_times = source_times[0] + relative_target
    endpoint_snap_s = 0.0
    endpoint_tolerance = max(16 * np.spacing(abs(source_times[-1]) or 1), arithmetic_tolerance / fps)
    if abs(target_times[-1] - source_times[-1]) <= endpoint_tolerance:
        endpoint_snap_s = float(source_times[-1] - target_times[-1])
        target_times[-1] = source_times[-1]
    if target_times[-1] > source_times[-1] or not np.all(np.diff(target_times) > 0):
        raise rt.ContractError("floating-point time precision cannot represent this CFR grid without extrapolation")
    if not np.allclose(np.diff(target_times), step, rtol=1e-8, atol=1e-12):
        raise rt.ContractError("source PTS origin has insufficient floating-point precision for target CFR; normalize clock upstream")
    crop = float(source_times[-1] - target_times[-1])
    if crop < 0 or crop >= step + endpoint_tolerance:
        raise rt.ContractError("internal endpoint policy violation")
    right = np.searchsorted(source_times, target_times, side="right")
    left = np.clip(right - 1, 0, len(source_times) - 1)
    right = np.clip(right, 0, len(source_times) - 1)
    span = source_times[right] - source_times[left]
    alpha = np.divide(target_times - source_times[left], span, out=np.zeros(len(target_times)), where=span > 0)
    if np.any(alpha < 0) or np.any(alpha > 1):
        raise rt.ContractError("interpolation mapping would extrapolate")
    result = (1 - alpha[:, None]) * motion[left] + alpha[:, None] * motion[right]
    # Normalize and use a relative clock inside scipy; finite near-unit inputs
    # have already passed the audit. The resulting rotations are unchanged at
    # source knots even if quaternion representation signs differ.
    relative_source = source_times - source_times[0]
    quaternion = Slerp(relative_source, Rotation.from_quat(motion[:, 3:7]))(target_times - source_times[0]).as_quat()
    hemisphere_changes = 0
    for i in range(1, len(quaternion)):
        if np.dot(quaternion[i - 1], quaternion[i]) < 0:
            quaternion[i] *= -1
            hemisphere_changes += 1
    result[:, 3:7] = quaternion
    coverage = {
        "source_first_sample_pts_s": float(source_times[0]), "source_last_sample_pts_s": float(source_times[-1]),
        "source_first_to_last_sample_duration_s": duration,
        "output_first_sample_pts_s": float(target_times[0]), "output_last_sample_pts_s": float(target_times[-1]),
        "output_first_to_last_sample_duration_s": float(target_times[-1] - target_times[0]),
        "crop_start_s": 0.0, "crop_duration_s": crop, "crop_less_than_one_target_interval": bool(crop < step + endpoint_tolerance),
        "target_interval_s": step, "source_endpoint_included": bool(target_times[-1] == source_times[-1]),
        "numeric_endpoint_snap_s": endpoint_snap_s, "padded_or_repeated_endpoint": False, "time_stretched": False,
    }
    mapping = {
        "schema_version": "sports-resampling-map/1.0", "method": "root_linear_quaternion_shortest_path_slerp_dof_linear",
        "source_timestamps_array_sha256": rt.array_sha256(source_times), "output_timestamps_array_sha256": rt.array_sha256(target_times),
        "source_frame_count": len(motion), "target_frame_count": len(result), "target_fps": fps,
        "target_rate_fraction": f"{rate.numerator}/{rate.denominator}", "coverage": coverage,
        "frames": [{"output_frame": i, "output_pts_s": float(t), "source_left_frame": int(l), "source_right_frame": int(r),
                    "source_left_pts_s": float(source_times[l]), "source_right_pts_s": float(source_times[r]),
                    "alpha_right": float(a), "source_interval_s": float(source_times[r] - source_times[l])}
                   for i, (t, l, r, a) in enumerate(zip(target_times, left, right, alpha))],
    }
    new_contract = copy.deepcopy(contract)
    new_time = new_contract["time"]
    new_time.pop("timestamps_path", None)
    new_time.pop("source_timestamps_sha256", None)
    new_time.update({"timestamps_s": target_times.tolist(), "provenance": "resampled_source_pts", "motion_fps": fps,
                     "resampling_evidence": "resampling-map.json", "resampling_coverage": coverage})
    if "source_fps" not in new_time and "motion_fps" in time:
        new_time["source_fps"] = time["motion_fps"]
    # motion_sha256 can only be assigned AFTER the exact output bytes exist.
    previous_motion_hash = new_contract.pop("motion_sha256", None)
    if previous_motion_hash:
        new_contract["parent_motion_sha256"] = previous_motion_hash
    history = copy.deepcopy(contract.get("processing_history", []))
    if contract.get("processing"):
        history.append(copy.deepcopy(contract["processing"]))
    new_contract["processing_history"] = history
    new_contract["processing"] = {
        "operation": "shared_source_pts_resample", "target_rate_fraction": mapping["target_rate_fraction"],
        "endpoint_policy": "retain_start_crop_fractional_tail_to_CFR; never_pad_or_time_stretch",
        "stage": "shared_before_P0_P1_assignment_and_independent_Estar_freeze",
        "source_motion_array_sha256": rt.array_sha256(motion), "source_timestamps_array_sha256": rt.array_sha256(source_times),
        "map": "resampling-map.json", "report": "resampling-report.json",
    }
    output_audit = rt.audit_reference(result, target_times, new_contract)
    if output_audit["status"] != "passed":
        raise rt.ContractError("resampled reference rejected: " + ", ".join(output_audit["issues"]))
    report = {
        "schema_version": "sports-resampling-report/1.0", "status": "passed", "physical_rollout": "not_run",
        "motion_quality_validated": False, "coverage": coverage, "target_fps": fps,
        "target_rate_fraction": mapping["target_rate_fraction"],
        "source_motion_array_sha256": rt.array_sha256(motion), "output_motion_array_sha256": rt.array_sha256(result),
        "source_timestamps_array_sha256": rt.array_sha256(source_times), "output_timestamps_array_sha256": rt.array_sha256(target_times),
        "input_audit": source_audit, "output_audit": output_audit,
        "implementation": {"resampler_sha256": rt.sha256_file(__file__), **rt.implementation_hashes()},
        "source_gap_policy": {"max_source_gap_s": max_source_gap_s, "max_gap_factor": max_gap_factor,
                              "actual_limit_s": limit, "source_gap_max_s": float(dt.max()), "source_gap_median_s": median_dt,
                              "larger_than_1_5_median_intervals": [int(i) for i in np.flatnonzero(dt > 1.5 * median_dt)]},
        "quaternion_method": "shortest_path_slerp_then_hemisphere_continuity; numeric_normalization_by_scipy",
        "output_quaternion_sign_flips_applied": hemisphere_changes,
        "preserved_metadata": ["joint_mapping", "joint_names", "adaptation", "representation", "world_frame_id", "robot_xml_sha256", "ik_config_sha256"],
        "interpretation": "Explicit common timing preprocessing; this does not validate unseen intermediate motion or physical tracking.",
    }
    return result, new_contract, mapping, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("motion", type=Path)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--target-fps", default="30", help="Positive number or rational, e.g. 30 or 30000/1001")
    parser.add_argument("--max-source-gap-s", type=float, default=.1)
    parser.add_argument("--max-gap-factor", type=float, default=2.5)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-trusted-pickle", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    names = ["resampled.csv", "resampled.contract.json", "resampling-map.json", "resampling-report.json"]
    targets = [args.output_dir / name for name in names]
    report_path = targets[-1]
    input_files: dict[str, Any] = {}
    try:
        contract = rt.read_json(args.contract)
        rt.validate_contract(contract)
        resolved = copy.deepcopy(contract)
        protected = {args.motion.resolve(), args.contract.resolve()}
        if "timestamps_path" in resolved["time"]:
            path = (args.contract.parent / resolved["time"]["timestamps_path"]).resolve()
            resolved["time"]["timestamps_path"] = str(path)
            protected.add(path)
        if protected.intersection(path.resolve() for path in targets):
            raise rt.ContractError("resampling outputs may not overwrite source motion/contract/timestamps")
        if not args.overwrite and any(path.exists() for path in targets):
            raise rt.ContractError("output already exists; choose a new directory or explicit --overwrite")
        input_files = {"motion": str(args.motion.resolve()), "motion_sha256": rt.sha256_file(args.motion),
                       "contract": str(args.contract.resolve()), "contract_sha256": rt.sha256_file(args.contract)}
        if "timestamps_path" in resolved["time"]:
            input_files["timestamps"] = resolved["time"]["timestamps_path"]
            input_files["timestamps_file_sha256"] = rt.sha256_file(resolved["time"]["timestamps_path"])
        motion = rt.load_motion(args.motion, resolved, allow_trusted_pickle=args.allow_trusted_pickle)
        times = rt.load_timestamps(resolved, len(motion))
        result, sidecar, mapping, report = resample_reference(motion, times, resolved, target_fps=args.target_fps,
                                                            max_source_gap_s=args.max_source_gap_s, max_gap_factor=args.max_gap_factor)
        mapping["input_files"] = input_files
        report["input_files"] = input_files
        sidecar["parent_motion_sha256"] = input_files["motion_sha256"]
        sidecar["parent_contract_sha256"] = input_files["contract_sha256"]
        # Include the original sidecar as evidence without changing its relative
        # paths or claiming its optional mapping was independently verified here.
        report["source_contract"] = contract
        # Validate serializability before publishing any artifact. In particular,
        # NumPy scalar flags must not leave a half-written map beside a CSV.
        for value in (mapping, sidecar, report):
            json.dumps(value, ensure_ascii=False, allow_nan=False)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(targets[0], result, delimiter=",", fmt="%.17g")
        reread = np.loadtxt(targets[0], delimiter=",", ndmin=2)
        if not np.array_equal(reread, result):
            raise rt.ContractError("CSV export did not round-trip the resampled values exactly")
        sidecar["motion_sha256"] = rt.sha256_file(targets[0])
        rt._write_json(targets[2], mapping, args.overwrite)
        sidecar["time"]["resampling_map_sha256"] = rt.sha256_file(targets[2])
        rt._write_json(targets[1], sidecar, args.overwrite)
        report["output_files"] = {path.name: {"path": str(path), "sha256": rt.sha256_file(path)} for path in targets[:-1]}
        rt._write_json(report_path, report, args.overwrite)
        print(json.dumps({"status": "passed", "frames": len(result), "crop_duration_s": report["coverage"]["crop_duration_s"],
                          "output": str(report_path)}))
        return 0
    except (rt.ContractError, ValueError, OSError, KeyError, TypeError) as exc:
        report = {"schema_version": "sports-resampling-report/1.0", "status": "rejected", "reason": str(exc),
                  "input_files": input_files, "physical_rollout": "not_run", "motion_quality_validated": False}
        if not report_path.exists():
            rt._write_json(report_path, report, False)
        print(json.dumps(report, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
