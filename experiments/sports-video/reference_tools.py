"""Offline G1 reference audit and translation-only court correction.

No pose estimator, fabricated foot anchors, training or simulator is invoked.
CSV is xyz[m] + quaternion[xyzw] + 29 named joints[rad]. GMR pickle loading
requires explicit trust because pickle can execute arbitrary code. GVHMR SMPL
parameters must be retargeted before this tool; they are not G1 joint angles.

Commands (paths in the contract are relative to its directory)::

    python reference_tools.py probe-video clip.mp4 --output pts.json
    python reference_tools.py audit motion.csv --contract contract.json --output audit.json
    python reference_tools.py repair-xy motion.csv --contract contract.json \
        --anchors anchors.json --output-dir corrected

Anchor document example, all observations supplied independently by the caller::

    {"schema_version":"sports-anchors/1.0", "world_frame_id":"court01",
     "representation":"robot_adapted", "calibrations":{}, "anchors":[
       {"anchor_id":"a1", "frame_index":12, "foot":"left",
        "contact_point":"sole", "ground_contact":true, "confidence":0.95,
        "observed_metric_xy":[1.2,0.4], "reference_foot_offset_xy":[0.1,0.08],
        "role":"fit", "source_evidence":"manual-frame-12.png"}]}

reference_foot_offset_xy is the CURRENT reference's world-axis foot-minus-root
offset (from externally verified FK), not a body-local coordinate. Pixel anchors
instead supply observed_pixel_xy + calibration_id. Each calibration requires
H_image_to_canvas, A_canvas_to_metric, ground_plane_z_m=0, line_semantics,
verified=true and evidence. Homography is rejected for non-ground observations.
Optional coordinate_transform explicitly maps observed metric coordinates into
the reference's frame/representation; no automatic body/root scaling is applied.

Fit and holdout anchors are separate. Only adjacent, explicitly identified
stance frames generate a foot-velocity term. Sparse observations do not imply
continuous stance. Unsupported regions are left unchanged and reported.
Numerical acceptance and synthetic unit tests do not establish motion quality.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import jsonschema
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

G1_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


class ContractError(ValueError):
    """Unusable or ambiguous input. No correction should be exported."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    # Include shape and dtype, not just a potentially ambiguous byte stream.
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(str((value.shape, value.dtype.str)).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def implementation_hashes() -> dict:
    return {"tool_sha256": sha256_file(__file__),
            "contract_schema_sha256": sha256_file(Path(__file__).with_name("reference_contract.schema.json"))}


def read_json(path: str | Path) -> dict:
    result = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(result, dict):
        raise ContractError(f"Expected a JSON object: {path}")
    return result


def validate_contract(contract: dict) -> None:
    schema = read_json(Path(__file__).with_name("reference_contract.schema.json"))
    errors = sorted(jsonschema.Draft202012Validator(schema).iter_errors(contract), key=lambda e: str(e.path))
    if errors:
        raise ContractError("contract: " + "; ".join(f"{list(e.path)} {e.message}" for e in errors[:5]))
    if contract["joint_names"] != G1_JOINT_NAMES:
        raise ContractError("joint_names must match the G1 SDK order; explicitly remap upstream, never silently reorder")
    mapping = contract.get("joint_mapping")
    if mapping and mapping["target_joint_names"] != contract["joint_names"]:
        raise ContractError("joint_mapping.target_joint_names disagrees with exported columns")
    # JSON allows non-finite values in some writers; schema alone cannot catch them.
    for value in [contract["adaptation"]["root_xy_scale"], contract["adaptation"]["time_scale"]]:
        if not np.isfinite(value) or value <= 0:
            raise ContractError("adaptation scales must be finite and positive")
    if mapping and not np.isfinite(mapping["offset_rad"]).all():
        raise ContractError("joint_mapping offsets must be finite")


def load_timestamps(contract: dict, count: int, base_dir: str | Path = ".") -> np.ndarray:
    time = contract["time"]
    if "timestamps_s" in time:
        values = time["timestamps_s"]
    else:
        path = Path(base_dir) / time["timestamps_path"]
        if path.suffix.lower() == ".npy":
            values = np.load(path, allow_pickle=False)
        elif path.suffix.lower() == ".json":
            loaded = read_json(path)
            values = loaded["timestamps_s"]
        else:
            values = np.loadtxt(path, delimiter=",")
        expected = time.get("source_timestamps_sha256")
        if expected and sha256_file(path) != expected.lower():
            raise ContractError("timestamps file hash mismatch")
    times = np.asarray(values, dtype=np.float64)
    if times.shape != (count,) or not np.isfinite(times).all():
        raise ContractError(f"timestamps must be {count} finite seconds, one per motion frame")
    if not np.all(np.diff(times) > 0):
        raise ContractError("timestamps must be strictly increasing; duplicates/dropouts require upstream repair")
    fps = time.get("motion_fps")
    if fps is not None:
        if not np.isfinite(fps) or fps <= 0:
            raise ContractError("motion_fps must be finite and positive")
        if not np.allclose(np.diff(times), 1.0 / fps, rtol=1e-4, atol=1e-6):
            raise ContractError("declared motion_fps conflicts with per-frame timestamps")
    return times


def load_motion(path: str | Path, contract: dict, *, allow_trusted_pickle: bool = False) -> np.ndarray:
    path = Path(path)
    if contract.get("motion_sha256") and sha256_file(path) != contract["motion_sha256"].lower():
        raise ContractError("motion file hash mismatch")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        motion = np.loadtxt(path, delimiter=",", comments="#", ndmin=2)
    elif suffix in {".npz", ".pkl", ".pickle"}:
        if suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                data = {key: archive[key] for key in archive.files}
        else:
            if not allow_trusted_pickle:
                raise ContractError("GMR pickle requires --allow-trusted-pickle; use CSV/NPZ for untrusted inputs")
            import pickle
            with path.open("rb") as stream:
                data = pickle.load(stream)
        if not isinstance(data, dict):
            raise ContractError("GMR archive must be a mapping")
        if "joint_names" in data and np.asarray(data["joint_names"]).tolist() != contract["joint_names"]:
            raise ContractError("archive joint_names disagrees with sidecar")
        if "quaternion_order" in data and str(np.asarray(data["quaternion_order"]).item()) != "xyzw":
            raise ContractError("archive quaternion_order must match xyzw sidecar")
        quat_key = "root_rot_xyzw" if "root_rot_xyzw" in data else "root_rot"
        if not all(key in data for key in ("root_pos", quat_key, "dof_pos")):
            raise ContractError("Expected GMR root_pos, root_rot[_xyzw], dof_pos; SMPL poses require retargeting")
        parts = [np.asarray(data[key], dtype=np.float64) for key in ("root_pos", quat_key, "dof_pos")]
        if any(part.ndim != 2 for part in parts) or [part.shape[1] for part in parts] != [3, 4, 29]:
            raise ContractError("GMR fields must be (N,3), (N,4), (N,29)")
        motion = np.concatenate(parts, axis=1)
        archive_fps = data.get("fps")
        if archive_fps is not None:
            times = load_timestamps(contract, len(motion), path.parent)
            fps = float(np.asarray(archive_fps).item())
            if not np.isfinite(fps) or fps <= 0 or not np.allclose(np.diff(times), 1 / fps, rtol=1e-4, atol=1e-6):
                raise ContractError("GMR archive fps conflicts with timestamps; container labels are not a resampling operation")
    else:
        raise ContractError("Supported: CSV or GMR NPZ/pickle; GVHMR .pt must first undergo GMR retargeting")
    return np.asarray(motion, dtype=np.float64)


def audit_reference(motion: np.ndarray, times: np.ndarray, contract: dict) -> dict:
    validate_contract(contract)
    issues, warnings = [], []
    result: dict[str, Any] = {"schema_version": "sports-reference-audit/1.0", "status": "rejected", "issues": issues,
                              "warnings": warnings, "physical_rollout": "not_run", "motion_quality_validated": False,
                              "implementation": implementation_hashes()}
    if motion.ndim != 2 or motion.shape[1] != 36 or len(motion) < 2:
        issues.append("motion_requires_at_least_two_rows_and_36_columns")
        return result
    if not np.isfinite(motion).all():
        issues.append("nonfinite_motion")
        return result
    if times.shape != (len(motion),) or not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
        issues.append("invalid_timestamps")
        return result
    dt = np.diff(times)
    quat = motion[:, 3:7]
    norm_error = np.abs(np.linalg.norm(quat, axis=1) - 1)
    if norm_error.max() > 1e-4:
        issues.append("quaternion_norm_error_exceeds_1e-4")
    flips = int(np.sum(np.sum(quat[:-1] * quat[1:], axis=1) < 0))
    if flips:
        warnings.append("quaternion_hemisphere_flips_preserved; use rotation-aware interpolation")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        qd = np.diff(motion[:, 7:], axis=0) / dt[:, None]
        root_xy_velocity = np.diff(motion[:, :2], axis=0) / dt[:, None]
    if not np.isfinite(qd).all() or not np.isfinite(root_xy_velocity).all():
        issues.append("nonfinite_velocity_from_timestamps")
        return result
    # Do not unwrap limited robot joints; that can disguise a genuine discontinuity.
    speeds = np.abs(qd).max(axis=0)
    limits = contract.get("joint_velocity_limits_rad_s")
    if limits is None:
        warnings.append("joint_velocity_limits_unavailable")
        velocity_violations = None
    else:
        limits = np.asarray(limits, dtype=float)
        if not np.isfinite(limits).all() or np.any(limits <= 0):
            raise ContractError("joint velocity limits must be finite and positive")
        velocity_violations = int(np.sum(np.abs(qd) > limits[None, :] + 1e-8))
        if velocity_violations:
            issues.append("joint_velocity_limit_exceeded")
    position_limits = contract.get("joint_limits_rad")
    if position_limits is None:
        warnings.append("joint_position_limits_unavailable")
        position_violations = None
    else:
        bounds = np.asarray(position_limits, dtype=float)
        if not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ContractError("joint position limits must be finite ordered lower/upper bounds")
        position_violations = int(np.sum((motion[:, 7:] < bounds[:, 0] - 1e-8) | (motion[:, 7:] > bounds[:, 1] + 1e-8)))
        if position_violations:
            issues.append("joint_position_limit_exceeded")
    mapping_verified = bool(contract.get("joint_mapping", {}).get("verified", False))
    if not mapping_verified:
        warnings.append("joint_mapping_not_independently_verified")
    timing_verified = contract["time"]["provenance"] in {"source_pts", "resampled_source_pts"}
    if not timing_verified:
        warnings.append("timing_not_verified_against_source_pts")
    if contract["time"]["provenance"] == "resampled_source_pts" and not contract["time"].get("resampling_evidence"):
        warnings.append("resampling_evidence_missing")
        timing_verified = False
    legacy_compatible = bool(np.allclose(dt, 1 / 30, rtol=1e-4, atol=1e-6))
    if not legacy_compatible:
        warnings.append("legacy_MotionLib_default_requires_CFR_30Hz; consume_timestamps_or_explicitly_resample_upstream")
    result.update({
        "status": "passed" if not issues else "rejected", "frames": len(motion), "duration_s": float(times[-1] - times[0]),
        "effective_fps": float((len(times) - 1) / (times[-1] - times[0])), "dt_min_s": float(dt.min()), "dt_max_s": float(dt.max()),
        "timestamps_array_sha256": array_sha256(times), "motion_array_sha256": array_sha256(motion),
        "quaternion_norm_error_max": float(norm_error.max()), "quaternion_hemisphere_flips": flips,
        "joint_speed_max_rad_s": {name: float(value) for name, value in zip(contract["joint_names"], speeds)},
        "joint_velocity_limit_violations": velocity_violations, "joint_position_limit_violations": position_violations,
        "root_xy_speed_max_m_s": float(np.linalg.norm(root_xy_velocity, axis=1).max()),
        "joint_mapping_verified_by_contract": mapping_verified, "timing_verified_by_contract": timing_verified,
        "legacy_motionlib_default_compatible": legacy_compatible,
        "ready_for_physical_diagnostic": not issues and not warnings,
        "representation": contract["representation"], "adaptation": contract["adaptation"],
    })
    return result


def _matrix3(value: Any, name: str, affine: bool = False) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    # Homographies are defined only up to scale: an absolute determinant cutoff
    # would incorrectly reject a valid H multiplied by a small scalar.
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) < 3:
        raise ContractError(f"{name} must be a finite invertible 3x3 matrix")
    if affine and not np.allclose(matrix[2], [0, 0, 1], atol=1e-10, rtol=0):
        raise ContractError(f"{name} must explicitly be affine with last row [0,0,1]")
    return matrix


def _xy(value: Any, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    if value.shape != (2,) or not np.isfinite(value).all():
        raise ContractError(f"{name} must be two finite coordinates")
    return value


def _project(matrix: np.ndarray, point: np.ndarray) -> np.ndarray:
    out = matrix @ np.r_[point, 1.0]
    if abs(out[2]) < 1e-10:
        raise ContractError("homography projects anchor to infinity")
    return out[:2] / out[2]


def project_ground_anchor(anchor: dict, calibrations: dict) -> np.ndarray:
    """Project an explicitly observed contact point, never pelvis/ankle/flight."""
    if anchor.get("ground_contact") is not True:
        raise ContractError("not_an_observed_ground_contact")
    if anchor.get("contact_point") not in {"sole", "heel_contact", "toe_contact"}:
        raise ContractError("point_is_not_a_ground_contact_surface; pelvis/ankle/airborne_points_are_invalid")
    if not isinstance(anchor.get("source_evidence"), str) or not anchor["source_evidence"].strip():
        raise ContractError("missing_external_anchor_evidence")
    pixels, metric = "observed_pixel_xy" in anchor, "observed_metric_xy" in anchor
    if pixels == metric:
        raise ContractError("supply_exactly_one_of_observed_pixel_xy_or_observed_metric_xy")
    if metric:
        return _xy(anchor["observed_metric_xy"], "observed_metric_xy")
    calibration = calibrations.get(anchor.get("calibration_id"))
    if not isinstance(calibration, dict) or calibration.get("verified") is not True:
        raise ContractError("missing_or_unverified_shot_calibration")
    if not calibration.get("line_semantics") or not calibration.get("evidence"):
        raise ContractError("calibration_requires_line_semantics_and_evidence")
    if calibration.get("ground_plane_z_m") != 0:
        raise ContractError("calibration_requires_explicit_ground_plane_z_m_zero")
    H = _matrix3(calibration.get("H_image_to_canvas"), "H_image_to_canvas")
    A = _matrix3(calibration.get("A_canvas_to_metric"), "A_canvas_to_metric", affine=True)
    return _project(A @ H, _xy(anchor["observed_pixel_xy"], "observed_pixel_xy"))


@dataclass(frozen=True)
class RepairConfig:
    min_confidence: float = 0.8
    max_anchor_gap_s: float = 2.0
    huber_delta_m: float = 0.05
    lambda_smooth: float = 0.01
    lambda_prior: float = 0.001
    lambda_contact: float = 0.01
    max_correction_m: float = 2.0
    max_iterations: int = 20


@dataclass
class RepairResult:
    motion: np.ndarray | None
    report: dict


def _support(times: np.ndarray, frame_ids: list[int], max_gap: float) -> np.ndarray:
    mask = np.zeros(len(times), dtype=bool)
    unique = sorted(set(frame_ids))
    for left, right in zip(unique[:-1], unique[1:]):
        if times[right] - times[left] <= max_gap:
            mask[left:right + 1] = True
    return mask


def _intervals(mask: np.ndarray, times: np.ndarray) -> list[dict]:
    boundaries = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(int)))
    return [{"first_frame": int(a), "last_frame": int(b - 1), "first_time_s": float(times[a]), "last_time_s": float(times[b - 1])}
            for a, b in zip(boundaries[::2], boundaries[1::2])]


def repair_root_xy(motion: np.ndarray, times: np.ndarray, contract: dict, anchors_doc: dict,
                   config: RepairConfig = RepairConfig()) -> RepairResult:
    """IRLS translation-only fit. A rejected result has motion=None and a report."""
    input_audit = audit_reference(motion, times, contract)
    rejected: list[dict] = []
    report: dict[str, Any] = {"schema_version": "sports-xy-repair/1.0", "status": "rejected", "config": asdict(config),
                              "input_audit": input_audit, "rejected_anchors": rejected,
                              "physical_rollout": "not_run", "motion_quality_validated": False,
                              "changed_fields": ["root_x", "root_y"], "transformations": [],
                              "implementation": implementation_hashes()}
    def fail(reason: str) -> RepairResult:
        report["reason"] = reason
        return RepairResult(None, report)
    if input_audit["status"] != "passed":
        return fail("input_reference_rejected")
    if not all(np.isfinite(v) and v > 0 for v in [config.max_anchor_gap_s, config.huber_delta_m, config.lambda_prior, config.max_correction_m]):
        raise ContractError("gap, robust threshold, prior and correction bound must be finite and positive")
    if (not np.isfinite(config.min_confidence) or not 0 <= config.min_confidence <= 1
            or not np.isfinite(config.lambda_smooth) or config.lambda_smooth < 0
            or not np.isfinite(config.lambda_contact) or config.lambda_contact < 0 or config.max_iterations < 1):
        raise ContractError("invalid repair configuration")
    if anchors_doc.get("schema_version") != "sports-anchors/1.0":
        return fail("anchor_schema_version_mismatch")
    same_frame = anchors_doc.get("world_frame_id") == contract["world_frame_id"]
    same_repr = anchors_doc.get("representation") == contract["representation"]
    transform = np.eye(3)
    explicit = anchors_doc.get("coordinate_transform")
    if explicit is not None:
        if not isinstance(explicit, dict):
            return fail("coordinate_transform_must_be_an_object")
        for key, expected in [("from_world_frame_id", anchors_doc.get("world_frame_id")), ("to_world_frame_id", contract["world_frame_id"]),
                              ("from_representation", anchors_doc.get("representation")), ("to_representation", contract["representation"])]:
            if explicit.get(key) != expected:
                return fail(f"coordinate_transform_{key}_mismatch")
        if not explicit.get("evidence"):
            return fail("coordinate_transform_requires_evidence")
        transform = _matrix3(explicit.get("matrix"), "coordinate_transform", affine=True)
        report["transformations"].append(explicit)
    elif not same_frame or not same_repr:
        return fail("anchor_frame_or_representation_mismatch_requires_explicit_transform")
    calibrations = anchors_doc.get("calibrations", {})
    if not isinstance(calibrations, dict) or not isinstance(anchors_doc.get("anchors"), list):
        return fail("calibrations_must_be_object_and_anchors_must_be_array")
    accepted = []
    seen_ids = set()
    seen_observations = set()
    for ordinal, anchor in enumerate(anchors_doc.get("anchors", [])):
        if not isinstance(anchor, dict):
            rejected.append({"anchor_id": f"row:{ordinal}", "reason": "anchor_must_be_an_object"})
            continue
        anchor_id = anchor.get("anchor_id", f"row:{ordinal}")
        try:
            if not isinstance(anchor_id, str) or not anchor_id:
                raise ContractError("anchor_id_must_be_a_nonempty_string")
            if anchor_id in seen_ids:
                raise ContractError("duplicate_anchor_id")
            seen_ids.add(anchor_id)
            idx = anchor.get("frame_index")
            if isinstance(idx, bool) or not isinstance(idx, int) or not 0 <= idx < len(motion):
                raise ContractError("invalid_frame_index")
            if anchor.get("foot") not in {"left", "right"}:
                raise ContractError("foot_must_be_left_or_right")
            confidence = float(anchor.get("confidence", -1))
            if not np.isfinite(confidence) or not config.min_confidence <= confidence <= 1:
                raise ContractError("insufficient_or_invalid_confidence")
            if anchor.get("role") not in {"fit", "holdout"}:
                raise ContractError("role_must_be_fit_or_holdout")
            observation_key = (idx, anchor["foot"], anchor["role"])
            if observation_key in seen_observations:
                raise ContractError("duplicate_frame_foot_role_observation")
            observed = _project(transform, project_ground_anchor(anchor, calibrations))
            offset = _xy(anchor.get("reference_foot_offset_xy"), "reference_foot_offset_xy")
            seen_observations.add(observation_key)
            accepted.append({**anchor, "anchor_id": anchor_id, "observed_xy": observed,
                             "offset_xy": offset, "confidence": confidence})
        except (ContractError, TypeError, ValueError) as exc:
            rejected.append({"anchor_id": anchor_id, "reason": str(exc), "source_evidence": anchor.get("source_evidence")})
    fit_keys = {(a["frame_index"], a["foot"]) for a in accepted if a["role"] == "fit"}
    usable = []
    for anchor in accepted:
        if anchor["role"] == "holdout" and (anchor["frame_index"], anchor["foot"]) in fit_keys:
            rejected.append({"anchor_id": anchor["anchor_id"], "reason": "holdout_overlaps_fit_frame_and_foot"})
        else:
            usable.append(anchor)
    fit = [a for a in usable if a["role"] == "fit"]
    mask = _support(times, [a["frame_index"] for a in fit], config.max_anchor_gap_s)
    report.update({"candidate_anchor_count": len(anchors_doc.get("anchors", [])), "valid_fit_count": len(fit),
                   "valid_holdout_count": sum(a["role"] == "holdout" for a in usable),
                   "supported_intervals": _intervals(mask, times), "unsupported_intervals": _intervals(~mask, times),
                   "support_policy": "between_at_least_two_fit_times_without_gaps_above_max_anchor_gap; zero_correction_elsewhere",
                   "calibrations": calibrations})
    if not mask.any():
        return fail("fewer_than_two_connected_fit_times; no_supported_interval")
    active = np.flatnonzero(mask)
    fit = [a for a in fit if mask[a["frame_index"]]]
    report["used_fit_count"] = len(fit)
    n = len(motion)
    frame_ids = np.array([a["frame_index"] for a in fit], dtype=int)
    weights = np.array([a["confidence"] for a in fit])
    desired = np.stack([a["observed_xy"] - motion[a["frame_index"], :2] - a["offset_xy"] for a in fit])
    design = sparse.csr_matrix((np.ones(len(fit)), (np.arange(len(fit)), frame_ids)), shape=(len(fit), n))[:, active]
    dt = np.diff(times)
    duration_weights = np.r_[dt[0] / 2, (dt[:-1] + dt[1:]) / 2, dt[-1] / 2]
    reg = sparse.diags(config.lambda_prior * duration_weights[active], format="csc")
    if n > 2 and config.lambda_smooth:
        scale = 2 / (dt[:-1] + dt[1:])
        rows = np.repeat(np.arange(n - 2), 3)
        cols = np.column_stack((np.arange(n - 2), np.arange(1, n - 1), np.arange(2, n))).ravel()
        coefficients = np.column_stack((scale / dt[:-1], -scale * (1 / dt[:-1] + 1 / dt[1:]), scale / dt[1:]))
        coefficients *= np.sqrt((dt[:-1] + dt[1:]) / 2)[:, None]
        smooth = sparse.csr_matrix((coefficients.ravel(), (rows, cols)), shape=(n - 2, n))[:, active]
        reg = reg + config.lambda_smooth * smooth.T @ smooth
    contacts, contact_targets = [], []
    by_stance: dict[tuple, dict[int, dict]] = {}
    for anchor in fit:
        if anchor.get("contact_segment_id"):
            key = (anchor["foot"], anchor["contact_segment_id"])
            by_stance.setdefault(key, {})[anchor["frame_index"]] = anchor
    for frames in by_stance.values():
        for idx in sorted(frames):
            if idx + 1 in frames:
                a, b = frames[idx], frames[idx + 1]
                weight = min(a["confidence"], b["confidence"])
                coefficient = math.sqrt(weight * dt[idx]) / dt[idx]
                contacts.append((idx, coefficient))
                contact_targets.append(-coefficient * (motion[idx + 1, :2] + b["offset_xy"] - motion[idx, :2] - a["offset_xy"]))
    contact_rhs = np.zeros((len(active), 2))
    if contacts and config.lambda_contact:
        rows = np.repeat(np.arange(len(contacts)), 2)
        cols = np.array([[i, i + 1] for i, _ in contacts]).ravel()
        vals = np.array([[-w, w] for _, w in contacts]).ravel()
        C = sparse.csr_matrix((vals, (rows, cols)), shape=(len(contacts), n))[:, active]
        reg = reg + config.lambda_contact * C.T @ C
        contact_rhs = config.lambda_contact * C.T @ np.array(contact_targets)
    delta_active = np.zeros((len(active), 2))
    converged = False
    for iteration in range(config.max_iterations):
        residual = np.linalg.norm(design @ delta_active - desired, axis=1)
        robust = np.minimum(1, config.huber_delta_m / np.maximum(residual, 1e-12))
        W = sparse.diags(weights * robust)
        lhs = (design.T @ W @ design + reg).tocsc()
        rhs = design.T @ W @ desired + contact_rhs
        updated = spsolve(lhs, rhs)
        if not np.isfinite(updated).all():
            return fail("nonfinite_solver_output")
        change = float(np.max(np.abs(updated - delta_active)))
        delta_active = updated
        if change < 1e-8:
            converged = True
            break
    delta = np.zeros((n, 2))
    delta[active] = delta_active
    maximum = float(np.linalg.norm(delta, axis=1).max())
    report.update({"iterations": iteration + 1, "converged": converged, "stance_velocity_pairs": len(contacts),
                   "max_correction_m": maximum, "delta_xy_array_sha256": array_sha256(delta),
                   "corrected_frame_count": len(active)})
    if not converged:
        return fail("irls_did_not_converge_within_iteration_budget")
    if maximum > config.max_correction_m:
        return fail("correction_exceeds_declared_bound; rejected_not_clipped")
    corrected = motion.copy()
    corrected[:, :2] += delta
    # Invariants are exact, not a tolerance-based promise.
    if not np.array_equal(corrected[:, 2:], motion[:, 2:]):
        raise AssertionError("XY correction changed a forbidden field")
    report["invariants"] = {"root_z_unchanged": True, "orientation_unchanged": True, "joint_positions_unchanged": True,
                            "timestamps_unchanged": True, "adaptation_unchanged": True, "unsupported_frames_unchanged": True}
    residual_rows = []
    for anchor in usable:
        idx = anchor["frame_index"]
        before = motion[idx, :2] + anchor["offset_xy"] - anchor["observed_xy"]
        after = corrected[idx, :2] + anchor["offset_xy"] - anchor["observed_xy"]
        residual_rows.append({"anchor_id": anchor["anchor_id"], "frame_index": idx, "role": anchor["role"],
                              "supported": bool(mask[idx]), "used_in_solver": anchor["role"] == "fit" and bool(mask[idx]),
                              "before_m": float(np.linalg.norm(before)),
                              "after_m": float(np.linalg.norm(after)), "source_evidence": anchor["source_evidence"]})
    report["anchor_residuals"] = residual_rows
    for role in ("fit", "holdout"):
        subset = [a for a in residual_rows if a["role"] == role]
        report[f"{role}_error"] = {"count": len(subset), "before_rmse_m": float(np.sqrt(np.mean([a["before_m"] ** 2 for a in subset]))) if subset else None,
                                   "after_rmse_m": float(np.sqrt(np.mean([a["after_m"] ** 2 for a in subset]))) if subset else None}
    report["output_audit"] = audit_reference(corrected, times, contract)
    report["status"] = "passed" if report["output_audit"]["status"] == "passed" else "rejected"
    report["interpretation"] = "Numerical XY transformation only; fit residual is not independent validation and no physical success is claimed."
    return RepairResult(corrected if report["status"] == "passed" else None, report)


def probe_video(path: str | Path, ffprobe: str = "ffprobe") -> dict:
    command = [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_streams", "-show_frames",
               "-show_entries", "stream=time_base,avg_frame_rate,r_frame_rate,width,height:frame=best_effort_timestamp_time,pts_time",
               "-of", "json", str(path)]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    data = json.loads(completed.stdout)
    if len(data.get("streams", [])) != 1:
        raise ContractError("Expected one selected video stream")
    times = []
    for frame in data.get("frames", []):
        pts = frame.get("pts_time", frame.get("best_effort_timestamp_time"))
        if pts is None:
            raise ContractError("A video frame has no PTS; nominal FPS is not a replacement")
        times.append(float(pts))
    values = np.asarray(times, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all() or not np.all(np.diff(values) > 0):
        raise ContractError("Video PTS are missing, nonfinite or nonmonotonic")
    stream = data["streams"][0]
    def rate(value: str | None) -> float | None:
        try:
            return float(Fraction(value or "0")) or None
        except (ValueError, ZeroDivisionError):
            return None
    return {"schema_version": "sports-video-pts/1.0", "video_path": str(Path(path).resolve()),
            "video_sha256": sha256_file(path), "stream": stream, "timestamps_s": times,
            "time_origin_pts_s": times[0], "frame_count": len(times), "duration_first_to_last_pts_s": times[-1] - times[0],
            "effective_fps": (len(times) - 1) / (times[-1] - times[0]), "nominal_avg_fps": rate(stream.get("avg_frame_rate")),
            "provenance": "source_pts", "resampled": False, "probe_command": command,
            "implementation": implementation_hashes()}


def _write_json(path: Path, value: dict, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if overwrite else "x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    probe = sub.add_parser("probe-video", help="Extract real per-frame PTS with ffprobe; never infer timing from 30 FPS")
    probe.add_argument("video", type=Path)
    probe.add_argument("--ffprobe", default="ffprobe")
    probe.add_argument("--output", type=Path, required=True)
    probe.add_argument("--overwrite", action="store_true")
    for name in ("audit", "repair-xy"):
        command = sub.add_parser(name)
        command.add_argument("motion", type=Path)
        command.add_argument("--contract", type=Path, required=True)
        command.add_argument("--allow-trusted-pickle", action="store_true")
        command.add_argument("--overwrite", action="store_true")
        if name == "audit":
            command.add_argument("--output", type=Path, required=True)
        else:
            command.add_argument("--anchors", type=Path, required=True)
            command.add_argument("--output-dir", type=Path, required=True)
            command.add_argument("--min-confidence", type=float, default=0.8)
            command.add_argument("--max-anchor-gap-s", type=float, default=2.0)
            command.add_argument("--huber-delta-m", type=float, default=0.05)
            command.add_argument("--lambda-smooth", type=float, default=0.01)
            command.add_argument("--lambda-prior", type=float, default=0.001)
            command.add_argument("--lambda-contact", type=float, default=0.01)
            command.add_argument("--max-correction-m", type=float, default=2.0)
    args = parser.parse_args(argv)
    output = args.output if args.command != "repair-xy" else args.output_dir / "repair-report.json"
    try:
        if args.command == "probe-video":
            if output.resolve() == args.video.resolve():
                raise ContractError("output may not overwrite source video")
            result = probe_video(args.video, args.ffprobe)
            _write_json(output, result, args.overwrite)
            print(json.dumps({"status": "passed", "frame_count": result["frame_count"], "output": str(output)}))
            return 0
        contract = read_json(args.contract)
        validate_contract(contract)
        # Resolve sidecar paths against the sidecar, including archives in another folder.
        resolved = copy.deepcopy(contract)
        if "timestamps_path" in resolved["time"]:
            resolved["time"]["timestamps_path"] = str((args.contract.parent / resolved["time"]["timestamps_path"]).resolve())
        protected_inputs = {args.motion.resolve(), args.contract.resolve()}
        if "timestamps_path" in resolved["time"]:
            protected_inputs.add(Path(resolved["time"]["timestamps_path"]).resolve())
        if args.command == "repair-xy":
            protected_inputs.add(args.anchors.resolve())
        if output.resolve() in protected_inputs:
            raise ContractError("output report may not overwrite an input artifact")
        motion = load_motion(args.motion, resolved, allow_trusted_pickle=args.allow_trusted_pickle)
        times = load_timestamps(resolved, len(motion))
        provenance = {"motion": str(args.motion.resolve()), "motion_sha256": sha256_file(args.motion),
                      "contract": str(args.contract.resolve()), "contract_sha256": sha256_file(args.contract)}
        if args.command == "audit":
            report = audit_reference(motion, times, resolved)
            report["input_files"] = provenance
            _write_json(output, report, args.overwrite)
            print(json.dumps({"status": report["status"], "output": str(output)}))
            return 0 if report["status"] == "passed" else 2
        anchors = read_json(args.anchors)
        config = RepairConfig(**{key: getattr(args, key) for key in asdict(RepairConfig()) if key != "max_iterations"})
        repair = repair_root_xy(motion, times, resolved, anchors, config)
        repair.report["input_files"] = {**provenance, "anchors": str(args.anchors.resolve()), "anchors_sha256": sha256_file(args.anchors)}
        if repair.motion is not None:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            targets = [args.output_dir / name for name in ("repaired.csv", "repaired.contract.json", "repair-report.json")]
            if not args.overwrite and any(path.exists() for path in targets):
                raise ContractError("output already exists; use a new directory or explicit --overwrite")
            if protected_inputs.intersection(path.resolve() for path in targets):
                raise ContractError("outputs may not overwrite source motion, contract, anchors or timestamps")
            # Seventeen significant digits round-trip float64 and preserve all non-XY values.
            np.savetxt(targets[0], repair.motion, delimiter=",", fmt="%.17g")
            reread = np.loadtxt(targets[0], delimiter=",", ndmin=2)
            if not np.array_equal(reread[:, 2:], motion[:, 2:]):
                raise AssertionError("export modified a forbidden motion field")
            new_contract = copy.deepcopy(contract)
            new_contract["time"].pop("timestamps_path", None)
            new_contract["time"].pop("source_timestamps_sha256", None)
            new_contract["time"]["timestamps_s"] = times.tolist()
            new_contract["motion_sha256"] = sha256_file(targets[0])
            new_contract["parent_motion_sha256"] = provenance["motion_sha256"]
            new_contract["processing"] = {"operation": "root_xy_only", "anchors_sha256": sha256_file(args.anchors), "report": "repair-report.json"}
            _write_json(targets[1], new_contract, args.overwrite)
            repair.report["output_files"] = {"motion": str(targets[0]), "motion_sha256": sha256_file(targets[0]),
                                              "contract": str(targets[1]), "contract_sha256": sha256_file(targets[1])}
        _write_json(output, repair.report, args.overwrite)
        print(json.dumps({"status": repair.report["status"], "output": str(output)}))
        return 0 if repair.motion is not None else 2
    except (ContractError, OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        failure = {"schema_version": "sports-reference-error/1.0", "status": "rejected", "reason": str(exc),
                   "physical_rollout": "not_run", "motion_quality_validated": False}
        # Do not replace an existing report merely because the attempted operation failed.
        if not output.exists():
            _write_json(output, failure, False)
        print(json.dumps(failure, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
