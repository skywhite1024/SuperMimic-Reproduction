"""Prepare a real-PTS 30 Hz clip and fixed-identity boxes; no model inference.

This adapter never chooses a replacement person. Optional offline crop proposals
retain the original rejected identities/keypoints and are never contact/3-D GT.
Only numpy and an FFmpeg executable are required. Original inputs are read-only.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def probe_pts(ffmpeg, path):
    cmd = [ffmpeg, "-hide_banner", "-copyts", "-i", str(path), "-map", "0:v:0",
           "-vf", "showinfo", "-fps_mode", "passthrough", "-f", "null", "-"]
    run = subprocess.run(cmd, capture_output=True, text=True, check=True)
    tb = re.search(r"config in time_base:\s*(\d+/\d+)", run.stderr)
    if not tb:
        raise ValueError("FFmpeg did not report the decoded video time base")
    frames = re.findall(r"\bn:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:.*?\bs:(\d+)x(\d+)", run.stderr)
    if len(frames) < 2 or [int(x[0]) for x in frames] != list(range(len(frames))):
        raise ValueError("Missing, duplicated or out-of-order decoded frames")
    if len({(x[2], x[3]) for x in frames}) != 1:
        raise ValueError("Changing video dimensions require a shot split")
    pts = [int(x[1]) for x in frames]
    t = np.asarray([float(Fraction(p) * Fraction(tb[1])) for p in pts])
    if not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Nonmonotonic or invalid source PTS")
    return {"time_base": tb[1], "pts": pts, "timestamps_s": t.tolist(),
            "width": int(frames[0][2]), "height": int(frames[0][3]), "probe_command": cmd}


def crop_proposals(records, times, target_id, offline=False):
    """Conservative offline bbox-only proposals; never fill keypoints/contacts."""
    proposals = []
    for i, r in enumerate(records):
        valid = not r["identity_rejected"] and r.get("target") is not None
        proposals.append({"source_frame_index": i, "original_gate_accepted": bool(valid),
            "original_rejection_reasons": r.get("identity_rejection_reasons", []),
            "identity_confirmation": "online_gate_accepted" if valid else "rejected",
            "bbox_xyxy": r["target"]["bbox_xyxy"] if valid else None,
            "bbox_confidence": r["target"]["bbox_confidence"] if valid else None,
            "observed_track_id": r["target"]["track_id"] if valid else None,
            "source_observation_indices": [i] if valid else [], "bbox_interpolated": False,
            "keypoints_filled": False, "contact_label": "unknown", "ground_anchor_allowed": False})
    if not offline:
        return proposals
    confirmed = [i for i, r in enumerate(records) if proposals[i]["original_gate_accepted"]
                 and r["target"]["track_id"] == target_id]
    if confirmed:
        first = confirmed[0]
        subsequent = records[first:first+3]
        stable = len(subsequent) == 3 and all(not r["identity_rejected"] and r.get("target")
                   and r["target"]["track_id"] == target_id for r in subsequent)
        if first <= 2 and stable:
            for i in range(first):
                r = records[i]
                observations = [x for x in r["all_detected_tracks"] if x["track_id"] == target_id and x["bbox_confidence"] >= .35]
                if r["identity_rejection_reasons"] == ["initial_identity_unconfirmed"] and len(observations) == 1:
                    obs = observations[0]
                    proposals[i].update(identity_confirmation="offline_retrospective_same_track",
                        bbox_xyxy=obs["bbox_xyxy"], bbox_confidence=obs["bbox_confidence"],
                        observed_track_id=target_id, source_observation_indices=[i],
                        later_confirmation_frame_indices=list(range(first, first+3)))
    for i in range(1, len(records)-1):
        if proposals[i]["bbox_xyxy"] is not None:
            continue
        prev, nxt = records[i-1], records[i+1]
        if records[i]["identity_rejection_reasons"] != ["locked_track_id_missing_or_ambiguous"]:
            continue
        # Both endpoints must be original accepted observations, never earlier proposals.
        if not all(not x["identity_rejected"] and x.get("target") and x["target"]["track_id"] == target_id for x in [prev, nxt]):
            continue
        if times[i+1]-times[i-1] > .12:
            continue
        b0, b1 = [np.asarray(x["target"]["bbox_xyxy"], dtype=float) for x in [prev, nxt]]
        center_displacement = np.linalg.norm((b1[:2]+b1[2:]-b0[:2]-b0[2:])/2)
        if center_displacement > .75*max(b0[3]-b0[1], b1[3]-b1[1]):
            continue
        alpha = float((times[i]-times[i-1])/(times[i+1]-times[i-1]))
        proposals[i].update(identity_confirmation="interpolated_proposal", bbox_xyxy=((1-alpha)*b0+alpha*b1).tolist(),
            observed_track_id=None, proposed_track_id=target_id, source_observation_indices=[i-1,i+1],
            bbox_interpolated=True, interpolation_alpha=alpha, interpolation_basis="linear bbox coordinates at source PTS",
            endpoint_center_displacement_px=float(center_displacement))
    return proposals


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--frames-jsonl", type=Path, required=True)
    p.add_argument("--pose-summary", type=Path, required=True, help="The matching per-clip summary.json")
    p.add_argument("--ffmpeg", required=True)
    p.add_argument("--start-frame", type=int, default=0, help="Inclusive original clip index; no automatic trimming")
    p.add_argument("--end-frame-exclusive", type=int)
    p.add_argument("--max-source-gap-s", type=float, default=0.1)
    p.add_argument("--allow-partial-identity", action="store_true", help="Retain rejected frames as NaN bbox with explicit partial/not-model-ready receipt")
    p.add_argument("--offline-crop-proposals", action="store_true", help="Use retrospective same-ID initial detections and isolated one-frame bbox interpolation; original gates/keypoints retained")
    p.add_argument("--write-bbx-pt", action="store_true", help="Write actual torch cache using pinned official bbox conversion functions; requires torch")
    p.add_argument("--gvhmr-source-root", type=Path, help="Explicit pinned GVHMR checkout; required with --write-bbx-pt")
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    if a.write_bbx_pt and a.gvhmr_source_root is None:
        p.error("--write-bbx-pt requires --gvhmr-source-root")
    video, frames_path, summary_path, out = [x.resolve() for x in (a.video, a.frames_jsonl, a.pose_summary, a.output_dir)]
    if out.exists():
        raise ValueError("Output directory already exists; use a fresh path")
    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    if sha(video) != summary["source_sha256"]:
        raise ValueError("Pose summary does not belong to this exact video")
    records = [json.loads(line) for line in frames_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    observed = probe_pts(a.ffmpeg, video)
    times = np.asarray(observed["timestamps_s"])
    if len(records) != len(times) or len(records) != summary["frames"]:
        raise ValueError("Video / pose / summary frame counts disagree")
    if [x["frame_index"] for x in records] != list(range(len(records))):
        raise ValueError("Pose frames must be a contiguous original-index ledger")
    declared = np.asarray([x["time_seconds"] for x in records])
    if not np.allclose(declared, times-times[0], rtol=0, atol=1e-5):
        raise ValueError("Pose frame times disagree with decoded original video PTS")
    end = len(records) if a.end_frame_exclusive is None else a.end_frame_exclusive
    start = a.start_frame
    if not (0 <= start < end <= len(records)) or end-start < 2:
        raise ValueError("Selected source interval requires at least two valid frames")
    if not np.isfinite(a.max_source_gap_s) or a.max_source_gap_s <= 0:
        raise ValueError("max-source-gap-s must be finite and positive")
    if np.max(np.diff(times[start:end])) > a.max_source_gap_s:
        raise ValueError("Large source PTS gap; split or review this interval")
    proposals = crop_proposals(records, times, summary["locked_target_id"], offline=a.offline_crop_proposals)
    ids = set()
    rejected_source = []
    for r in records[start:end]:
        if r["identity_rejected"] or r.get("target") is None:
            rejected_source.append({"source_frame_index": r["frame_index"], "source_pts_s": float(times[r["frame_index"]]),
                                    "reasons": r.get("identity_rejection_reasons", ["target_missing"])})
            if proposals[r["frame_index"]]["bbox_xyxy"] is None and not a.allow_partial_identity:
                raise ValueError(f"Identity-rejected source frame {r['frame_index']}; explicit review/split required")
            continue
        target = r["target"]
        ids.add(target["track_id"])
        if target["track_id"] != r["locked_target_track_id"]:
            raise ValueError("Target identity does not match the frozen track")
        b = np.asarray(target["bbox_xyxy"], dtype=float)
        if b.shape != (4,) or not np.isfinite(b).all() or np.any(b[2:] <= b[:2]):
            raise ValueError("Invalid target bounding box")
    if len(ids) != 1 or next(iter(ids)) != summary["locked_target_id"]:
        raise ValueError("Selected interval has no single frozen target identity")
    duration = times[end-1]-times[start]
    target_t = times[start] + np.arange(int(np.floor(duration*30+1e-9))+1)/30
    # Nearest original decoded frame, earlier frame on ties; boxes use the same index.
    candidates = times[start:end]
    hi = np.searchsorted(candidates, target_t).clip(0, len(candidates)-1)
    lo = np.maximum(hi-1, 0)
    chosen = np.where(abs(candidates[lo]-target_t) <= abs(candidates[hi]-target_t), lo, hi)+start
    crop = float(times[end-1]-target_t[-1])
    if crop < -1e-8 or crop >= 1/30+1e-8:
        raise ValueError("Invalid endpoint crop")
    original_accepted = np.asarray([not records[i]["identity_rejected"] and records[i].get("target") is not None for i in chosen])
    accepted = np.asarray([proposals[i]["bbox_xyxy"] is not None for i in chosen])
    boxes = np.full((len(chosen), 4), np.nan, dtype=np.float32)
    keypoints = np.full((len(chosen), 17, 3), np.nan, dtype=np.float32)
    for j, i in enumerate(chosen):
        if accepted[j]:
            boxes[j] = proposals[i]["bbox_xyxy"]
        if original_accepted[j]:
            keypoints[j] = records[i]["target"]["keypoints_xy_confidence"]
    if a.write_bbx_pt:
        import ast
        import torch
        helper_source = a.gvhmr_source_root.resolve()/"hmr4d/utils/geo/hmr_cam.py"
        parsed = ast.parse(helper_source.read_text(encoding="utf-8"))
        funcs = [node for node in parsed.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"get_bbx_xys", "get_bbx_xys_from_xyxy"}]
        if len(funcs) != 2:
            raise ValueError("Official bbox helper functions unavailable")
        scope = {"torch": torch}
        exec(compile(ast.Module(body=funcs, type_ignores=[]), str(helper_source), "exec"), scope)
    out.mkdir(parents=True)
    output_video = out/"input_30hz.mp4"
    w, h = observed["width"], observed["height"]
    decode_cmd = [a.ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(video), "-map", "0:v:0",
                  "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    encode_cmd = [a.ffmpeg, "-hide_banner", "-loglevel", "error", "-n", "-f", "rawvideo", "-pix_fmt", "rgb24",
                  "-s", f"{w}x{h}", "-r", "30", "-i", "pipe:0", "-an", "-c:v", "libx264", "-crf", "18",
                  "-preset", "fast", "-pix_fmt", "yuv420p", str(output_video)]
    # Read all source frames to verify decode count. No frame synthesis or endpoint padding.
    with (out/"decode.log").open("wb") as dl, (out/"encode.log").open("wb") as el:
        dec = subprocess.Popen(decode_cmd, stdout=subprocess.PIPE, stderr=dl)
        enc = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE, stderr=el)
        next_out = 0
        try:
            for i in range(len(times)):
                frame = dec.stdout.read(w*h*3)
                if len(frame) != w*h*3:
                    raise ValueError(f"Decoder stopped at source frame {i}")
                while next_out < len(chosen) and chosen[next_out] == i:
                    enc.stdin.write(frame)
                    next_out += 1
            if dec.stdout.read(1) or next_out != len(chosen):
                raise ValueError("Unexpected decode count or incomplete target mapping")
            enc.stdin.close()
            dec.stdout.close()
            decode_code, encode_code = dec.wait(), enc.wait()
            if decode_code != 0 or encode_code != 0:
                raise ValueError("FFmpeg decode/encode failure; see logs")
        except BaseException:
            dec.kill()
            enc.kill()
            dec.wait()
            enc.wait()
            raise
    encoded = probe_pts(a.ffmpeg, output_video)
    local_t = target_t-times[start]
    if len(encoded["pts"]) != len(local_t) or not np.allclose(encoded["timestamps_s"], local_t, rtol=0, atol=1e-6):
        raise ValueError("Encoded video does not have the requested exact 30 Hz frame grid")
    np.savez(out/"fixed_identity.npz", bbx_xyxy=boxes, source_keypoints_coco17=keypoints,
             source_frame_index=chosen, timestamps_s=local_t, original_video_timestamps_s=target_t,
             identity_valid=original_accepted, crop_bbox_usable=accepted,
             bbox_interpolated=np.asarray([proposals[i]["bbox_interpolated"] for i in chosen]))
    if a.write_bbx_pt:
        bbox_tensor = torch.from_numpy(boxes)
        torch.save({"bbx_xyxy": bbox_tensor,
                    "bbx_xys": scope["get_bbx_xys_from_xyxy"](bbox_tensor, base_enlarge=1.2),
                    "identity_valid": torch.from_numpy(original_accepted),
                    "crop_bbox_usable": torch.from_numpy(accepted),
                    "bbox_interpolated": torch.tensor([proposals[i]["bbox_interpolated"] for i in chosen]),
                    "source_frame_index": torch.from_numpy(chosen),
                    "timestamps_s": torch.from_numpy(local_t),
                    "model_ready": bool(accepted.all()),
                    "rejected_bbox_encoding": "NaN if unresolved; offline proposals are crop-only, never identity/contact/3D ground truth"}, out/"bbx.pt")
    mapping = [{"frame_index": j, "time_seconds": float(local_t[j]),
                "original_video_time_seconds": float(target_t[j]), "source_frame_index": int(i),
                "source_frame_pts_seconds": float(times[i]), "sampling_error_seconds": float(times[i]-target_t[j]),
                "original_match_frame_index": records[i].get("source_video_frame"),
                "identity_valid": bool(original_accepted[j]), "original_gate_false": not bool(original_accepted[j]),
                "crop_bbox_usable": bool(accepted[j]), "identity_rejection_reasons": records[i].get("identity_rejection_reasons", []),
                "identity_confirmation": proposals[i]["identity_confirmation"],
                "bbox_proposal_source": proposals[i],
                "source_body_pose_usable": bool(records[i].get("target") and records[i]["target"].get("body_pose_usable")),
                "source_bbox_confidence": proposals[i]["bbox_confidence"],
                "bbox_interpolated": proposals[i]["bbox_interpolated"], "image_interpolated": False,
                "source_frame_repeated": bool(j > 0 and i == chosen[j-1])}
               for j, i in enumerate(chosen)]
    (out/"frame-map.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    (out/"identity-rejections.json").write_text(json.dumps({"source_rejections": rejected_source,
        "resampled_rejected_frames": [f for f in mapping if not f["identity_valid"]],
        "handling": "Original gates, rejected keypoints and contact unknown preserved; explicit offline bbox proposals may supply crops only",
        "offline_policy_enabled": a.offline_crop_proposals,
        "source_crop_proposals": [x for x in proposals[start:end] if not x["original_gate_accepted"]]}, indent=2), encoding="utf-8")
    with (out/"frame-contract.jsonl").open("w", encoding="utf-8") as f:
        for row in mapping:
            f.write(json.dumps(row)+"\n")
    manifest = {"schema_version": "gvhmr-fixed-identity-input/1.0", "status": "model_ready_except_weights" if accepted.all() else "partial_identity_not_model_ready",
        "model_ready_crop_only": bool(accepted.all()), "source_identity_rejected_frames": len(rejected_source),
        "target_identity_rejected_frames": int((~original_accepted).sum()), "unresolved_target_bbox_frames": int((~accepted).sum()),
        "identity_mask_key": "fixed_identity.npz:identity_valid", "crop_mask_key": "fixed_identity.npz:crop_bbox_usable",
        "offline_crop_policy_enabled": a.offline_crop_proposals,
        "offline_retrospective_source_frames": [x["source_frame_index"] for x in proposals[start:end] if x["identity_confirmation"] == "offline_retrospective_same_track"],
        "interpolated_source_bbox_frames": [x["source_frame_index"] for x in proposals[start:end] if x["bbox_interpolated"]],
        "complete_source_frame_interval_selected": start == 0 and end == len(records),
        "clip_id": summary["clip_id"], "source_video": str(video), "source_video_sha256": sha(video),
        "pose_frames_path": str(frames_path), "pose_frames_sha256": sha(frames_path),
        "pose_summary_path": str(summary_path), "pose_summary_sha256": sha(summary_path),
        "source_pts": observed, "encoded_pts": encoded, "target_fps": 30,
        "frame_count": len(chosen), "locked_target_id": next(iter(ids)),
        "selected_source_frames": [start, end], "selected_source_point_duration_s": float(duration),
        "target_point_duration_s": float(local_t[-1]), "crop_duration_s": crop,
        "discarded_prefix_duration_s": float(times[start]-times[0]),
        "discarded_suffix_duration_s": float(times[-1]-times[end-1]),
        "endpoint_policy": "last target PTS <= last selected source PTS; fractional tail cropped < 1/30 s; no endpoint padding",
        "sampling": "nearest decoded source PTS; earlier on ties; same source index for image and bbox",
        "duplicate_target_frames_from_upsampling": int(len(chosen)-len(np.unique(chosen))),
        "identity_review": "original online gates retained; offline bbox-only proposals explicit; no replacement ID, filled keypoints, ground-contact claim or manual identity GT",
        "keypoint_usage": "source YOLO COCO17 retained for audit only; baseline runs official ViTPose inside fixed bbox",
        "output_video": "input_30hz.mp4", "output_video_sha256": sha(output_video),
        "identity_npz": "fixed_identity.npz", "identity_npz_sha256": sha(out/"fixed_identity.npz"),
        "frame_map": "frame-map.json", "frame_map_sha256": sha(out/"frame-map.json"),
        "bbx_pt": "bbx.pt" if a.write_bbx_pt else None,
        "bbx_pt_sha256": sha(out/"bbx.pt") if a.write_bbx_pt else None,
        "bbox_helper_sha256": sha(helper_source) if a.write_bbx_pt else None,
        "identity_rejections_sha256": sha(out/"identity-rejections.json"),
        "decode_command": decode_cmd, "encode_command": encode_cmd,
        "script_sha256": sha(__file__)}
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out/"receipt.json").write_text(json.dumps({"status": manifest["status"], "manifest_sha256": sha(out/"manifest.json"),
        "source_video_sha256": sha(video), "output_video_sha256": sha(output_video),
        "source_frames": len(times), "target_frames": len(chosen), "identity_rejected_source": len(rejected_source),
        "identity_rejected_target": int((~original_accepted).sum()), "unresolved_target_bbox_frames": int((~accepted).sum()),
        "prefix_frames_removed": start, "suffix_frames_removed": len(records)-end,
        "bbox_interpolation": any(x["bbox_interpolated"] for x in proposals[start:end]),
        "image_interpolation": False, "nearest_pts_sampling": True,
        "inference_executed": False, "model_ready_crop_only": bool(accepted.all())}, indent=2), encoding="utf-8")
    print(json.dumps({k:manifest[k] for k in ("status","frame_count","selected_source_frames","crop_duration_s","locked_target_id")}, indent=2))


if __name__ == "__main__":
    main()
