"""Headless research adapter for the pinned official GVHMR checkout.

No model downloads, largest-box selection, keypoint substitution, or rendering.
Uses official preprocessing/prediction with a pre-bound fixed-identity bbox cache.
Respect GVHMR's research/non-commercial license and each model's terms.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

PIN = "ee960bb6e2ea2d381aa97f08e9b71ef320b624b1"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True, help="Explicit pinned official GVHMR checkout")
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--camera-mode", choices=["simplevo", "static"], default="simplevo")
    p.add_argument("--static-camera-evidence", type=Path, help="Required review record if selecting static mode")
    p.add_argument("--f-mm", type=float, help="Optional documented focal-length estimate; absence uses upstream estimate")
    a = p.parse_args()
    source, prepared, out = [x.resolve() for x in (a.source_root, a.prepared, a.output_dir)]
    if out.exists():
        raise ValueError("Output directory exists; prevent stale preprocessing-cache reuse")
    if subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip() != PIN:
        raise ValueError("GVHMR commit differs from the audited pinned source")
    subprocess.run(["git", "-C", str(source), "diff", "--exit-code", "HEAD", "--"], check=True, capture_output=True)
    manifest_path = prepared/"manifest.json"
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    if m["schema_version"] != "gvhmr-fixed-identity-input/1.0" or m["target_fps"] != 30:
        raise ValueError("A prepared 30 Hz, source-PTS input is required")
    if m.get("status") == "partial_identity_not_model_ready" or m.get("model_ready_crop_only") is False:
        raise ValueError("Prepared input contains unresolved crop frames; see identity-rejections.json; no implicit fallback at inference")
    for key, hash_key in [("output_video", "output_video_sha256"), ("identity_npz", "identity_npz_sha256"), ("frame_map", "frame_map_sha256")]:
        if sha(prepared/m[key]) != m[hash_key]:
            raise ValueError(f"Prepared input hash mismatch: {key}")
    z = np.load(prepared/m["identity_npz"], allow_pickle=False)
    boxes = z["bbx_xyxy"]
    if boxes.shape != (m["frame_count"], 4) or not np.isfinite(boxes).all():
        raise ValueError("Invalid fixed-identity bbox tensor")
    if not np.allclose(z["timestamps_s"], np.arange(len(boxes))/30, rtol=0, atol=1e-7):
        raise ValueError("Prepared times do not follow a 30 Hz grid")
    evidence = None
    if a.camera_mode == "static":
        if a.static_camera_evidence is None or not a.static_camera_evidence.is_file():
            raise ValueError("Static-camera mode needs an explicit camera review record")
        evidence = {"path": str(a.static_camera_evidence.resolve()), "sha256": sha(a.static_camera_evidence)}
    required = ["inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz",
                "inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt",
                "inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt",
                "inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth"]
    missing = [x for x in required if not (source/x).is_file()]
    if missing:
        raise FileNotFoundError("User-supplied, appropriately licensed model files required: " + ", ".join(missing))
    os.chdir(source)
    sys.path.insert(0, str(source))
    import torch
    import hydra
    from hydra import compose, initialize_config_module
    from tools.demo import demo
    if not torch.cuda.is_available():
        raise RuntimeError("Official GVHMR inference requires a CUDA-capable configured environment")
    with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
        demo.register_store_gvhmr()
        cfg = compose(config_name="demo", overrides=["video_name=fixed_identity"])
    cfg.output_dir = str(out)
    cfg.preprocess_dir = str(out/"preprocess")
    cfg.video_path = str(prepared/m["output_video"])
    cfg.static_cam = a.camera_mode == "static"
    cfg.use_dpvo = False
    cfg.verbose = False
    cfg.f_mm = a.f_mm
    (out/"preprocess").mkdir(parents=True)
    bbox = torch.from_numpy(boxes).float()
    torch.save({"bbx_xyxy": bbox, "bbx_xys": demo.get_bbx_xys_from_xyxy(bbox, base_enlarge=1.2).float()}, cfg.paths.bbx)

    def forbidden_person_selection(*args, **kwargs):
        raise RuntimeError("Person re-selection is forbidden: fixed identity bbox cache must be used")
    demo.Tracker.get_one_track = forbidden_person_selection
    record = {"schema_version": "gvhmr-fixed-identity-run/1.0", "status": "started",
              "source_commit": PIN, "prepared_manifest": str(manifest_path),
              "prepared_manifest_sha256": sha(manifest_path), "frame_count": len(boxes),
              "camera_mode": a.camera_mode, "static_camera_evidence": evidence, "f_mm": a.f_mm,
              "keypoints": "official ViTPose inside fixed identity bbox; no YOLO keypoint substitution",
              "crop_proposal_policy": {"offline_enabled": m.get("offline_crop_policy_enabled", False),
                  "original_source_gate_rejected_frames": m.get("source_identity_rejected_frames"),
                  "offline_retrospective_source_frames": m.get("offline_retrospective_source_frames", []),
                  "interpolated_source_bbox_frames": m.get("interpolated_source_bbox_frames", []),
                  "ground_anchor_allowed": False},
              "model_sha256": {x: sha(source/x) for x in required}, "torch_version": torch.__version__,
              "script_sha256": sha(__file__)}
    receipt = out/"inference-receipt.json"
    receipt.write_text(json.dumps(record, indent=2), encoding="utf-8")
    try:
        demo.run_preprocess(cfg)
        data = demo.load_data_dict(cfg)
        if int(data["length"]) != len(boxes):
            raise ValueError("Official decoder and prepared input frame counts disagree")
        model = hydra.utils.instantiate(cfg.model, _recursive_=False)
        model.load_pretrained_model(cfg.ckpt_path)
        model = model.eval().cuda()
        pred = demo.detach_to_cpu(model.predict(data, static_cam=cfg.static_cam))
        if len(pred["smpl_params_global"]["transl"]) != len(boxes):
            raise ValueError("GVHMR output length differs from its original time grid")
        torch.save(pred, cfg.paths.hmr4d_results)
        record.update(status="gvhmr_inference_complete_not_robot_evaluated", output_path=str(cfg.paths.hmr4d_results),
                      output_sha256=sha(cfg.paths.hmr4d_results))
    except BaseException as e:
        record.update(status="failed", error_type=type(e).__name__, error=str(e))
        raise
    finally:
        receipt.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
