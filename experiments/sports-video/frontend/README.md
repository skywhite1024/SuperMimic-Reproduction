# Sports-video frontend adapters

These original adapters connect a source video and an existing fixed-person tracking ledger to pinned GVHMR and GMR checkouts. Upstream code, robot assets, videos, pose data, and model weights are **not distributed in this directory**. All external inputs use explicit paths.

The preprocessing implementation was exercised on five real clips before this source transfer: 1,632 source frames at 25 Hz became 1,956 frames at 30 Hz, with source PTS and crop provenance retained. The source transfer changed only checkout/audit path arguments; verification after transfer is limited to `--help` and syntax compilation. It did not rerun preprocessing, 3D inference, IK, or policy evaluation.

## Files and dependencies

| Entry point | Purpose | Runtime dependencies |
|---|---|---|
| `prepare_gvhmr_identity.py` | Decode original PTS; prepare a 30 Hz video, fixed-person crop proposals, and per-frame provenance | NumPy, FFmpeg; PyTorch and an explicit GVHMR checkout for `--write-bbx-pt` |
| `run_gvhmr_fixed_identity.py` | Run official ViTPose/HMR2/GVHMR with the prepared crops; forbid person reselection | Pinned GVHMR environment, CUDA, separately supplied licensed models |
| `retarget_g1_headless.py` | Retarget every GVHMR frame to G1; export 36-column motion and a reference sidecar | Pinned GMR environment, `daqp`, licensed SMPL-X model, matching compiled mapping audit |
| `audit_g1_mapping.py` | Compare compiled GMR/runtime models, qpos order, axes, ranges, zero poses, and FK probes | MuJoCo, NumPy, external XML/assets, runtime joint constants |

[source-pins.json](source-pins.json) records the official source commits and file hashes:

- GVHMR: [`ee960bb6e2ea2d381aa97f08e9b71ef320b624b1`](https://github.com/zju3dv/GVHMR/tree/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1).
- GMR: [`25f56db79ac411258b377b6d4ccba0c8ff3372ac`](https://github.com/YanjieZe/GMR/tree/25f56db79ac411258b377b6d4ccba0c8ff3372ac).

Keep these checkouts outside this publication repository. GVHMR's official requirements target Linux Python 3.10, PyTorch 2.3.0 + cu121, and a Linux PyTorch3D wheel. Use a separate environment for GMR if its dependencies conflict; its default solver is `daqp`, while the pinned dependency list specifies `qpsolvers[proxqp]`, so install `qpsolvers[daqp]` explicitly. Save the resolved environment before inference: the upstream dependency lists are not complete locks. A working Windows CPU preprocessing environment does not establish CUDA inference readiness.

GVHMR's source license covers educational/research/nonprofit use; GMR uses MIT, and its Unitree assets carry a separate BSD license. Model licenses are separate. These adapters neither download models nor register accounts or accept model terms. They invoke external upstream code; no upstream implementation or assets are vendored here.

## Input contract and time

The pose input consists of a per-clip `summary.json` and one record for every decoded source frame in `frames.jsonl`. The summary binds `clip_id`, `source_sha256`, `frames`, and `locked_target_id`. Each frame records:

- `frame_index`, `time_seconds`, and optionally `source_video_frame` in the original match.
- `locked_target_track_id`, `identity_rejected`, and `identity_rejection_reasons`.
- `target`, when accepted: `track_id`, `bbox_xyxy`, `bbox_confidence`, `keypoints_xy_confidence` in COCO17 order, and `body_pose_usable`.
- `all_detected_tracks`: original `track_id`, `bbox_xyxy`, and `bbox_confidence` observations, needed for optional retrospective crop proposals.

The preparer checks the video hash, decoded frame count, and ledger times against FFmpeg's integer PTS and time base. Original inputs remain read-only. Each output directory must be new.

Do this resampling **before GVHMR**. The pinned upstream demo copies all input frames into a 30 fps writer, while the GMR loader declares 30 Hz; using all 25 Hz frames unchanged would accelerate the motion. Relabeling a later robot export cannot undo that error. The adapter samples the nearest source image at each 30 Hz timestamp and uses exactly the same source index for its crop. Interior duplicated images are recorded as normal upsampling. It does not interpolate RGB, stretch time, or pad the terminal frame to invent duration. The final target PTS does not exceed the last source PTS; any cropped tail is less than one 30 Hz step.

Use one common prepared input before the P0/P1 branches. Keep the E* evaluation references independently frozen. The adjacent `resample_reference.py` operates on correctly timestamped motion; it is not a substitute for this video preprocessing and should not resample these already-30-Hz exports again.

## Prepare a clip

Run from this directory. Replace the example paths with existing external files/checkouts; the Python interpreter must provide the dependencies for the stage.

```bash
python prepare_gvhmr_identity.py \
  --video /data/clips/example.mp4 \
  --frames-jsonl /data/pose2d/example/frames.jsonl \
  --pose-summary /data/pose2d/example/summary.json \
  --ffmpeg /usr/bin/ffmpeg \
  --gvhmr-source-root /external/GVHMR \
  --allow-partial-identity --offline-crop-proposals --write-bbx-pt \
  --output-dir /outputs/prepared/example_full_30hz
```

`--offline-crop-proposals` permits only the following explicit, auditable bbox operations:

1. The first at most two frames may use their **original same-ID detections** retrospectively if the next three original accepted frames confirm that ID. These crops are tagged `offline_retrospective_same_track`; the original false online gate remains false.
2. One isolated missing crop may be linearly interpolated using source PTS if both adjacent original accepted observations have the same ID, their span is at most 0.12 s, and their center displacement is at most 0.75 times the larger bbox height. The crop is tagged `interpolated_proposal` with endpoint indices and interpolation weight. Original RGB, missing keypoints, and unknown contacts are preserved.

Long gaps, replacement IDs, or unresolved crops remain rejected. `--allow-partial-identity` allows these frames to be retained as NaN crops in a `partial_identity_not_model_ready` output; the inference wrapper rejects that output. Without this flag, unresolved identity frames stop preparation. No largest-box fallback is used.

Outputs include `input_30hz.mp4`, `bbx.pt`, `fixed_identity.npz`, `frame-map.json`, `frame-contract.jsonl`, `identity-rejections.json`, `manifest.json`, and `receipt.json`. `identity_valid` means the **original online gate**; `crop_bbox_usable` means the crop is available under the explicit preparation policy. Do not confuse those masks. Every crop proposal has `ground_anchor_allowed=false`. A crop box or projected ankle is not a ground-contact or 3D label.

The bbox tensor uses the pinned official conversion functions read from the explicitly supplied GVHMR checkout; the helper file hash is retained. ViTPose still runs inside the prepared crops. Source YOLO keypoints are audit evidence, not an implicit replacement for ViTPose.

## Supply models and run GVHMR

The official layout below must exist inside the checkout selected by `--source-root`. Supply files under their respective terms; no model is included here.

| Path relative to GVHMR checkout | Purpose |
|---|---|
| `inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz` | Human body model, required even without rendering |
| `inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt` | Temporal motion model |
| `inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt` | Image features |
| `inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth` | Official pose estimator |

```bash
python run_gvhmr_fixed_identity.py \
  --source-root /external/GVHMR \
  --prepared /outputs/prepared/example_full_30hz \
  --output-dir /outputs/gvhmr/example \
  --camera-mode simplevo
```

The wrapper validates the pinned commit, tracked-source cleanliness, prepared hashes, crop shape, and 30 Hz grid. It creates a fresh bbox cache and makes `Tracker.get_one_track` raise if invoked. It bypasses the upstream FPS-copy step and rendering. SimpleVO is the default; static camera mode requires `--static-camera-evidence /data/reviews/camera.json`. A missing `--f-mm` uses the upstream intrinsics estimate, not a calibrated camera.

The recorded five-clip preparation reached this wrapper's missing-model check. That result is not a successful GVHMR inference claim. Once the required files exist, the wrapper still needs a working CUDA environment.

## Verify the G1 contract and retarget

The published [compiled mapping receipt](provenance/g1-mapping-audit.json) records an actual MuJoCo 3.11.0 comparison. Machine-specific source paths were replaced with logical paths; measured values and input hashes are unchanged. The exact original receipt hash is retained.

| Measurement | Recorded result |
|---|---|
| Both model dimensions | `nq=36`, `nv=35`, `njnt=30`, `nu=29` |
| Exported DOF order | Identity permutation `0..28`; every sign `+1`, every zero offset `0` |
| FK comparison | 14 bodies × 30 probes: zero pose and each joint at +0.1 rad |
| Maximum body-position / rotation-matrix difference | `0 m` / `0` |
| Range differences | 15 substantively narrower GMR ranges; 6 wrist differences from rounding |

This proves the recorded kinematic column mapping, not collision, inertia, actuator, dynamics, or tracking equivalence. Audit output against runtime joint limits; no automatic clipping is performed. File hashes are byte-level measurements from the audited worktrees; line-ending changes on another platform also change them. If either XML's bytes or the runtime contract change, regenerate the audit rather than bypassing its hash checks:

```bash
python audit_g1_mapping.py \
  --source-xml /external/GMR/assets/unitree_g1/g1_mocap_29dof.xml \
  --runtime-xml /runtime/assets/robots/g1/supermimic.xml \
  --runtime-constants /runtime/tasks/motion_tracking/g1/supermimic/robot.py \
  --output /outputs/g1-mapping-audit.json
```

Retarget using the matching receipt. The licensed GVHMR SMPL-X NPZ can be reused; choose `--body-model-ext npz` explicitly. PKL is also supported with an explicit `pkl` argument. The loader does not modify installed SMPL-X files or convert body-model data.

```bash
python retarget_g1_headless.py \
  --source-root /external/GMR \
  --gvhmr-result /outputs/gvhmr/example/hmr4d_results.pt \
  --inference-receipt /outputs/gvhmr/example/inference-receipt.json \
  --prepared /outputs/prepared/example_full_30hz \
  --body-model-root /external/GVHMR/inputs/checkpoints/body_models \
  --body-model-ext npz \
  --mapping-audit /outputs/g1-mapping-audit.json \
  --runtime-xml /runtime/assets/robots/g1/supermimic.xml \
  --output-dir /outputs/g1/example_raw
```

The wrapper includes frame 0, unlike the pinned upstream demo loop that increments its index before processing. It asserts frame counts and exports root xyz, root quaternion **xyzw**, and 29 joint angles in radians. The MuJoCo wxyz quaternion is explicitly reordered. There is no floor clamp, smoothing, time repair, or post-export joint clipping; failures retain a receipt and completed partial qpos when available.

The resulting reference is `robot_adapted`: GMR also scales the pelvis world trajectory, using its effective human scale table. The sidecar records that scale and preserves the GVHMR result as a separate source. An uncalibrated GMR root path is not a measured metric court trajectory. Use the adjacent `reference_tools.py audit` before constructing P0/P1 data. XY repair still requires independent grounded-foot anchors and a documented metric coordinate transform.

## Actual preparation evidence

[provenance/preprocessing-summary.json](provenance/preprocessing-summary.json) contains aggregate results, source/output hashes, and workspace-relative locations of the original receipts. Those original artifacts remain under `research/sports-video-execution-20261110/frontend-source/` in the parent SuperMimic workspace and are not part of this repository.

| Clip | Source frames at 25 Hz | Output frames at 30 Hz | Original rejected frames retained | Interpolated crop source frames |
|---|---:|---:|---:|---|
| `fb20k_0044_rally_000` | 357 | 428 | 2 | None |
| `fb20k_0044_rally_001` | 266 | 319 | 2 | None |
| `fb20k_0045_rally_000` | 434 | 520 | 4 | 370, 380 |
| `fb20k_0045_rally_001` | 302 | 362 | 2 | None |
| `fb20k_0046_rally_001` | 273 | 327 | 2 | None |

No initial frames were removed. Final sub-step crops are recorded in the JSON (0–0.02 s); endpoint padding was not used. All five outputs had usable crop boxes, but original identity gates and missing keypoints were preserved. `model_ready_except_weights` in these receipts describes crop/video/time preparation, not manual identity ground truth, an installed GPU environment, reliable 3D motion, or physical success.

[provenance/transfer.json](provenance/transfer.json) records original and publication script hashes and the exact CLI-only changes. [provenance/transfer-validation.json](provenance/transfer-validation.json) records the lightweight verification after transfer.
