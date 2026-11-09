# Sports-video XY experiment

This branch starts from `main` at `9f6a70d457420a951dcfc5074fc9c8f8d028f4ce`.
It implements the proposed controlled experiment. Engineering checks do not establish sports tracking success.

| Reference | F0: two zero columns | F1: reference-yaw XY error |
|---|---|---|
| P0: original audited reference | A | C |
| P1: only root XY repaired | B | D |

Both policies have 101 observation dimensions and start from the same full 16K checkpoint. New decoder columns and Adam moments start at zero. F0 keeps those columns frozen. F1 uses simulator root position, so it is a privileged-state simulation diagnostic; deployment would need an independently validated position estimate. The original reward, termination, world-model dimensions, critic, and existing normalizers remain fixed.

## Current evidence

- Real source acquisition: five continuous badminton rallies, 25 Hz, 1,632 frames. Source manifests and local video reports are outside this public repository.
- A persistent near-player 2D track has been processed for all five clips. Missing and low-confidence frames remain flagged. Coverage is not keypoint accuracy or a 3D result.
- CPU checks cover checkpoint migration, optimizer continuity, F0 freezing/F1 updates, native simulation, timestamps, XY-only repair, interpolation, and failure recording.
- The initial cloud engineering attempt failed before training because the motion filename was abbreviated incorrectly. Its log is retained; the corrected retry uses a fresh output directory.
- The corrected H100 engineering job completed 20 continuation iterations for each of F0 and F1 (4 world-model warmup iterations and 16 policy updates). This validates the runtime interface on existing engineering clips, not badminton tracking.
- Its full cloud receipt was recovered in 44 chunks and SHA256-verified. Both conditions collected 14,592 transitions and retained all normalizer tensors. F0's added columns remained exactly zero; F1's maximum added weight reached 0.0005736961. See [`engineering-evidence.json`](engineering-evidence.json). Checkpoint readback was verified on the cloud; checkpoint bytes were not downloaded locally.
- All five full video intervals have real 30 Hz model inputs (1,956 target frames), identity crop caches, and source-PTS maps. The original 2D rejection flags remain available. Two isolated missing crop boxes are explicitly interpolated proposals, not contact or pose ground truth.
- Compiled MuJoCo checks confirm the 29-DOF GMR/SDK export mapping and 30 forward-kinematics probes. Fifteen GMR joint ranges are narrower; six wrist range differences are rounding within 1e-5 rad. This does not prove dynamic equivalence.
- With no SMPL-X model available, an alternative [GEM-X/SOMA-X frontend](frontend/gem-soma/README.md) has processed one complete 428-frame rally on CPU and produced a G1 reference. The unchanged 16K controller failed at 6.92 seconds on CPU and at 7.76 seconds in the separate CUDA run. Real short CUDA continuation now completes the 14.22-second reference with both F0 and F1; F0 performs better in this single-clip trial. See [actual results and limitations](ACTUAL-SPORTS-PILOT.md). Reference quality, old40 retention, the remaining four clips and independent evaluation remain incomplete; no generalization or sports-quality success is claimed.
- The J old40 batch diagnostic retained 28 complete, 2 failed and 90 budget-truncated rollouts; all 9 original batch/single numerical checks failed. Q subsequently reproduced F0's regression on `fallAndGetUp1_subject5` with the original single-environment evaluator: parent and F1 completed 168.2 s, F0 failed at 13.36 s. F1's separate J `subject1` failure remains unverified in one environment. [J/N/Q code and compact evidence](followups/README.md) retain these negative results. N confirmed a BF16 batch-shape effect on identical frozen inputs, but both completed repeat40 physical prefixes still failed the original thresholds; only4/6 requested prefixes completed before timeout.

## Entry points

- `reference_tools.py probe-video|audit|repair-xy`: timestamps, coordinates, verified contact anchors and XY-only fitting. Synthetic software tests are not physical validation.
- `resample_reference.py`: one shared PTS-based resampling operation, before P0/P1 branching. Explicitly records any sub-frame tail crop.
- `train.py`: F0/F1 continuation with a frozen motion manifest, shared parent hash and finite budgets.
- `evaluate.py`: native first-failure evaluation, no post-start reset. Full planned length remains the return denominator. Early failure metrics are explicitly prefix-only.
- `cloud_engineering.py`: short single-GPU integration check on existing engineering clips; never counted as sports training.
- [`followups/README.md`](followups/README.md): byte-frozen J batch evaluation, N batch-shape isolation and Q single-environment regression follow-up; existing I runtime sources are preserved.
- [`frontend/README.md`](frontend/README.md): pinned upstream versions, explicit-path video preparation, fixed-identity GVHMR, headless GMR, and measured mapping evidence. Third-party checkouts and model files are external dependencies.

Training manifests must contain `reference_variant` and `clips` entries with `path`, `sha256`, `fps`, and `group` (`old`/`new`). All clips must have one explicitly established sampling rate. Do not relabel 25 Hz motion as 30 Hz. Correct video timing must be established before GVHMR inference, whose learned motion representation assumes 30 Hz.

P1 may only alter root XY; it may not change height, orientation, joints, or time to make tracking pass. Court projection of an ankle is not verified ground contact. A frozen independent E* reference is required for comparison, alongside P0/P1 cross-evaluation. Retain original old40 regression, separate CPU/CUDA and raw/bound20 protocols, and report every failure.

This directory does not contain broadcast source videos, licensed body-model weights, private chat exports, or training checkpoints. Do not publish those assets implicitly with the code.

The cloud controller now also emits receipt chunks shorter than the platform's 500-character log limit. That logging-only change was checked by a byte-exact local roundtrip of the actual receipt; the completed H100 run used the source payload hash recorded in the evidence file.
