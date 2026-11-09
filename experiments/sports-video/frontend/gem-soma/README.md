# Public GEM-X / SOMA-X route

This alternative removes the SMPL/SMPL-X asset dependency. It uses the official pretrained GEM-X native77 models, SOMA-X 0.3.3, and NVIDIA's SOMA retargeter. We have not trained a replacement human-body model.

The first complete 428-frame badminton rally was inferred on CPU in 824.859 seconds and converted into a 30 Hz, 36-column G1 reference. The original 16K controller failed after 6.92 of 14.22 planned seconds. These are real pipeline and failure results, not successful sports tracking. `actual-evidence.json` records the outputs and checks.

## External dependencies

Use separate environments for GEM and the retargeter if necessary. No third-party code, weights, body assets, sports footage, pose data or checkpoints are distributed here. The original upstream terms apply to each dependency. `source-pins.json` lists official URLs, revisions and verified model hashes.

- GEM: Python 3.12, PyTorch 2.10.0, Transformers 5.3.0, py-soma-x 0.3.3. The completed run used CPU PyTorch. CUDA execution is a separate experiment.
- Retargeting: Newton 1.3.0, Warp 1.14.0, usd-core 26.3, NumPy, SciPy and MuJoCo. Supply an external `soma-retargeter` checkout and its G1 assets. `--retargeter-deps` optionally selects an isolated dependency directory.
- Inputs: the full 30 Hz prepared-video directory from the adjacent fixed-identity preparer. Its crop convention was numerically checked against official GEM. Original identity rejections and proposal flags remain intact.

## Run

```bash
python run_gem_fixed_identity.py \
  --prepared /data/prepared/rally_full_30hz \
  --weights-dir /models/gem --soma-assets /models/soma \
  --gem-root /external/GEM-X --output /outputs/gem-raw \
  --device cpu --batch-size 2 --export-joints --verify-weight-hashes

python gem_to_soma_bvh.py \
  --predictions /outputs/gem-raw/raw_predictions.pt \
  --conditions /outputs/gem-raw/conditions.pt \
  --gem-root /external/GEM-X --soma-assets /models/soma \
  --output /outputs/g1-raw \
  --retarget-python /envs/retarget/bin/python \
  --retargeter-root /external/soma-retargeter \
  --runtime-xml /runtime/unisupermimic/assets/robots/g1/supermimic.xml
```

Output paths must be new. The inference script checks source revision, weight sizes/hashes and prepared-input hashes. The first command exports unprocessed predictions: no person reselection, time change or contact IK. Body-model transforms are flattened across the full timeline before SOMA geometry evaluation, retaining the actual per-frame parameters.

The converter serializes all 78 public transforms (77 controlled joints plus the virtual root), checks full-frame FK readback, then invokes official IK. It transforms SOMA `(x,y,z)` into robot `(x,-z,y)`, converts metres to BVH centimetres and back exactly once, and exports root xyz, xyzw quaternion and 29 joint angles in radians. The retargeter keeps its human-to-robot scaling, 24 IK iterations and joint limits. Contact processing, postprocessing and smoothing are explicitly off in this raw baseline.

## Interpretation and ground convention

Raw GEM root translation is an integrated learned world coordinate system; its first hip is at the origin. It is not a calibrated court position or measured height. The first diagnostic reference applies a separately recorded, whole-sequence constant vertical translation of +0.769676 m to put the minimum G1 collision sole at Z=0. This leaves every XY coordinate, orientation and joint unchanged, but does not enforce contact.

The resulting reference has both soles above 3 cm for 396/428 frames and above 10 cm for 128/428 frames; maximum joint finite-difference speed is 28.413 rad/s. These identify quality concerns, not verified jump labels or actuator violations. All raw artifacts and the failed physical trajectory are retained outside this repository.

Contact postprocessing changes XYZ and joints and therefore defines a new P0 parser. It cannot be counted as the planned XY-only P1 repair. Freeze that parser before comparing P0/P1 and F0/F1, and keep an independently defined evaluation reference. No acceptance claim follows from an attractive reference animation.

The source transfer adds explicit paths for external retargeter dependencies. The transferred converter was rerun on all 428 actual frames; BVH and G1 CSV equality to the original raw result is recorded in `actual-evidence.json`. Model inference was not needlessly repeated for that path-only transfer.
