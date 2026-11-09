# Actual badminton continuation and old-motion regression

Continuation improved survival on the one trained badminton reference, but F0 also regressed on a previously tracked old motion. A separate single-environment check reproduced that failure: the parent and F1 completed 168.2 seconds, while F0 failed at 13.36 seconds. Old-skill retention and a benefit from the new XY input have not been established.

The public GEM-X / SOMA-X frontend produced a complete 428-frame G1 reference without SMPL-X assets. Starting from the existing 16K SuperMimic checkpoint, both short continuation conditions completed its 14.22-second CUDA / MuJoCo Warp rollout. This is a single training-clip result, not generalization or a validated badminton skill.

## R: public frontend completed all five clips

The subsequent R job completed public pretrained GEM-X / SOMA-X inference for **5/5 full rallies: 1,956 frames at 30 Hz, 65.2 seconds**, in 539.35 seconds total on one H100. It restored four genuine weight-download receipt sidecars missing from the previous package; the scientific entry, weights and prepared inputs were unchanged. It performed no training or controller evaluation. The downloaded archive and all 44 member hashes were verified; all 30 PT outputs were finite and matched the frozen frame clock and identity flags.

The subsequent local CPU conversion completed **raw G1 retargeting for all five clips: 1,956 rows, 36 columns, 30 Hz**, with the original conversion settings. All 12 identity-rejected frames remain rejected, including two mid-clip interpolated crop proposals. These raw references have zero explicit vertical offset and no ground/contact calibration; they are not accepted sports motions or five trained/evaluated G1 clips. The CPU/CUDA first-clip comparison also changes decoded RGB and batch size, so its differences cannot be attributed solely to GPU numerics. See [compact R evidence](followups/evidence/r-actual.json) and the [frontend README](frontend/gem-soma/README.md). The I controller results below still use the **original CPU first-clip reference with its constant whole-clip Z offset**, not any of these new raw CUDA-derived references.

## What was actually trained

The frontend uses public pretrained weights. The controller was **fine-tuned from the same 16K parent**, not trained from scratch: each condition completed 100 iterations, 400 world-model updates and 80 policy updates. Both observed 160,768 new simulation transitions. F0 and F1 have the same 101-dimensional observation interface; F0's extra columns remain zero and frozen, while F1 receives reference-yaw root XY error. Existing normalizer tensors, rewards and native failure conditions were unchanged.

Both F0 and F1 used **50% new-reference / 50% old40 replay sampling**; the experimental difference was the extra XY observation only.

The reference is the original retargeted motion with one constant whole-clip ground-height offset. It does not use the later contact or root-Z repair candidates. No P1 XY-repaired reference was used here.

## Actual CUDA results

| Policy | Native outcome | Full-reference normalized return | XY RMSE over recorded trajectory | XY RMSE over common first 7.76 s |
|---|---|---:|---:|---:|
| Parent 16K | Failed at 388/711 steps, 7.76 s | 0.350380 | 1.149255 m, failure prefix | 1.149255 m |
| F0 | Completed 711/711 steps, 14.22 s | 0.762455 | 0.227672 m, full clip | 0.124076 m |
| F1 | Completed 711/711 steps, 14.22 s | 0.736471 | 0.401444 m, full clip | 0.210478 m |

F0 performed better than F1 in this trial. The result supports continuation learning on this clip; it does not demonstrate a benefit from XY feedback. The original full reference remains the return denominator after early failure. CPU and CUDA parent rollouts are separate evidence.

One H100 was used. Peak reserved CUDA memory was about 4.09 GB for F0 and 4.07 GB for F1. The measured stage times were 151.03 and 90.88 seconds, including different warm-up costs; these are not a fair F0/F1 speed comparison. Four H100s are sufficient for parallel experiments of this size, but do not solve reconstruction quality, independent evaluation or data licensing.

## Old40 batch diagnostic J: incomplete and numerical checks failed

The original I sequential old40 evaluation ran out of its budget. Parent and F0 each completed 2/40; F1 completed 1/40 and had one budget-truncated rollout. The remaining 38 per condition were not run. Those original results remain unchanged in [`sports-pilot-evidence.json`](sports-pilot-evidence.json).

J then evaluated all 40 fixed old clips in 40 environments per model. Each environment was assigned its own reference from frame zero. Scoring stopped at its first original native failure or reference end; subsequent backend states were discarded, with no resets. This used the same frozen I checkpoints, raw actions, seed zero and BF16 policy path. It did not train.

| J policy | Completed | Native failure | Budget-truncated | Original 2-second batch/single checks passed |
|---|---:|---:|---:|---:|
| Parent | 10 | 0 | 30 | 0/3 |
| F0 | 9 | 1 | 30 | 0/3 |
| F1 | 9 | 1 | 30 | 0/3 |
| Total | 28 | 2 | 90 | 0/9 |

The three fixed check indices were 0, 20 and 39. Root-position differences were below 1 mm, but every check exceeded other predeclared tolerances, including raw-action and joint-position differences. The thresholds were not relaxed. N subsequently confirmed a BF16 batch-shape contribution for identical frozen inputs, but that did not resolve the physical-prefix mismatch. CPU inference uses a different FP32 path and cannot validate CUDA equivalence.

J's F0 failed `fallAndGetUp1_subject5` at 13.34 seconds; F1 failed `fallAndGetUp1_subject1` at 7.92 seconds. The latter has **not** been checked again in a single environment. J's aggregate returns include 90 truncated trajectories with different observed lengths: they must not be used to rank the policies or accept retention. Full per-clip status, prefix lengths, original thresholds, differences and hashes are retained in [`followups/evidence/j-actual.json`](followups/evidence/j-actual.json).

## Single-environment follow-up Q confirms the selected F0 regression

Q selected `fallAndGetUp1_subject5` because J's parent completed it and F0 failed. This is an outcome-triggered diagnostic, not a randomly selected or held-out evaluation. It reused the original I single-environment evaluator, the same checkpoints, raw actions, BF16, seed zero, frame-zero initialization, original reward and termination, and no resets or precision override.

| Q policy | Original native outcome | Executed / planned steps | Full-reference normalized return | XY RMSE scope |
|---|---|---:|---:|---|
| Parent 16K | Completed 168.20 s | 8410 / 8410 | 0.7814170930 | 1.629714 m, full clip |
| F0 | Failed at 13.36 s | 668 / 8410 | 0.0662132265 | 0.221228 m, failure prefix |
| F1 | Completed 168.20 s | 8410 / 8410 | 0.7784957572 | 0.499297 m, full clip |

All three reached an actual native terminal outcome without budget truncation. The selected F0 regression therefore also occurs under the original single-environment protocol; it cannot be dismissed as only J's batch numerical mismatch. F0's shorter XY-error prefix is not a fair comparison with the other full-length errors. F1 completing **subject5** does not clear its J failure on **subject1**, nor establish old40 retention or a general advantage of XY feedback. Full metrics and SHA-verified log-recovery provenance are in [`followups/evidence/q-actual.json`](followups/evidence/q-actual.json).

## Precision diagnostic N: forward effect confirmed, new physical gate failed

N's actual receipt was recovered and hash-verified. Across all 15 fixed frozen-input probes, BF16 actual40 reproduced J's saved actions exactly and repeat40 matched actual40 exactly; batch1 differed by up to 0.03125. Under the separate strict FP32 forward-only condition, with both AMP and TF32 disabled, the batch1/40 difference fell to 1.103e-6. That comparison does not isolate TF32 alone and is not a FP32 physical evaluation.

The 110-second soft budget expired at 110.670 seconds. **Four of six** requested 100-step physical prefixes finished; the fifth only initialized and executed zero steps before its budget stop, and the sixth never started. Both completed new repeat40 prefixes still failed the unchanged J tolerances, including raw-action differences of 0.03125 and joint-velocity differences above 0.107. The new gate is therefore not merely missing evidence: it already has two observed failures.

The repeat40 first actions already differed from J by 0.015625/0.00390625 at indices 0/20, before physical integration, despite equal recorded initial states. The remaining discrepancy needs actual history/goal/context capture; it cannot be assigned solely to accumulated physics error. Simply completing the missing run cannot make all checks pass. The [frozen N script](followups/precision-n/diagnose_precision.py) and [compact actual evidence](followups/evidence/n-status.json) preserve all original J thresholds and failures. A separate [first-input diagnostic](followups/precision-n/diagnose_first_input.py) is prepared with a finite budget and zero physical steps. It has no new CUDA outcome, training, full-path equivalence or old40 acceptance claim.

## Remaining limits

Reference contact, jump fidelity and motion quality remain unaccepted. There is no frozen independent E*, no full P0/P1 × F0/F1 experiment, no multi-seed test, no held-out sports result and no physical G1 deployment. Constant-height and contact-processing failures are retained separately.

## Provenance

[`sports-pilot-evidence.json`](sports-pilot-evidence.json) records the exact I hashes and downloaded actual metrics. The selected result archive was 439,726,080 bytes; its SHA256 and all 33 member hashes were verified locally. Both final checkpoint files were downloaded and matched the training proof. Videos are rendered from saved CUDA states using forward kinematics; they do not rerun the policy. Failed trajectories stop at their last real state.

[`followups/README.md`](followups/README.md) provides J/N/Q source and finite-budget commands, plus compact J/N/Q/R evidence. R adds frontend metadata only. The existing I `train.py`, `sports_runtime.py` and `evaluate.py` remain byte-identical to the downloaded run. The original cloud runtime-lock bytes are preserved separately; the existing repository lock has the same 63 hash entries with different serialization. No validated simulation or training tests were rerun for this packaging update.

The repository contains code and compact evidence, not broadcast videos, model weights or training checkpoints. Full local artifacts are described by the user's execution report outside this repository.
