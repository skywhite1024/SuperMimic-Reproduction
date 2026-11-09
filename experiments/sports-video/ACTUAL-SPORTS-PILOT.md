# First real badminton continuation experiment

The public GEM-X / SOMA-X frontend produced a complete 428-frame G1 reference without SMPL-X assets. Starting from the existing 16K SuperMimic checkpoint, both short continuation conditions completed its 14.22-second CUDA / MuJoCo Warp rollout. This is a single training-clip result, not generalization or a validated badminton skill.

## What was actually trained

The frontend uses public pretrained weights. The controller was **fine-tuned from the same 16K parent**, not trained from scratch: each condition completed 100 iterations, 400 world-model updates and 80 policy updates. Both observed 160,768 new simulation transitions. F0 and F1 have the same 101-dimensional observation interface; F0's extra columns remain zero and frozen, while F1 receives reference-yaw root XY error. Existing normalizer tensors, rewards and native failure conditions were unchanged.

The reference is the original retargeted motion with one constant whole-clip ground-height offset. It does not use the later contact or root-Z repair candidates. No P1 XY-repaired reference was used here.

## Actual CUDA results

| Policy | Native outcome | Full-reference normalized return | XY RMSE over recorded trajectory | XY RMSE over common first 7.76 s |
|---|---|---:|---:|---:|
| Parent 16K | Failed at 388/711 steps, 7.76 s | 0.350380 | 1.149255 m, failure prefix | 1.149255 m |
| F0 | Completed 711/711 steps, 14.22 s | 0.762455 | 0.227672 m, full clip | 0.124076 m |
| F1 | Completed 711/711 steps, 14.22 s | 0.736471 | 0.401444 m, full clip | 0.210478 m |

F0 performed better than F1 in this trial. The result supports continuation learning on this clip; it does not demonstrate a benefit from XY feedback. The original full reference remains the return denominator after early failure. CPU and CUDA parent rollouts are separate evidence.

One H100 was used. Peak reserved CUDA memory was about 4.09 GB for F0 and 4.07 GB for F1. The measured stage times were 151.03 and 90.88 seconds, including different warm-up costs; these are not a fair F0/F1 speed comparison. Four H100s are sufficient for parallel experiments of this size, but do not solve reconstruction quality, independent evaluation or data licensing.

## Incomplete evidence retained

The fixed sequential old40 regression ran out of its evaluation budget. Parent and F0 each completed 2/40; F1 completed 1/40 and had one budget-truncated rollout. The remaining 38 per condition were not run. These incomplete denominators cannot establish old-skill retention. A separate batched evaluator is being investigated and must not be silently substituted for the original numerical protocol.

Reference contact, jump fidelity and motion quality remain unaccepted. There is no frozen independent E*, no full P0/P1 × F0/F1 experiment, no multi-seed test, no held-out sports result and no physical G1 deployment. Constant-height and contact-processing failures are retained separately.

## Provenance

[`sports-pilot-evidence.json`](sports-pilot-evidence.json) records the exact hashes and downloaded actual metrics. The selected result archive was 439,726,080 bytes; its SHA256 and all 33 member hashes were verified locally. Both final checkpoint files were downloaded and matched the training proof. Videos are rendered from saved CUDA states using forward kinematics; they do not rerun the policy. Failed trajectories stop at their last real state.

The repository contains code and compact evidence, not broadcast videos, model weights or training checkpoints. Full local artifacts are described by the user's execution report outside this repository.
