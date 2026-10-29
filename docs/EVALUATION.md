# Evaluation contract

- Baseline: the same fixed 40 LAFAN1 training motions, deterministic policy, clip frame zero, no observation/action noise or randomization, friction 1.0.
- Preserve every clip and its full denominator after early failure; report macro and frame-weighted normalized returns and failed-clip counts separately. A normalized return is not a success probability.
- New library: exactly 15 motions; old retention library: exactly 40. The 14 initially untrained motions and the previously trained 88_08 form the same 15-motion reporting denominator.
- Raw and constant-height-calibrated references are separate variants. CPU/mjbatch and CUDA/mjwarp are separate backend experiments.
- Continuous testing: 120 seconds, fixed schedules, only t=0 initialization, no resets after t=0, no retries hidden from the denominator. Report first failure times, accumulated failure frames, tracking and contact quality.
- Continuing diagnostic sampling after a hard failure does not make the sequence a pass. Preserve complete failure-tail videos and trajectories.
- The community baseline, the 101D global-position extension, the single-motion specialist and the unified policy are different experimental conditions.
- Frozen source/data/contract hashes identify historical inputs. `SOURCE_PROVENANCE.json` also identifies the publication copies; publication sanitization must not be mistaken for a new historical training run.
- Overall unified 55-motion + strict continuous-control goal remains incomplete in this release.
