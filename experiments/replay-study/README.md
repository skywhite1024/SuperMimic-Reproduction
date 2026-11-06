# Course replay study: four ratios, three continuation seeds

Question: under one common update budget, how does old-motion replay trade off
retention of 40 LAFAN1 motions against adaptation to 15 additional motions?
This is an ablation of the independent community implementation, not a
paper-scale SuperMimic reproduction or unseen-motion benchmark.

## Protocol

`plan.json` defines old replay fractions `0, .25, .5, .75` and training seeds
`0, 1, 2`. Every run loads the same SHA-verified full 16K checkpoint. Its native
99D observations, architecture, reward, imagination horizon and termination
rules are retained. Parent observation/goal/dynamics/value normalizers remain
fixed. `inputs.lock.json` pins all 55 raw motion files by SHA256; paths use
portable forward slashes.

Only the group mixture for world-model batches and policy imagination starts
varies. Both use the same ratio, then uniform clip and eligible-start sampling.
Collection always uses a 50/50 old/new group schedule, with uniform clips and
nonterminal reference frames. A balanced first reset covers all 55 clips.
Collection rules are shared; realized trajectories can differ as policies change.
Formal replay raises an error if an active clip has no eligible starts rather
than silently dropping it. Engineering runs explicitly relax that requirement.

Each seed controls new collection and continuation updates; the parent training
seed is shared. Loading full optimizer state does not restore historical replay,
simulator state or RNG streams. Every run builds a fresh replay buffer with the
same warm-up rule.

## Current execution status

On 2026-11-06, 12 CPU engineering runs completed four updates each (one WM-only
and three joint), with full-size networks and reduced batches. All WM, policy
and critic tensors changed, saved tensors matched readback, and normalization
statistics stayed fixed. These establish execution, not a replay-performance
conclusion. Seed 0 used engineering-v2; seeds 1/2 used engineering-v3 after
additional provenance/finite-value checks. These are pre-freeze code iterations.
An initial zero-update sampler initialization failure is retained.
An additional 1,024-environment CPU collection check populated all 55 strict
replay pools after the common 64-step warm-up (minimum 494 eligible starts per
clip across the WM/policy pools); it performed no optimizer updates.

A fresh CPU/MuJoCo full-55 evaluation of the common parent retained all clips
and failed tails without resets after initialization. The original hard gate
passed 40/40 old motions and 5/15 additional motions; macro normalized returns
were 0.844512 and 0.460141. This gate is not strict tracking/contact success.
The CPU numbers must not be combined with historical CUDA results.

`evidence/engineering-20261106.json` records the executed artifacts and limits.
No new GPU job has been submitted: live account activity/quota could not be
read because the cloud console connection timed out. The formal budget is
unfrozen. The 2,000-update / 7,200-second values are provisional, not results.

## Reproduce and continue

First follow the repository's environment/data setup. Restore the full parent
with `python tools/fetch_checkpoints.py --name lafan40-full` (check that tool's
help if its CLI changes), or pass an existing matching file with `--parent`.
Never download or overwrite a second copy unnecessarily.

From the repository root, with the installed project Python:

```bash
python -m unittest discover -s experiments/replay-study -p test_sampling.py
python experiments/replay-study/run_matrix.py
python experiments/replay-study/run_matrix.py --execute --phase engineering --device cpu --seeds 0 1 2 --prefix engineering-local-a
```

Use fresh labels. Runtime logs, checkpoints and arrays remain in ignored `runs/`
directories. The runner stops on an error and preserves failed outputs.

After institutional location/network requirements, account-wide queued/running
ACP and GPU CCI activity, and current quota are verified, commit source and run
a finite single-H100 pilot. Pilot seed 100 is excluded from the three main seeds:

```bash
python experiments/replay-study/run_matrix.py --execute --phase pilot --device cuda --prefix pilot-a
```

Measure peak memory, warm-up/steady-update time, complete-55 evaluation time and
actual collection/replay coverage. Freeze one main budget before inspecting
main outcomes; keep all 12 runs and evaluation/diagnostic overhead within the
provisional 48 GPU-hour ceiling. Set `main_budget_frozen` only after recording
the measured decision, then commit the frozen plan. The matrix is sequential;
it does not allocate cloud resources or automatically restart failed jobs.

```bash
python experiments/replay-study/run_matrix.py --execute --phase main --device cuda --prefix main-a
python experiments/replay-study/evaluate_matrix.py --prefix main-a --label main-a
python experiments/replay-study/evaluate_matrix.py --execute --prefix main-a --label main-a
```

Main execution rejects per-run budget overrides. `evaluate_matrix.py` requires
all 12 successful main checkpoints from the same frozen plan, then evaluates
the parent plus all 12 on CUDA/MJWarp. It preserves each complete trajectory and
checks 40/15 denominators. It never substitutes a CPU score for GPU evaluation.

## World-model diagnostic and reporting

Before main training, collect a fixed evaluation-only transition probe with
the common parent on the matching GPU backend:

```bash
python experiments/replay-study/wm_probe.py --collect --device cuda
python experiments/replay-study/wm_probe.py --probe experiments/replay-study/runs/frozen-probe-cuda/windows.npz --checkpoints PARENT.pt CANDIDATE.pt --output experiments/replay-study/runs/diagnostics/main-a.json
```

The probe contains 440 eight-step windows, eight per clip, from four fixed
reference phases under the parent policy. These trajectories never enter a
training buffer. Prediction diagnostics can use CPU arithmetic; the metadata
identifies the physics backend that generated the transitions. This measures
within-library prediction error, not unseen-motion generalization. It may
underrepresent the state distribution of an adapted policy; report that limit.

Report all three seed points, mean and sample standard deviation per ratio;
old-motion forgetting is baseline old return minus candidate old return, and
adaptation gain is candidate additional return minus baseline additional return.
Include full-clip macro returns, original-gate failures, tracking errors, actual
update counts, wall/GPU time and failed trials. Normalized return is not a
success probability. CPU engineering diagnostics do not select a winning ratio.
Optional no-reset continuous/hardware tests remain separate; no robot execution
is performed by these scripts.
