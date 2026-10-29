# Controlled continuation of the G1/LAFAN1 community baseline

Live state on 2026-10-26: training stopped early at iteration 16000 after two
independent all-40 evaluations met the community target. Primary macro return
was 0.8474919199943542 and confirmation was 0.8477334976196289, both with 0/40
early failures. The different returns are retained, not asserted bit-identical.
ACP `supermimic-resume-[account]-1026c1`
(`d5195d2569884639b489bed761d74356`) was submitted at 09:18:07 +08 and entered
running at 09:18:51. Its later auxiliary strict replay comparison failed at
`dance1_subject1` (`diagnose.py:79`, return tolerance 1e-5), so the original
pipeline is preserved as failed, platform terminal 10:14:06. This is distinct
from the completed training and two successful quality evaluations.

A separate finite auxiliary-only ACP `supermimic-post-[account]-1026d1`
(`b4bf233f7ba84ae0a6386ca65d3b33b3`) was submitted at 10:29:00 and entered
running at 10:29:25. It does not train or alter primary results. Its numerical
rollouts retain all-40 differences from the primary evaluations and a truthful
strict-match flag; per-step traces describe these independent trajectories.
The auxiliary job reached platform success at 10:54:22. Its selected diagnostic
rollout had macro return 0.846662700176239 and 0/40 failures. All three originally
failed clips completed their full lengths in that auxiliary rollout. Numerical
replay is not bit-identical: the old checkpoint's auxiliary run had macro
0.817372 and two failures, rather than its historical 0.802814 and three.

Final artifacts are verified under `downloaded/`: 80 files, two checkpoints,
two primary evaluations and two independent all-40 diagnostic rollouts, six
compressed per-step traces and three 15-second videos. The 284226494-byte archive
SHA256 is `2e188a90f3dc6bea903fc7f29b768620c971ee250cb8ff0fb7eb9f57b701f6bc`.
Every payload hash and both checkpoint CRCs passed. CPU deserialization of the
final full checkpoint matched all three models and complete optimizer moments
to the training snapshot; counters are WM160000 / policy15500 / critic310000.
All 1125 video frames decoded; sampled frames and all four scientific figures
were visually inspected. The browser verified 40 comparison rows, four loaded
figures and three playable-format videos with 960x540 metadata and 15s duration.
Local model inference was not performed.

Cloud resource cleanup passed: all four submitted ACP jobs are terminal; the
reused zero-GPU CCI is stopped with no instances, DNAT is unbound and its exact
original 1651-byte command is restored/published (SHA256
`2c81f1b1fc679c393562231dcb6ab0d7b24532c1b959375eaebf138fe149d6f4`).
The two new GPU job windows total a conservative 1.3558 GPU hours, within the
authorized 6-hour cap; this is not a billing statement. Original cloud outputs
and the original failed pipeline are retained. See
`final-resource-verification.json`, `budget-ledger.json` and
`report/continuation-comparison.html`.

Zero early failures is defined by the frozen height/tilt termination gates.
Global root-position drift remains: the selected auxiliary traces peak at
1.35m, 2.18m and 2.45m for the three clips. This remains a community baseline,
not a strict reproduction of the complete paper or an unseen-motion test.

Automatic approval blocked optional deletion of two 1651-byte local temporary
configuration backups without a more specific reason. Both backups are
retained; this does not affect cloud cleanup. No alternate deletion was tried.

The cloud CPU preflight passed after fixing tuple versus
JSON-array comparison in the audit code. It verified 54 source files, 76 assets,
the complete parent checkpoint and optimizer counters. GPU load identity then
passed for complete model and optimizer moment states, followed by actual
12000→12005 updates with replay size 66. `execution.json`
is the current evidence record. The first failed CPU preflight and its original
scripts are retained under `revisions/initial-preflight-v1/` locally; cloud
`.initial-v1` copies preserve the two repaired audit scripts.

This experiment retains the complete 12000-iteration baseline, its checkpoints,
data, evaluator and report. The only training change is to resume seed 0 from the
full model/critic/optimizer checkpoint and extend total iterations to 24000.
Neither optimizer learning rates nor the 250000-iteration cosine schedule are
reset. Replay is rewarmed for at least 64 collection steps before updating;
RNG, simulation state, adaptive start-bin statistics and replay are not restored
bit for bit. This is a continuation boundary, not uninterrupted training.

The original learner implementation remains unchanged. A dedicated entrypoint
creates a separate policy directory from the recorded full configuration and
verifies the same simulation contract. All model state and complete optimizer
moment tensor hashes must match the parent after loading. Five full-size
updates must pass finite-state, optimizer-counter and replay-warmup checks.
Training then continues in the same process. It pauses at iterations 16000,
20000 and 24000 for separate evaluator processes, retaining every checkpoint and
all 40 full-denominator results. These pauses do not clear replay.

The acceptance target is macro normalized return >=0.82 and 0/40 early failures.
It is the agreed community-baseline target, not a claim that the original paper
explicitly published the LAFAN1 zero-failure count. A qualifying checkpoint is
reloaded in another process for a second complete evaluation. A confirmed
qualifier stops the finite continuation early. Otherwise the total limit is
24000 iterations. Best is declared by fewest failed clips, then highest macro
return; highest-macro and final checkpoints are reported separately.

Rewards, observations, physics, thresholds, dependency versions, data order and
the primary evaluator are frozen by SHA256. Evaluation starts every clip at
frame zero, with deterministic policy, no noise/randomization and friction 1.
An early failure keeps `full_clip_steps * 0.1` as its denominator. The same 40
training clips are evaluated; this does not test unseen-motion generalization.

The parent and selected checkpoint also get instrumented separate rollouts.
The initially planned strict all-40 replay-identity gate failed and is retained
as failed. The auxiliary-only run exports its own all-40 returns, survival steps
and flags, plus every difference from the primary evaluator. It does not relax
or replace primary acceptance metrics. Frozen termination gates must exactly
explain each recorded transition in these auxiliary trajectories. Traces cover
all three original failed clips: root height, the exact tilt component used by
the termination gate, root position/orientation, yaw, joint-position RMSE and
action magnitude/rate. Three videos show the new selected policy near the new
failure or the original failure time. Joint-position error is not action error.

Cloud deployment uses a finite zero-GPU ACP preflight, then one H100/4CPU/64GiB
ACP in `computing_cluster`, one replica, restart count zero, node SSH disabled.
It reuses the existing personal AFS environment and assets. The first program
has a roughly three-hour bound; the new experiments together have a six GPU-hour
cap. Five-second resource guards retain the previous 6GiB host/2GiB GPU reserve.
Any additional experiment needs evidence and an independent directory; this
script never launches a second seed or version trial automatically.

Prepared scripts do not prove a run was submitted. See `execution.json` when
present for current submission, actual evidence and final resource state.
The old baseline report is deliberately not overwritten.

## Result acceptance and local analysis

After both GPU tasks reach verified terminal states, the auxiliary
`postprocess-20261026-a/collect_complete.py`
packages a frozen allowlist with an archive receipt and per-file SHA256 manifest.
Retrieval uses the existing zero-GPU debug CCI only for a short attended window;
its process timeout is not a resource-release mechanism. The operator must stop
the application, verify no instances remain, unbind the temporary DNAT and
restore its exact original command while it is stopped.

From the `implementation` directory, after retrieving the real archive and
receipt, use an empty output directory:

```powershell
.asset-venv\Scripts\python.exe verify_results.py --archive continuation-20261025-a\continuation-results.tar.gz --receipt continuation-20261025-a\result-transfer-receipt.json --output continuation-20261025-a\downloaded
.asset-venv\Scripts\python.exe continuation-20261025-a\audit_downloaded.py --results continuation-20261025-a\downloaded
.asset-venv\Scripts\python.exe continuation-20261025-a\analyze_diagnostics.py --results continuation-20261025-a\downloaded
.asset-venv\Scripts\python.exe continuation-20261025-a\make_comparison.py --results continuation-20261025-a\downloaded
```

These analysis programs do not rerun model inference locally. They verify the
downloaded evidence, evaluation contracts and checkpoint containers, decode all
frames of the three new videos, and build the comparison from actual results.
`diagnostic-analysis.json` reports exact terminal gates, tracking/action summary
statistics and all-40 changes. The descriptive trace does not prove the learning
cause. Only the three original failed clips have per-step diagnostic traces.

The HTML report refers to sibling evidence, figures and videos; keep this whole
experiment directory together. Its final resource badge depends on the separate
`final-resource-verification.json`, populated only after live console checks.
