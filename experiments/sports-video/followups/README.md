# Frozen I follow-ups: J batch evaluation, N precision isolation, Q single-clip check

The continuation improved its trained badminton clip, but Q reproduced an old-motion failure of F0. J did not complete old40 and failed all original numerical parity checks. N confirmed identical-input BF16 batch-shape differences, but both completed repeat40 physical prefixes still failed. Read the [actual results](../ACTUAL-SPORTS-PILOT.md) before interpreting these diagnostic tools.

No weights, broadcast videos, reference CSVs or trajectory arrays are included. The executed J/Q/N sources are copied byte-for-byte. The separate post-N first-input script is newly prepared and has not run on CUDA. Existing I `train.py`, `sports_runtime.py` and `evaluate.py` are unchanged. [`source-provenance.json`](source-provenance.json) records all hashes and the previously verified local CPU mock receipts; those tests were not rerun while packaging.

| Diagnostic | Native physics / policy inference | Fixed budget | Actual evidence |
|---|---|---|---|
| J | 40 CUDA environments; BF16 batch40 | 500 s/model, 1800 s outer | [120 retained clip records and 9 failed parity checks](evidence/j-actual.json) |
| Q | 1 CUDA environment/model; original BF16 policy1 | 600 s/model, 900 s outer; 2 H100 lanes | [Three native outcomes on the selected subject5 clip](evidence/q-actual.json) |
| N | Frozen-input forward comparisons and 1 physical environment with policy1/repeat40 | 110 s soft, 115 s TERM + 5 s KILL; 1 H100 | [4/6 complete prefixes, 1 zero-step stop, 1 not started; both completed repeat40 prefixes failed](evidence/n-status.json) |

## Required external artifacts

Provide the original I `evaluation-checkpoints.json`, its unchanged parent/F0/F1 checkpoints and reference manifests, the verified UNI/MuJoCo-Warp runtime, and J's actual `old40-manifest.json`, three summaries, `runtime/evaluate_batch.py` and saved per-clip trajectories for N. Model hashes are retained in the evidence. These are explicit inputs, not files bundled by this repository.

The archival I runtime lock has SHA `39c2bd247ea2f26195fdbd5e609ae7a200c6a3297983aa4de3766ba034837c13`; its exact bytes are in [`frozen-i/runtime-source-lock.json`](frozen-i/runtime-source-lock.json). The lock already at the study root contains the same 63 entries but is serialized differently. Q's exact-byte preflight requires the archival copy in the external I runtime directory. Do not change an existing experiment directory to make its preflight pass; reconstruct a separate archival runtime if necessary, retaining the original I Python source hashes.

All commands below require fresh output directories, already provisioned compatible dependencies and appropriate existing GPU allocation. They describe finite diagnostic protocols; this package does not submit resources or rerun completed jobs. Set paths to your actual external artifacts:

```bash
export SUPERMIMIC_UNI=/path/to/frozen-runtime-root/implementation/uni
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export MUJOCO_GL=disable
I_RUNTIME=/path/to/archived-I-runtime
I_SPEC=/path/to/existing-I/evaluation-checkpoints.json
```

## J: original 40-environment evaluation

```bash
CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=10s 1790s \
  python -u batch-j/run_reused_i.py \
  --source-spec "$I_SPEC" --runtime-dir "$I_RUNTIME" \
  --output /path/to/fresh-J-output --device cuda \
  --max-seconds 1700 --per-model-seconds 500
```

Each env_id is assigned the manifest clip at frame zero. After native failure or reference end, the scoring ledger freezes that prefix. Inactive worlds continue internally with zero action and reference end-1 to keep the backend valid, but all post-stop samples and rewards are discarded; there is no reset. The original MDP hard-failure rule is rescored even though manager termination transport is suppressed for the batch. Raw actions, reward, native failure thresholds and clip time are unchanged.

The separate original single-environment checks use fixed indices 0/20/39 for 100 control steps. They all failed in the actual J run; do not widen tolerances or silently call batch40 equivalent to policy1. J's 90 truncated rollouts and different observed lengths invalidate using its aggregate returns for retention acceptance or policy ranking. Failures and truncated prefixes remain listed individually.

## Q: outcome-triggered single-environment follow-up

Edit a copy of [`single-q/q-plan.example.json`](single-q/q-plan.example.json): replace the explicit path placeholders with the existing I spec, checkpoints, reference and runtime; use a fresh output. Keep the code/model/reference hashes unchanged. The template is not the executed cloud plan; its original plan SHA is recorded inside it.

```bash
timeout --signal=TERM --kill-after=10s 890s \
  python -u single-q/run_regression_q.py --plan /path/to/edited-q-plan.json
```

The original two-GPU schedule runs parent on GPU0 and F0 then F1 on GPU1, with one physical environment per worker. It calls the unchanged I `evaluate_clip`; no precision override or training. The target is **subject5**, selected after J's F0 failure. Parent/F1 completed 8410 steps while F0 failed at step668. This supports a regression on this selected clip, not a random sample of retention. F1's J **subject1** failure has no single-environment follow-up in this evidence. [`q-transport-hashes.json`](evidence/q-transport-hashes.json) records all recovered log chunk payload SHA values, independently of the derived compact metric JSON.

## N: finite precision isolation, actual negative gate retained

```bash
CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=5s 115s \
  python -u precision-n/diagnose_precision.py \
  --i-spec "$I_SPEC" --runtime-dir "$I_RUNTIME" \
  --j-root /path/to/existing-J-output --model parent --max-seconds 110 \
  --output /path/to/fresh-precision-parent-n
```

The script reconstructs policy histories from J's saved states/actions, freezes steps 0/1/10/50/99 at indices 0/20/39, and compares identical inputs under batch1, repeat40 and actual40. It records AMP/TF32/runtime flags and requires actual40 to reproduce J's recorded actions. A strict FP32 forward-only comparison is separate; flags are restored before physical checks. Six single-environment prefixes then compare original policy1 and repeat40 policy inference against J using its original comparator and thresholds.

`new_policy40_short_prefix_gate_passed` is false in the recovered actual receipt. Four prefixes completed 100 steps; the fifth stopped after initialization with zero steps and the sixth was not started. Both completed repeat40 prefixes failed the original tolerances. On the 15 identical frozen inputs, repeat40 matched actual40 and J exactly, while batch1 differed up to 0.03125. Strict FP32 reduced the latter to 1.103e-6 but changed both AMP and TF32 and did not run physical evaluation.

The first actual repeat40 action already differs before integration, so matching the MLP batch shape alone is insufficient. Completing the missing sixth check cannot undo the two observed failures. CPU mock receipts cover adapter and bookkeeping behavior only; CPU inference bypasses CUDA BF16 autocast.

## Prepared post-N first-input isolation; not executed on CUDA

[`diagnose_first_input.py`](precision-n/diagnose_first_input.py) initializes the same three single environments but forbids physical steps. It reuses the original N frozen inputs, checks original model/runtime/receipt hashes, captures actual state/history/goal/context, and compares four frozen/native history-goal combinations in the same policy instance. Repeating before input construction is compared with repeating the completed inputs. Per-component differences distinguish observations from future-reference goals. A separate strict FP32 control changes AMP and TF32 together; it retains existing MotionLib caches and is not a fully FP32 motion-library rebuild.

```bash
CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=5s 295s \
  python -u precision-n/diagnose_first_input.py \
  --i-spec "$I_SPEC" --runtime-dir "$I_RUNTIME" \
  --j-root /path/to/existing-J-output \
  --n-root /path/to/existing-precision-parent-n --max-seconds 250 \
  --output /path/to/fresh-first-input-output
```

This is a prepared one-GPU, four-CPU diagnostic with zero restarts and a maximum 300-second outer budget, not a submitted task. It reads N's existing `precision-receipt.json`, `frozen-real-J-inputs.pt` and three `original_policy1` initial trajectories. Captures are saved before forwards and retained on timeout. A completed receipt means input evidence was collected; it never marks the original J/N gate passed or reports a new performance score.
