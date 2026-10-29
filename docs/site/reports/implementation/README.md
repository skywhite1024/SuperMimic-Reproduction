# SuperMimic LAFAN1 baseline on SenseCore

Latest result (2026-10-26): the controlled full-checkpoint continuation stopped
early at 16000 iterations. The fixed all-40 primary evaluation achieved macro
**0.84749192**, with an independent reload confirmation of **0.84773350**;
both had **0/40 early failures**. The agreed community target is met. See the
[continuation report](continuation-20261025-a/report/continuation-comparison.html)
and [its implementation/evidence notes](continuation-20261025-a/README.md).
The auxiliary numerical replay differences and global positional drift remain
explicitly reported. Final checkpoints, logs, all-40 results and videos were
downloaded and verified; all related cloud compute resources are released.
The original 12000-iteration baseline below remains preserved as historical
evidence, together with its independent report and execution record.

This directory implements the authorized single-H100, Unitree G1, 40-clip
LAFAN1 community baseline. Training, functional resume acceptance, independent
evaluation and video generation completed on 2026-10-25. The 12000-iteration,
seed-0 run achieved macro normalized return **0.80281353**, frame-weighted
**0.80940831**, and **3/40 early failures**. This does **not** meet the reference
goal of about 0.82 and zero failures; normalized return is not a success rate.

Open [the report](report/supermimic-report.html) for curves, all 40 results,
failure records and three videos. Verified downloads are under `downloaded/`:
the inference checkpoint is `runs/baseline-seed0/ckpt_012000.pt`, and the full
optimizer checkpoint is `runs/baseline-seed0/last.pt`. The 53-file archive and
every extracted file passed SHA256 verification. Both checkpoints passed ZIP
CRC checks; the inference checkpoint was actually loaded for cloud evaluation.
All 1000 video frames decoded locally. Local model inference was not rerun.

Failed clips are `dance2_subject4`, `fallAndGetUp1_subject5`, and
`fallAndGetUp2_subject3`. The separate evaluation uses the same training motion
library. The upstream README reference used UniLab 1.5.0, whereas the pinned
source lock used here installs 1.5.1. No controlled experiment establishes the
cause of the difference. The original 40-clip denominator and failures remain.

`execution.json` records job identities and verified boundaries. This run's ACP
jobs are terminal; the supervised transfer CCI is stopped with an empty instance
list, DNAT unbound and its original command restored. See
`final-resource-verification.json` for the final check, which postdates the
frozen cloud result summary. Cloud data has been retained.

## Frozen inputs

- Community implementation: Motphys/Uni-SuperMimic,
  `faf7818e8abac3ec6abb8f00e6e904d3577f800b`.
- Dataset: `lvhaidong/LAFAN1_Retargeting_Dataset`, revision
  `ce1572906efe6157840e8474d5a0d7aa87481e74`.
- 40 CSV clips, 264,705 source frames at 30 Hz, 36 columns per frame.
- Robot: 29 hinges plus floating pelvis, 14 tracked bodies, 35 meshes.
- Runtime pins come from the upstream `uv.lock`; the hash-enforced export is
  `requirements-upstream.lock.txt`. Key versions are UniLab 1.5.1,
  PyTorch 2.14.1, MuJoCo/MuJoCo-Warp 3.11.0, and Warp 1.16.0.

The original research checkouts remain untouched under `../research/20261024`.
`uni/` is a separate implementation checkout. `evidence/source-files.json`
and `evidence/asset-downloads.json` identify the staged code and data bytes.

## Scope and protocol

The community configuration has 99-dimensional single-frame proprioception,
including six dimensions of reference-relative root orientation. The audited
paper table lists 93 dimensions. Default BF16, collision settings, physics and
network widths are retained. Results must be described as a community baseline,
not a strict reproduction of all paper experiments.

Training uses mjwarp, 4096 environments, seed 0 and 12000 iterations with the
upstream model sizes, 32-step imagination horizon and critic. The finite smoke
reduces parallelism and batches, disables compilation and shortens world-model
warmup; it still executes simulation collection, world-model optimization,
BPTT policy optimization and critic updates. It then exits and resumes from a
full optimizer checkpoint. Model parameter hashes and optimizer counters must
advance after both stages. Replay is rewarmed; RNG/replay continuation is not
bit-exact. Resume identity checks cover complete model states and optimizer
step counters; separate hashes of optimizer moment tensors were not recorded.

Independent evaluation uses one environment per clip, starts each clip at frame
zero, uses the deterministic policy, disables noise and randomization and sets
friction to 1.0. It evaluates the same 40-clip training library in a separate
process; this is tracking performance, not unseen-motion generalization.
An early failure keeps the complete resampled clip denominator:
`(clip_length - 1) * 0.1`. The evaluator writes per-clip survival steps, full
length, denominator, normalized return and failure status, plus macro and
length-weighted averages. A target around 0.82 with 0/40 failures is a reference
goal, not a result already achieved.

## Scheduled commands

All cloud writes are confined to:

```text
/mnt/afs/[account]/SuperMimic/lafan1-20261024/
```

After transferring the staging archive, prepare with zero GPUs in
`computing_cluster` using `cpu-final.command.txt` (four CPU cores, 16 GiB RAM,
one replica, maximum restarts zero). It deploys `cloud_prepare_final.sh` with
a SHA256 check and a one-hour upper bound. This consolidates the hardlink
installation and headless verification that succeeded in the recorded runs.
The consolidated script passes Bash syntax checking; a separate clean rebuild
of this consolidated version has not been run. The current environment has
passed actual asset, rendering, GPU-training and resume acceptance checks.
The staged archive avoids a verified unreachable Hugging Face endpoint, and the
package mirror must serve files with the original pinned hashes. Source and
raw data are verified before installation; MuJoCo asset validation runs again
after installation.

After environment preparation succeeds, submit `gpu-acceptance-offline.command.txt` on
one H100, four CPU cores and 64 GiB host memory. It verifies and deploys the
audit wrappers before running:

```bash
cd /mnt/afs/[account]/SuperMimic/lafan1-20261024/uni
MUJOCO_GL=disable WANDB_MODE=offline HF_HUB_OFFLINE=1 timeout --signal=TERM --kill-after=30s 2700s \
  .venv/bin/python -u ../run_gpu_v2.py --mode acceptance
```

The recorded acceptance used `gpu-acceptance.command.txt`. The recommended
offline variant only adds the local asset mode already used by formal training;
it has been syntax checked but has not been submitted as a separate acceptance.

After inspecting the acceptance receipt, `gpu-full.command.txt` runs the
default baseline and then a separate evaluation process. It verifies the v2
runtime and creates v3 with shorter progress messages because SenseCore CMS
truncates individual log records at 500 characters. The training implementation
is unchanged. The startup command installs the CPU-verified OSMesa library
version for videos; simulation and evaluation use `MUJOCO_GL=disable`, while
video children use OSMesa. `HF_HUB_OFFLINE=1` keeps prepared asset resolution
offline. The driver refuses to overwrite an
existing baseline directory. Explicit recovery of a partial formal run must be
reviewed against its checkpoint and receipts before use.

`run_gpu_v3.py` records a resource sample every five seconds, applies a 6 GiB
host-memory and 2 GiB GPU-memory reserve, and terminates its own process group
on time or reserve limits. ACP must use one replica and maximum restarts zero.
Check the platform terminal state after exit. The guards do not replace that
resource-release verification.

## Evidence

- `execution.json`: current verified progress, job identities and resource state.
- `evidence/asset-downloads.json`: immutable raw-asset hashes.
- `evidence/source-files.json`: staged source hashes.
- `evidence/staging.json`: allowlisted transfer archive hash and Linux uv hash.
- Cloud `evidence/environment.json`: installed package versions after success.
- Cloud `evidence/acceptance.json`: actual smoke/resume update proofs.
- Cloud `runs/*/resources.jsonl`: sampled GPU and cgroup memory usage.
- Cloud `runs/baseline-evaluation.json`: complete independent evaluation.

After downloading and verifying the result bundle, local audit commands are:

```powershell
.\.asset-venv\Scripts\python.exe verify_results.py --archive <archive> --receipt <receipt> --output downloaded
.\.asset-venv\Scripts\python.exe audit_videos.py --results downloaded
.\.asset-venv\Scripts\python.exe make_report.py --results downloaded
```

`requirements-local-assets-and-audit.txt` freezes the local Python 3.12 asset,
plotting and video-audit environment. It is separate from the Linux GPU lock.
The verifier checks the archive hash, rejects unsafe paths and checks every
file hash. The video audit decodes every frame and retains short failure clips.

The temporary CPU CCI is a file-transfer route only. Its SSH fingerprint was
checked against the trusted console before acceptance, and it must be manually
stopped with the instance list empty and DNAT unbound after transfer. The missing
school stop helper remains an unresolved platform condition.
