# SuperMimic Reproduction — Unitree G1

Community-implementation experiments with **40 LAFAN1 motions**, **15 additional motions**, and **120-second continuous control**. This repository records the implementation, evaluation contracts, complete results, and failures. It is not the official SuperMimic implementation or a claim of reproducing all paper experiments.

**中文：这是复现实验与工程记录。基础 LAFAN1 目标已达到；新增动作与旧动作兼顾的统一模型目标仍未全部达到。所有失败和完整分母保留。**

- [Read the seven complete reports](https://skywhite1024.github.io/SuperMimic-Reproduction/)
- [Browse the committed report assets](docs/site/): seven reports, 266 complete-duration browser video copies, figures and numerical results.
- [Reproduction guide](docs/REPRODUCING.md) · [Evaluation protocol](docs/EVALUATION.md) · [Source/data attribution](THIRD_PARTY.md)
- [Local/server Git workflow](docs/SERVER_WORKFLOW.md) · [Clean-clone 120-second validation](validation/clean-clone-continuous-seed0.json)
- [Server source reconciliation](validation/cloud-source-reconciliation-20261029.json): historical AFS snapshot, matching public checkout, and separately retained server-only sources.
- [Checkpoint manifest](artifacts/manifest.json): named inference and training-resume artifacts, sizes, SHA256 and exact installation paths.

## Recorded outcomes

| Experiment | Recorded result | Boundary |
|---|---|---|
| LAFAN1 16K community baseline | Primary macro normalized return 0.84749192; independent confirmation 0.84773350; 0/40 early failures in both | Same 40-motion training library; normalized return is not a success rate |
| Basic continuous adapter + offline lookahead | Fixed six sequences: 6/6; six further frozen sequences: 6/6; 120 s, zero resets after initialization | Basic LAFAN1 motions; does not establish acrobatic generalization |
| 88_08 specialist | Single hand-supported backward flip passes its declared inversion/upright-ending gates | Single-motion specialist; not the unified 55-motion policy |
| Unified stage 3 (CUDA) | Raw new motions 10/15; calibrated new motions 12/15; old LAFAN1 38/40 | Combined tracking/contact quality: raw 7/15, calibrated 11/15 |
| Unified stage 3 continuous tests | 0/3 strict passes; first failures approximately 9.16 s, 6.82 s, 40.98 s | Full failure tails retained; disabling automatic termination is not counted as passing |
| BONES-SEED | Preparation snapshot only; paused | No full-library training claim |

CPU and CUDA evaluations are separate experiments. See report rows for exact protocol, checkpoint and backend. Do not combine their scores or select successful clips across different policies.

## Repository layout

```text
implementation/                    # audited baseline and controlled continuation
  uni/                             # vendored, unchanged pinned upstream Git source
interactive-g1-20261027-a/           # basic continuous control and specialist experiments
acro-library-20261027-b/             # unified 15-new + 40-old experiments and evaluations
envs/                              # GPU lock, CPU requirements and observed environment
tools/                             # setup, asset/checkpoint retrieval and publication verification
artifacts/manifest.json             # checkpoint identities; binaries are release assets
docs/site/                         # complete report website, including all 266 videos
SOURCE_PROVENANCE.json              # original/public source hashes and publication transformations
UPSTREAM.lock.json                 # pinned source and dataset revisions
```

The historical relative directory layout is retained to keep the recorded scripts understandable. Original experiments remain unchanged outside this public checkout. Platform-only examples have personal paths replaced by `USER_ID`; they require per-user configuration and are not cloud resource provisioning tools.

## Start here

Viewing the committed reports requires only Python:

```bash
python -m http.server 8000 --directory docs/site
```

Open `http://localhost:8000`. To reproduce policy inference or training, follow [the reproduction guide](docs/REPRODUCING.md). Reports are historical evidence; rerunning inference is a separate step.

Source-only validation uses no GPU and downloads no data:

```bash
python tools/check_release.py
```

The publication checkout was also tested from a separate clean Git clone using the recorded CPU environment: one full 120-second / 29-segment basic continuous trial, zero post-initialization resets, zero original hard-gate failures, XY RMSE 0.20531774 m. Input references were reconstructed and hash-verified. This is a functional clean-clone test, not a fresh dependency-install test or a rerun of all historical experiments.

## Provenance and limits

- Upstream: `Motphys/Uni-SuperMimic`, commit `faf7818e8abac3ec6abb8f00e6e904d3577f800b`, preserved in `implementation/uni` with its LICENSE, NOTICE and LICENSES directory.
- LAFAN1 source revision: `ce1572906efe6157840e8474d5a0d7aa87481e74`.
- Recorded cloud stack: Python 3.12, UniLab 1.5.1, Torch 2.14.1, MuJoCo/MuJoCo-Warp 3.11.0, Warp 1.16.0; one H100 for the reported GPU stages.
- The community baseline uses 99D proprioception; the global-position extension uses 101D. Their checkpoints cannot be interchanged without the recorded architecture conversion. The paper's 93D observation specification is a separate protocol.
- Motion data is retrieved from pinned upstream sources; raw AMASS/SMPL-X archives and model body assets are not redistributed here. Source licenses and applicable data terms remain separate.
- Training resumes rewarm replay and do not restore the entire simulator/RNG/replay state bit-for-bit. Numerical identity across hardware/backends is not promised.
- Geometric contact audits are sampled geometric estimates, not measured contact forces.

## License

Original code added by this reproduction project is provided under Apache-2.0. Vendored source retains its original notices and component licenses. The code license does not relicense third-party motion data, robot assets, checkpoints' dependencies, or referenced material. See [THIRD_PARTY.md](THIRD_PARTY.md).


## Retention v1 (2026-10-30)

The finite 1000-update candidate and its full parent are published with matched CPU/CUDA evaluations.
See [the experiment and reproduction instructions](docs/RETENTION_V1.md) and the complete linked report.
The frozen raw-reference goal is false; all failures remain included.


The [2026-10-30 teacher qualification](docs/TEACHER_QUALIFICATION.md) adds complete raw/calibrated stage2 controls,32full diagnostic videos and isolated CPU engineering evidence. The protected stage2 teacher is0/3; the historical8808specialist passes its one-motion control. These findings do not complete the unified55-motion goal or trigger another GPU job.
