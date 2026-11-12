# Source and data attribution

## Vendored training implementation

`implementation/uni` contains Git-tracked files of [Motphys/Uni-SuperMimic](https://github.com/Motphys/Uni-SuperMimic), fixed at `faf7818e8abac3ec6abb8f00e6e904d3577f800b`. Its Apache-2.0 LICENSE, NOTICE, and component LICENSES directory are preserved. See `UPSTREAM.lock.json` and `SOURCE_PROVENANCE.json`.

This is a community implementation. The original SuperMimic paper and authors should be cited separately; this repository does not represent official author code.

## Official source snapshots added on 2026-11-12

`upstream/supermimic` preserves [HybridRobotics/supermimic](https://github.com/HybridRobotics/supermimic) at `abc124c1bd5976341034668af3d12c087f2f1f55`, including the Apache-2.0 license and author acknowledgements. `upstream/mjlab_research` preserves [HybridRobotics/mjlab_research](https://github.com/HybridRobotics/mjlab_research) at `dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd`, including its Apache-2.0 license and component notices. Robot assets retain upstream attribution and applicable terms. Both trees are byte-identical snapshots, separately identified by [SOURCES.lock.json](upstream/SOURCES.lock.json).

These directories contain official upstream code; the historical experiments above remain community-implementation results. Importing official source does not attribute those experiments to the authors or establish reproduction of their results. No motion dataset or model-weight download is included in this import.

## Motion and robot assets

- G1/LAFAN1 snapshot: [lvhaidong/LAFAN1_Retargeting_Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset), revision `ce1572906efe6157840e8474d5a0d7aa87481e74`.
- Additional G1 motion CSV source: [lez666/chunwan-flip](https://github.com/lez666/chunwan-flip), commit `c493a897f2ca81153e980a65ee6f0faf7cb07487`; manifests retain exact original source URLs and hashes.
- The latter source identifies AMASS/CMU origins and GMR retargeting. Its repository-level MIT declaration does not replace upstream data/body-model terms. This repository does not redistribute raw AMASS/SMPL-X archives.
- Robot/model assets and downloaded motions retain their own applicable licenses and terms; the reproduction project's code license does not relicense them.

The downloader retrieves assets from their original pinned sources. Generated frozen-reference CSVs are reconstructed locally from committed manifests.

## Reports and videos

The reports and rendered simulation videos describe this reproduction's experiments. The 151 browser copies retain the original video duration and frame rate, with H.264 compression. Historical file hashes in reports describe original files; `docs/site/public-manifest.json` identifies the browser copies. External references and their authors remain attributed in the reports.
