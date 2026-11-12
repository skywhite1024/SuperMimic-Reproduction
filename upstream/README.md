# Official upstream snapshots — 2026-11-12

These are exact, unmodified Git trees imported into an isolated local review branch. They are separate from the historical community implementation in `implementation/uni`.

| Directory | Official repository | Commit |
|---|---|---|
| `supermimic` | https://github.com/HybridRobotics/supermimic | `abc124c1bd5976341034668af3d12c087f2f1f55` |
| `mjlab_research` | https://github.com/HybridRobotics/mjlab_research | `dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd` |

The sibling layout preserves SuperMimic's editable dependency path `../mjlab_research`. Both repositories retain their original licenses, tests, documentation, workflows and robot assets. Nested workflows are source snapshots; they do not run as this repository's root CI.

See [the review and migration recommendation](../docs/OFFICIAL_UPSTREAM_REVIEW_20261112.md), [source hashes](SOURCES.lock.json), and [verification evidence](../validation/official-upstream-review-20261112.json).

Verify all 583 files without installing the simulation stack, from the repository root with Python 3.13+:

```bash
python tools/verify_official_import.py
```

This checks bytes and syntax only. Runtime training, checkpoint playback and official paper results have not been reproduced. In particular, the SuperMimic `uv.lock` and current sibling `mjlab_research` dependency requirements disagree. Do not treat their combination as an already validated environment or overwrite the historical environment with it.

The local Git remotes `official` and `official-mjlab` retain upstream histories. Remote configuration is local and is not carried by a clone of this branch. Source provenance remains available in the lock file. Future updates should pin both commits together, preserve an unchanged source snapshot and validate a separate environment before changing the experiment baseline.
