# Retention v1: finite 1000-update experiment

This release records one controlled candidate, its matched bound20 parent, and all failures.
The deployed model is one 101D student policy. Teachers are used only during training.
The first 200 updates train the world model only; the remaining 800 are joint updates.
The original experiment plan, references, rewards, physics, denominators and acceptance thresholds remain frozen.

Candidate classification: `candidate_unsuccessful_no_automatic_extension`.
The complete raw-reference goal is `False`; the calibrated-reference goal is `False`.
These are project engineering criteria, not an author-scale SuperMimic benchmark result.

| Backend | Raw new strict /15 | Calibrated new strict /15 | Old gate /40 | Raw sequences /3 | Calibrated sequences /3 |
|---|---:|---:|---:|---:|---:|
| CUDA | 7 | 9 | 37 | 0 | 0 |
| CPU | 7 | 9 | 38 | 0 | 0 |

Read the [complete report](site/reports/acro-library-20261027-b/retention-1030-report.html) for every clip,
all full 120-second trials, contact diagnostics, original hard gates, and full failed-tail videos.
Contact geometry is diagnostic; contact forces and real-robot performance were not measured.

## Re-evaluate the released policy

Install the pinned environment using [REPRODUCING.md](REPRODUCING.md). Then run from the repository root:

```sh
python tools/fetch_inputs.py --baseline --new-library
python tools/reconstruct_references.py
python tools/fetch_checkpoints.py --name retention-v1-1000
python tools/evaluate_released_retention.py --name retention-v1-1000 --verify-only
python tools/evaluate_released_retention.py --name retention-v1-1000 --suite library --reference-variant corrected --group all --device cpu --mode released-retention-calibrated-cpu
python tools/evaluate_released_retention.py --name retention-v1-1000 --suite library --reference-variant raw --group new --device cpu --mode released-retention-raw-cpu
python tools/evaluate_released_retention.py --name retention-v1-1000 --suite sequences --reference-variant raw --device cpu --mode released-retention-raw-sequences-cpu
python tools/evaluate_released_retention.py --name retention-v1-1000 --suite sequences --reference-variant corrected --device cpu --mode released-retention-calibrated-sequences-cpu
```

Use fresh output names. CUDA is a separate assessment: change both `--device` and `--mode` and report its results separately.
For library quality, run `assess_tracking.py`, `audit_physical_contacts.py`, and `combine_acceptance.py`
from `acro-library-20261027-b/scripts` with the selected `--run` name.
For continuous quality, use `assess_sequences.py` and `audit_sequence_contacts.py`.
All evaluators retain the full window after failure and prohibit resets after t=0.

## Exact parent and complete checkpoints

`tools/fetch_checkpoints.py --name retention-parent-stage3-full` retrieves the exact full parent used by the controls.
Select `--name retention-parent-stage3-full` in the released evaluator to run its bound20 controls.
`tools/fetch_checkpoints.py --name retention-v1-1000-full` retrieves the complete candidate including optimizer state.
The old 16K and stage2 teacher checkpoints remain in v0.1.0 and are listed in `artifacts/manifest.json`.
No second training block is triggered by any evaluator.

## Original and public hashes

`SOURCE_PROVENANCE.json` records the original and published byte hashes. Public path redaction and newline
normalization can change text-file hashes while the checkpoint retains its original plan hash.
The released evaluator validates the exact release checkpoint, original plan identity, public plan bytes,
all frozen evaluation source files, and pinned vendored source. It then uses the same bound20 loader and
native evaluation implementations. It never edits a checkpoint or substitutes acceptance thresholds.
Historical cloud operational drivers are audit records: configure a new per-user plan before any fresh cloud training.
The three public video renderers use portable installed-font fallbacks; both overlay font sizes are preserved.
Their original/public hashes and the presentation-only adaptation are recorded in SOURCE_PROVENANCE.json.

Raw motion archives, reference CSVs, environments, credentials, temporary transfers and private SSH materials
are excluded from the repository. Fetch motion inputs from the pinned original sources.

## Publication entry validation

The released entry was run on all 15 raw new motions using verified local pinned inputs and the existing CPU environment. Every per-clip result and four complete trajectory arrays exactly matched the private CPU assessment; strict acceptance remained 7/15. One full 2.52-second clip was also rendered and decoded with the portable-font public renderer. A fresh dependency installation and original-input network downloads were not retested. The initial isolated fixture omitted robot meshes; that pre-simulation failure and its correction are retained in the validation receipt.
