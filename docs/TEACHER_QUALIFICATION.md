# Bounded teacher qualification, 2026-10-30

The frozen 101D stage 2 teacher fails all three protected motions under the unchanged strict tracking/contact contracts. It is not a qualified preservation target for all three. The previous 1000-update candidate remains unadopted and the unified 55-motion goal remains incomplete.

| CPU control | Original hard gate | Tracking | Contact geometry | Combined strict | Protected strict |
|---|---:|---:|---:|---:|---:|
| Stage2 teacher, raw15 |13/15|10/15|11/15|8/15|0/3|
| Stage2 teacher, calibrated15 |15/15|12/15|11/15|10/15|0/3|

All 30 complete physical traces start at native frame 0, prohibit resets after t=0 and preserve failed tails. The finite diagnostic took 298.828 seconds on two local CPU cores, with zero training updates and no new GPU job. All 30 full 720p videos were decoded and hashed. Raw/calibrated trajectories for the three protected motions are identical.

Read the [complete report](site/reports/acro-library-20261027-b/teacher-qualification-1030-report.html), including the full denominators, videos, reward decomposition, reference geometry and all failed checks.

## Two additional single-motion controls

The previously released 99D 88_08 specialist passes the same strict per-motion contracts under bound20: normalized native return 0.669032, minimum actual geometric contact distance -0.037086 m. Both raw and calibrated full 4.72-second controls pass, with zero postt0 resets and two complete videos. These controls have denominator 1 per reference, a different architecture and a different training scope. They are not a unified 15/55-motion score and do not replace the 101D student.

The unmodified 88_08 reference has ground penetration reaching 8.93 cm, but the specialist still passes the fixed acceptance criteria. Reference-pose geometry alone therefore does not establish dynamic infeasibility. No reference, physics, reward or acceptance threshold was changed.

## Offline implementation evidence

An independent zero-update adapter probe reconstructs all 236 retained specialist actions exactly from real histories, using the first 99 columns of 101D observations. Thirty-two real histories produce finite student gradients with frozen teacher parameters. Its historical MSE field was mislabeled as parent states; the [clarification](site/reports/acro-library-20261027-b/evidence/qualified-teacher-adapter-1030-clarification.json) records that the histories come from specialist states and the evaluated student is the parent.

A separate isolated CPU engineering run performs six updates: 2 world-model-only and 4 joint. Optimizer-step deltas are 6/4/4, all three networks change, teachers and normalizers remain unchanged, and saved tensors read back exactly. Two independent policy loaders return exactly equal bounded outputs on four real histories. The first functional probe used a one-row counter for four histories and failed before comparison; its correction and failure receipt are retained. This engineering checkpoint is not adopted and has no physical acceptance claim.

## Reproduce the released physical controls

Use the pinned environment and explicit input setup in [REPRODUCING.md](REPRODUCING.md), then run from the repository root:

```sh
python tools/fetch_inputs.py --baseline --new-library
python tools/reconstruct_references.py
python tools/fetch_checkpoints.py --name stage2-full
python tools/evaluate_released_teacher.py --name stage2-full --verify-only
python tools/evaluate_released_teacher.py --name stage2-full --reference-variant raw --mode released-teacher-raw15-cpu
python tools/evaluate_released_teacher.py --name stage2-full --reference-variant corrected --mode released-teacher-calibrated15-cpu
python tools/fetch_checkpoints.py --name acro88-08
python tools/evaluate_released_teacher.py --name acro88-08 --verify-only
python tools/evaluate_released_teacher.py --name acro88-08 --reference-variant raw --mode released-specialist8808-raw-cpu
python tools/evaluate_released_teacher.py --name acro88-08 --reference-variant corrected --mode released-specialist8808-calibrated-cpu
```

Choose fresh mode names. The portable entry checks exact release bytes and original/public source lineage, then uses the unchanged native full-clip evaluator and bound20wrapper. For each new stage2run, execute `assess_tracking.py`, `audit_physical_contacts.py` and `combine_acceptance.py` with its `--run` name. For the specialist, the first two scripts support denominator 1; the unchanged per-clip contact/tracking adjudication is recorded explicitly in `qualify_frozen_specialist1030.py` and in the single-clip assessment JSON. The original 15-clip combiner deliberately rejects a one-clip denominator.

Both model assets already exist in v0.1.0. No duplicate release or model upload is needed. Historical local/cloud orchestrators and their input-hash plans are audit records; prepare a fresh per-user plan before rerunning them. Raw motion files and trajectory NPZs remain outside Git; numerical summaries, complete videos and provenance are published.

## Next finite hypothesis

The [prepared proposal](site/reports/acro-library-20261027-b/evidence/qualified-expert-retention-v2-proposal.json) uses the qualified 99D teacher only for 88_08, keeps the native objective for 88_07/09, and retains the old-library teacher, sampling, one 101D student and all acceptance contracts. It proposes at most one H100, 1000 new updates and 5400 seconds, with a CUDA specialist precheck before any update. The full finite-trial driver, startup command, unchanged specialist bundle and local raw/calibrated teacher gates are prepared and verified. Cloud staging, a live resource/network preflight and the CUDA teacher gate remain prerequisites. No new GPU job has been submitted, and no automatic second training block is triggered by these diagnostics.


## Public entry validation

The actual public entry was run in the isolated pinned-input fixture on all 15 raw stage 2 motions and the raw 88_08 specialist. Every per-clip metric, four complete trajectory arrays, and all tracking/contact records exactly match the private runs; stage 2 strict acceptance remains 8/15. Existing verified inputs and dependencies were reused. A fresh install, original-input network download and public calibrated resimulation were not retested.
