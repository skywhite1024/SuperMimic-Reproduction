# Reproducing the experiments

Use Python 3.12. Run commands from the repository root. Read the recorded limitations before comparing numerical results.

## 1. Select an environment

CPU inference / Windows environment:

```bash
python -m venv .venv
# Linux: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1+cpu
python -m pip install -r envs/requirements-cpu.txt
python -m pip install --no-deps -e implementation/uni
```

`envs/cpu-observed-freeze.txt` records the actual original Windows environment; it is evidence, not a cross-platform universal lock. `mjbatch-uni==0.2.5` and MuJoCo 3.11.0 are required by the recorded CPU runtime.

Linux GPU training environment:

```bash
python -m pip install uv==0.12.2
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --require-hashes -r implementation/requirements-upstream.lock.txt
uv pip install --python .venv/bin/python --no-deps -e implementation/uni
```

Use the original Linux image/CUDA stack described in the baseline notes. GPU scripts intentionally retain the recorded one-H100 guard. Different GPU support has not been validated; CPU inference is separately supported. Set `MUJOCO_GL=disable` for training/evaluation; video children use an appropriate headless renderer (OSMesa in the recorded Linux runs).

## 2. Fetch pinned assets and reconstruct references

```bash
python tools/fetch_inputs.py --baseline
python tools/fetch_inputs.py --new-library
python tools/reconstruct_references.py
```

The baseline downloader verifies the immutable 40-motion/robot asset snapshot and invokes unchanged upstream generation/validation tools. The new-library downloader uses the exact URLs and hashes in `library-manifest.json`. Data remains subject to its original terms; raw AMASS/SMPL-X files are not included.

Frozen reference manifests are committed. Reference reconstruction must match their saved CSV hashes before any evaluation. If the reconstruction reports a hash mismatch, retain the generated output for diagnosis and do not relabel it as the original frozen reference.

## 3. Download a named checkpoint

```bash
python tools/fetch_checkpoints.py --list
python tools/fetch_checkpoints.py --name continuous-basic
```

The manifest distinguishes inference policies from full training-resume checkpoints. Downloads are verified by SHA256 and placed at the paths expected by the historical scripts. Existing matching files are reused; conflicting files are never silently overwritten.

## 4. Reproduce the basic 120-second continuous trial

```bash
python interactive-g1-20261027-a/scripts/evaluate_continuous.py --mode reproduction-basic-new-01 --checkpoint interactive-g1-20261027-a/policy/global-adapter-600-deployed.pt --reference-set references-lookahead --seeds 0 1 2 3 4 5
python interactive-g1-20261027-a/scripts/render_video.py --mode reproduction-basic-new-01 --seed 0
```

Use a fresh mode name: recorded results must not be overwritten. The policy advances MuJoCo physics; reference pose writing is not used as robot control. Only t=0 initialization is permitted. The evaluator continues sampling after failures, but accumulates the original failure gates separately.

## 5. Evaluate the unified policy

```bash
python tools/fetch_checkpoints.py --name unified-stage3
python acro-library-20261027-b/scripts/evaluate_library.py --checkpoint acro-library-20261027-b/policy/unified-h100-stage3-4000.pt --mode reproduction-stage3-raw-new-01 --group new --device cpu
python acro-library-20261027-b/scripts/evaluate_ground_variant.py --checkpoint acro-library-20261027-b/policy/unified-h100-stage3-4000.pt --mode reproduction-stage3-calibrated-all-01 --group all --device cpu
```

These are new CPU experiments, not a re-creation of the recorded CUDA numbers. Full trajectories and strict physical-quality audits are separate from the original height/tilt gates. Use the committed assessment scripts and report complete 15/40 denominators.

## 6. Resume the recorded third training stage

After installing GPU dependencies, fetching both motion libraries and the `stage2-full` parent checkpoint:

```bash
python acro-library-20261027-b/scripts/train_replay_stage3.py --initial acro-library-20261027-b/runs/unified-h100-stage2-4000/final-full.pt --label reproduction-stage3-new-01 --device cuda --steps 4000 --wm-warmup 400 --num-envs 1024
```

This executes actual collection and world-model/policy/critic updates. Use your scheduler's finite time limit, a checkpoint on interruption, one replica and no automatic restart loop. It does not provision cloud hardware. Checkpoint identity, replay balancing and saved-state readback are checked by the script. GPU execution from this public checkout must be validated independently; copying source is not a claim of rerunning training.

The `implementation` scripts preserve the original baseline and 12K→16K controlled continuation. The source for the unsubmitted retention trial is included as a preparation record; it is not an achieved result or an instruction to submit it.
