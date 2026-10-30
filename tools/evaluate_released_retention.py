"""Evaluate a named released checkpoint with verified original/public provenance."""
import argparse
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repository', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--name', choices=['retention-v1-1000', 'retention-parent-stage3-full'], default='retention-v1-1000')
    parser.add_argument('--checkpoint-file', type=Path)
    parser.add_argument('--suite', choices=['library', 'sequences'], default='library')
    parser.add_argument('--reference-variant', choices=['raw', 'corrected'], default='raw')
    parser.add_argument('--group', choices=['all', 'new', 'lafan'], default='new')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--mode')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    repository = args.repository.resolve()
    root = repository / 'acro-library-20261027-b'
    provenance = json.loads((repository / 'SOURCE_PROVENANCE.json').read_text(encoding='utf8'))
    lookup = {row['path']: row for row in provenance['files']}
    plan_name = 'acro-library-20261027-b/retention-trial-v1-plan.json'
    plan_path = repository / plan_name
    original_plan_sha = lookup[plan_name]['original_sha256']
    assert original_plan_sha == 'e737aacaf113cf602be0eb6cb7b64a5b47e14451acdb003a4c19d9cbaa575a53'
    assert sha(plan_path) == lookup[plan_name]['public_sha256']
    plan = json.loads(plan_path.read_text(encoding='utf8'))
    for name, digest in plan['files'].items():
        relative = 'acro-library-20261027-b/' + name
        source = lookup[relative]
        assert source['original_sha256'] == digest, relative
        assert sha(repository / relative) == source['public_sha256'], relative
    extra_sources = ['acro-library-20261027-b/scripts/runtime.py',
                     'acro-library-20261027-b/scripts/audit_sequence_contacts.py']
    vendored = [name for name in lookup if name.startswith('implementation/uni/')]
    assert json.loads((repository / 'UPSTREAM.lock.json').read_text())['commit'] == 'faf7818e8abac3ec6abb8f00e6e904d3577f800b'
    for relative in extra_sources + vendored:
        path = repository / relative
        assert path.resolve().is_relative_to(repository) and sha(path) == lookup[relative]['public_sha256'], relative
    artifacts = json.loads((repository / 'artifacts/manifest.json').read_text(encoding='utf8'))['checkpoints']
    selected = next(row for row in artifacts if row['name'] == args.name)
    checkpoint = (args.checkpoint_file or repository / selected['path']).resolve()
    assert checkpoint.is_file() and checkpoint.stat().st_size == selected['bytes']
    assert sha(checkpoint) == selected['sha256'], 'Fetch the exact named release artifact'
    import torch
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert saved['pi']['obs_norm.mean'].numel() == 101
    parent_control = args.name == 'retention-parent-stage3-full'
    if parent_control:
        assert selected['sha256'] == plan['initial_full_sha256']
    else:
        assert saved['variant']['action_output_bound'] == [-20, 20]
        assert saved['variant']['retention_trial_plan_sha256'] == original_plan_sha
    del saved
    verification = {'status': 'released_checkpoint_and_original_public_source_lineage_verified',
                    'checkpoint_name': args.name, 'checkpoint_sha256': selected['sha256'],
                    'original_trial_plan_sha256': original_plan_sha, 'public_trial_plan_sha256': sha(plan_path),
                    'guarded_source_files': len(plan['files']) + len(extra_sources) + len(vendored), 'physics_or_quality_assessed': False}
    if args.verify_only:
        print(json.dumps(verification, indent=2))
        return
    assert args.mode, 'Choose a fresh --mode; preserve all prior runs'
    assert args.mode.replace('-', '').replace('_', '').isalnum()
    os.environ.update(SUPERMIMIC_UNI=str(repository / 'implementation/uni'), HF_HUB_OFFLINE='1',
                      WANDB_MODE='offline', MUJOCO_GL='disable')
    sys.path.insert(0, str(root / 'scripts'))
    sys.path.insert(0, str(repository / 'implementation/uni/src'))
    import runtime
    assert runtime.ROOT == root
    from evaluate_action_bound import BoundedPolicy
    evaluator = importlib.import_module('evaluate_library' if args.suite == 'library' else
                                       'evaluate_sequences_corrected' if args.reference_variant == 'corrected' else 'evaluate_sequences')
    if args.suite == 'library' and args.reference_variant == 'corrected':
        from stage2_references import references
        evaluator.references = references
    native_load = evaluator.load_policy
    loaded = []

    def load(environment, weights, device=None):
        policy = BoundedPolicy(native_load(environment, weights, device), environment.supermimic_mdp.motion.clip_len - 1).eval()
        loaded.append(policy)
        return policy

    evaluator.load_policy = load
    forwarded = ['--checkpoint', str(checkpoint), '--mode', args.mode, '--device', args.device]
    if args.suite == 'library':
        forwarded += ['--group', args.group]
    sys.argv = [sys.argv[0], *forwarded]
    evaluator.main()
    assert len(loaded) == 1 and sha(checkpoint) == selected['sha256']
    path = root / 'runs' / args.mode / 'summary.json'
    summary = json.loads(path.read_text(encoding='utf8'))
    summary.update(controller_variant='action-output-bound-20-v1' if parent_control else 'consistent-output-bound-20-retention-v1',
                   reference_variant='raw' if args.reference_variant == 'raw' else 'ground-calibration-v1',
                   frozen_parent_control=parent_control, action_output_bound=[-20, 20], one_student_policy=True,
                   deployed_teachers=0, retention_trial_plan_sha256=original_plan_sha,
                   publication_trial_plan_sha256=sha(plan_path), release_lineage_verification=verification)
    policy = loaded[0]
    names = [row['clip'] for row in summary['clips']] if args.suite == 'library' else [f'seed-{row["seed"]}' for row in summary['trials']]
    summary['action_bound_observations'] = [
        {'clip': name, 'active_action_elements': int(total), 'clipped_elements': int(clipped), 'peak_raw_output': float(peak)}
        for name, total, clipped, peak in zip(names, policy.active_elements.cpu(), policy.clipped_elements.cpu(), policy.peak_raw.cpu())]
    path.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf8')
    print('RELEASED_RETENTION_EVALUATION_COMPLETE ' + json.dumps({'mode': args.mode, 'checkpoint_sha256': selected['sha256'],
                                                               'post_initialization_resets': summary['post_initialization_resets']}))


if __name__ == '__main__':
    main()
