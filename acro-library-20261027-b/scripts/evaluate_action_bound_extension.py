"""Complete the frozen same-weight bound ablation on raw clips and fixed sequences."""
import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path
from runtime import ROOT
from evaluate_action_bound import BoundedPolicy


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--suite', choices=['library', 'sequences'], required=True)
    parser.add_argument('--reference-variant', choices=['raw', 'corrected'], required=True)
    args, remaining = parser.parse_known_args()
    plan_path = ROOT / 'action-bound-consistency-v1-extension-plan.json'
    plan = json.loads(plan_path.read_text(encoding='utf8'))
    for name, digest in plan['files'].items():
        assert sha(ROOT / name) == digest, name
    checkpoint = Path(remaining[remaining.index('--checkpoint') + 1]).resolve()
    assert sha(checkpoint) == plan['checkpoint_sha256']
    mode = remaining[remaining.index('--mode') + 1]
    assert mode in plan['runs'], mode
    expected = plan['runs'][mode]
    assert expected['suite'] == args.suite and expected['reference_variant'] == args.reference_variant
    module_name = ('evaluate_library' if args.suite == 'library' else
                   'evaluate_sequences_corrected' if args.reference_variant == 'corrected' else 'evaluate_sequences')
    evaluator = importlib.import_module(module_name)
    if args.suite == 'library' and args.reference_variant == 'corrected':
        from stage2_references import references
        evaluator.references = references
    native = evaluator.load_policy
    loaded = []
    def load(env, saved, device=None):
        policy = BoundedPolicy(native(env, saved, device), env.supermimic_mdp.motion.clip_len - 1).eval()
        loaded.append(policy)
        return policy
    evaluator.load_policy = load
    sys.argv = [sys.argv[0], *remaining]
    evaluator.main()
    assert len(loaded) == 1 and sha(checkpoint) == plan['checkpoint_sha256']
    path = ROOT / 'runs' / mode / 'summary.json'
    result = json.loads(path.read_text(encoding='utf8'))
    policy = loaded[0]
    names = ([row['clip'] for row in result['clips']] if args.suite == 'library' else
             [f'seed-{row["seed"]}' for row in result['trials']])
    result.update(controller_variant='action-output-bound-20-v1',
                  reference_variant='ground-calibration-v1' if args.reference_variant == 'corrected' else 'raw',
                  controlled_ablation_only=True, raw_library_result=args.reference_variant == 'raw',
                  training_updates_in_this_ablation=0, same_checkpoint_weights_unchanged=True,
                  action_bound=[-20, 20], extension_plan_sha256=sha(plan_path))
    result['action_bound_observations'] = [
        {'clip': name, 'active_action_elements': int(total), 'clipped_elements': int(clipped), 'peak_raw_output': float(peak)}
        for name, total, clipped, peak in zip(names, policy.active_elements.cpu(), policy.clipped_elements.cpu(), policy.peak_raw.cpu())]
    path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')


if __name__ == '__main__':
    main()
