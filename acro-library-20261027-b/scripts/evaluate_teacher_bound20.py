"""Qualify the pinned stage2 teacher using the unchanged full-clip evaluator."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import torch
from runtime import ROOT
from evaluate_action_bound import BoundedPolicy
import evaluate_library

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--reference-variant', choices=['raw','corrected'], required=True)
    args, remaining = parser.parse_known_args()
    plan_path = ROOT/'evidence/retention-1030-next-diagnostic-plan.json'
    plan = json.loads(plan_path.read_text(encoding='utf8'))
    checkpoint = Path(remaining[remaining.index('--checkpoint')+1]).resolve()
    assert checkpoint == (ROOT/plan['checkpoint']).resolve()
    assert digest(checkpoint) == plan['checkpoint_sha256']
    assert remaining[remaining.index('--group')+1] == 'new'
    assert remaining[remaining.index('--device')+1] == 'cpu' and '--clips' not in remaining
    if args.reference_variant == 'corrected':
        from stage2_references import references
        evaluate_library.references = references
    loaded = []
    native = evaluate_library.load_policy
    def load(env, saved, device=None):
        assert saved['pi']['obs_norm.mean'].numel() == 101
        policy = BoundedPolicy(native(env,saved,device),env.supermimic_mdp.motion.clip_len-1).eval()
        loaded.append(policy)
        return policy
    evaluate_library.load_policy = load
    sys.argv = [sys.argv[0], *remaining]
    evaluate_library.main()
    assert len(loaded) == 1 and digest(checkpoint) == plan['checkpoint_sha256']
    mode = remaining[remaining.index('--mode')+1]
    path = ROOT/'runs'/mode/'summary.json'
    result = json.loads(path.read_text(encoding='utf8'))
    assert len(result['clips']) == 15 and result['post_initialization_resets'] == 0
    result.update(controller_variant='action-output-bound-20-v1',
                  reference_variant='raw' if args.reference_variant == 'raw' else 'ground-calibration-v1',
                  teacher_qualification_only=True, training_updates_in_this_diagnostic=0,
                  action_output_bound=[-20,20], one_policy=True, deployed_policies=1,
                  weights_unchanged=True, qualification_plan_sha256=digest(plan_path))
    policy = loaded[0]
    result['action_bound_observations'] = [
        {'clip':row['clip'],'active_action_elements':int(total),'clipped_elements':int(clipped),'peak_raw_output':float(peak)}
        for row,total,clipped,peak in zip(result['clips'],policy.active_elements.cpu(),policy.clipped_elements.cpu(),policy.peak_raw.cpu())]
    path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')

if __name__ == '__main__':
    main()
