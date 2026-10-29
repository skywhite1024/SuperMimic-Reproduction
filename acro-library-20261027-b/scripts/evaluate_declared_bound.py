"""Explicit bound20 evaluator for the retention variant and its frozen parent control."""
import argparse,hashlib,importlib,json,sys
from pathlib import Path
import torch
from runtime import ROOT
from evaluate_action_bound import BoundedPolicy


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--suite',choices=['library','sequences'],required=True)
    parser.add_argument('--reference-variant',choices=['raw','corrected'],required=True)
    parser.add_argument('--parent-control',action='store_true')
    args,remaining=parser.parse_known_args()
    plan_path=ROOT/'retention-trial-v1-plan.json';plan=json.loads(plan_path.read_text(encoding='utf8'))
    checkpoint=Path(remaining[remaining.index('--checkpoint')+1]).resolve()
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if args.parent_control:
        assert digest==plan['initial_full_sha256']
    else:
        assert saved['variant']['action_output_bound']==[-20,20]
        assert saved['variant']['retention_trial_plan_sha256']==hashlib.sha256(plan_path.read_bytes()).hexdigest()
    del saved
    evaluator=importlib.import_module('evaluate_library' if args.suite=='library' else 'evaluate_sequences_corrected' if args.reference_variant=='corrected' else 'evaluate_sequences')
    if args.suite=='library' and args.reference_variant=='corrected':
        from stage2_references import references
        evaluator.references=references
    loaded=[];native=evaluator.load_policy
    def load(env,saved,device=None):
        policy=BoundedPolicy(native(env,saved,device),env.supermimic_mdp.motion.clip_len-1).eval();loaded.append(policy);return policy
    evaluator.load_policy=load;sys.argv=[sys.argv[0],*remaining];evaluator.main()
    assert len(loaded)==1 and hashlib.sha256(checkpoint.read_bytes()).hexdigest()==digest
    mode=remaining[remaining.index('--mode')+1];path=ROOT/'runs'/mode/'summary.json';result=json.loads(path.read_text())
    result.update(controller_variant='action-output-bound-20-v1' if args.parent_control else 'consistent-output-bound-20-retention-v1',
                  reference_variant='raw' if args.reference_variant=='raw' else 'ground-calibration-v1',
                  frozen_parent_control=args.parent_control,action_output_bound=[-20,20],one_student_policy=True,deployed_teachers=0,
                  retention_trial_plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest())
    policy=loaded[0]
    names=[row['clip'] for row in result['clips']] if args.suite=='library' else [f'seed-{row["seed"]}' for row in result['trials']]
    result['action_bound_observations']=[{'clip':name,'active_action_elements':int(total),'clipped_elements':int(clipped),'peak_raw_output':float(peak)}
        for name,total,clipped,peak in zip(names,policy.active_elements.cpu(),policy.clipped_elements.cpu(),policy.peak_raw.cpu())]
    path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':main()
