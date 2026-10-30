"""Bounded physical controls for the separately frozen qualified-expert hypothesis."""
import argparse,hashlib,importlib,json,sys
from pathlib import Path
import torch
from runtime import ROOT
from evaluate_action_bound import BoundedPolicy

def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--role',choices=['teacher','candidate'],required=True)
    parser.add_argument('--suite',choices=['library','sequences'],required=True)
    parser.add_argument('--reference-variant',choices=['raw','corrected'],required=True)
    args,remaining=parser.parse_known_args()
    plan_path=ROOT/'qualified-expert-v2-plan.json';plan=json.loads(plan_path.read_text())
    checkpoint=Path(remaining[remaining.index('--checkpoint')+1]).resolve();saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if args.role=='teacher':
        assert sha(checkpoint)==plan['new_teacher_sha256'] and args.suite=='library'
        assert remaining[remaining.index('--clips')+1]=='88_08_stageii_unitree_g1'
    else:
        assert saved['variant']['qualified_expert_plan_sha256']==sha(plan_path)
        assert saved['variant']['action_output_bound']==[-20,20]
    evaluator=importlib.import_module('evaluate_library' if args.suite=='library' else 'evaluate_sequences_corrected' if args.reference_variant=='corrected' else 'evaluate_sequences')
    if args.suite=='library' and args.reference_variant=='corrected':
        from stage2_references import references
        evaluator.references=references
    native=evaluator.load_policy;loaded=[]
    def load(env,weights,device=None):
        policy=BoundedPolicy(native(env,weights,device),env.supermimic_mdp.motion.clip_len-1).eval();loaded.append(policy);return policy
    evaluator.load_policy=load;sys.argv=[sys.argv[0],*remaining];evaluator.main();assert len(loaded)==1
    mode=remaining[remaining.index('--mode')+1];path=ROOT/f'runs/{mode}/summary.json';summary=json.loads(path.read_text());policy=loaded[0]
    summary.update(controller_variant='qualified-expert-v2-bound20',reference_variant='raw' if args.reference_variant=='raw' else 'ground-calibration-v1',
                   action_output_bound=[-20,20],one_deployed_policy=True,qualified_expert_plan_sha256=sha(plan_path),role=args.role)
    names=[row['clip'] for row in summary['clips']] if args.suite=='library' else [f'seed-{row["seed"]}' for row in summary['trials']]
    summary['action_bound_observations']=[{'clip':name,'active_action_elements':int(total),'clipped_elements':int(clipped),'peak_raw_output':float(peak)}
        for name,total,clipped,peak in zip(names,policy.active_elements.cpu(),policy.clipped_elements.cpu(),policy.peak_raw.cpu())]
    path.write_text(json.dumps(summary,indent=2)+'\n')

if __name__=='__main__':main()
