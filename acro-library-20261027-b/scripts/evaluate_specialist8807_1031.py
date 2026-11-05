"""Frozen raw/calibrated physical evaluation of parent and independent 99D specialist."""
import argparse, hashlib, json, sys
from pathlib import Path
import torch
from runtime import ROOT
from evaluate_action_bound import BoundedPolicy
import evaluate_library as evaluator

def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--role',choices=['parent','candidate'],required=True)
    parser.add_argument('--reference-variant',choices=['raw','corrected'],required=True)
    args,remaining=parser.parse_known_args()
    checkpoint=Path(remaining[remaining.index('--checkpoint')+1]);saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert saved['pi']['obs_norm.mean'].numel()==99
    if args.role=='parent':
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()=='0afcca9af97bd9db1f2dd083875898f962acab0b0a70c7313b221e575b70b05b'
    else:
        assert saved['variant']['target_clip']=='88_07_stageii_unitree_g1'
        assert saved['variant']['specialist_plan_sha256']==hashlib.sha256((ROOT/'specialist8807-1031d-plan.json').read_bytes()).hexdigest()
    if args.reference_variant=='corrected':
        from stage2_references import references
        evaluator.references=references
    native=evaluator.load_policy;loaded=[]
    def load(env,weights,device=None):
        policy=BoundedPolicy(native(env,weights,device),env.supermimic_mdp.motion.clip_len-1).eval();loaded.append(policy);return policy
    evaluator.load_policy=load;sys.argv=[sys.argv[0],*remaining];evaluator.main();assert len(loaded)==1
    mode=remaining[remaining.index('--mode')+1];path=ROOT/f'runs/{mode}/summary.json';summary=json.loads(path.read_text());policy=loaded[0]
    summary.update(controller_variant='specialist8807-bound20',reference_variant='raw' if args.reference_variant=='raw' else 'ground-calibration-v1',
                   action_output_bound=[-20,20],one_deployed_policy=True,independent_specialist_not_unified=True,role=args.role)
    summary['action_bound_observations']=[{'clip':row['clip'],'active_action_elements':int(total),'clipped_elements':int(clipped),'peak_raw_output':float(peak)}
        for row,total,clipped,peak in zip(summary['clips'],policy.active_elements.cpu(),policy.clipped_elements.cpu(),policy.peak_raw.cpu())]
    path.write_text(json.dumps(summary,indent=2)+'\n')
if __name__=='__main__':main()
