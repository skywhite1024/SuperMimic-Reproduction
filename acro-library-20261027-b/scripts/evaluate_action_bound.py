"""Frozen same-weight action-bound ablation; preserve all unbounded evaluations."""
import hashlib,json,sys
from pathlib import Path
import torch
from runtime import ROOT
from stage2_references import references
import evaluate_library


class BoundedPolicy(torch.nn.Module):
    def __init__(self,policy,lengths):
        super().__init__();self.policy=policy;self.lengths=lengths;self.calls=0
        self.active_elements=torch.zeros_like(lengths);self.clipped_elements=torch.zeros_like(lengths)
        self.peak_raw=torch.zeros_like(lengths,dtype=torch.float32)

    def forward(self,history,goal):
        raw=self.policy(history,goal);active=self.calls<self.lengths
        self.active_elements.add_(active.long()*raw.shape[-1])
        self.clipped_elements.add_(((raw.abs()>20)&active[:,None]).sum(-1))
        self.peak_raw=torch.maximum(self.peak_raw,torch.where(active,raw.abs().amax(-1),0))
        self.calls+=1
        return raw.clamp(-20,20)


def main():
    plan=ROOT/'action-bound-consistency-v1-plan.json';frozen=json.loads(plan.read_text())
    checkpoint=Path(sys.argv[sys.argv.index('--checkpoint')+1]).resolve()
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==frozen['checkpoint_sha256']
    loaded=[];native=evaluate_library.load_policy
    def load(env,saved,device=None):
        policy=BoundedPolicy(native(env,saved,device),env.supermimic_mdp.motion.clip_len-1).eval();loaded.append(policy);return policy
    evaluate_library.references=references;evaluate_library.load_policy=load
    evaluate_library.main()
    assert len(loaded)==1
    mode=sys.argv[sys.argv.index('--mode')+1];path=ROOT/'runs'/mode/'summary.json';result=json.loads(path.read_text())
    policy=loaded[0]
    result.update(controller_variant='action-output-bound-20-v1',reference_variant='ground-calibration-v1',
                  controlled_ablation_only=True,raw_library_result=False,training_updates_in_this_ablation=0,
                  same_checkpoint_weights_unchanged=True,action_bound=[-20,20],plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest())
    result['action_bound_observations']=[{'clip':row['clip'],'active_action_elements':int(total),'clipped_elements':int(clipped),'peak_raw_output':float(peak)}
        for row,total,clipped,peak in zip(result['clips'],policy.active_elements.cpu(),policy.clipped_elements.cpu(),policy.peak_raw.cpu())]
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==frozen['checkpoint_sha256']
    path.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
