"""A predeclared single-motion specialist control; never a unified-library score."""
import hashlib
import json
import sys
from pathlib import Path
from runtime import ROOT
from evaluate_action_bound import BoundedPolicy
import evaluate_library

def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    plan_path=ROOT/'evidence/specialist-teacher-8808-1030-plan.json'
    plan=json.loads(plan_path.read_text())
    variant=sys.argv[1];assert variant in ['raw','corrected']
    checkpoint=(ROOT/plan['checkpoint']).resolve();assert sha(checkpoint)==plan['checkpoint_sha256']
    for name,digest in plan['source_sha256'].items():assert sha(ROOT/name)==digest,name
    if variant=='corrected':
        from stage2_references import references
        evaluate_library.references=references
    native=evaluate_library.load_policy;loaded=[]
    def load(env,saved,device=None):
        assert saved['pi']['obs_norm.mean'].numel()==99
        policy=BoundedPolicy(native(env,saved,device),env.supermimic_mdp.motion.clip_len-1).eval();loaded.append(policy);return policy
    evaluate_library.load_policy=load
    mode=plan['modes'][variant]
    sys.argv=[sys.argv[0],'--checkpoint',str(checkpoint),'--mode',mode,'--device','cpu','--group','new','--clips',plan['clip']]
    evaluate_library.main()
    assert len(loaded)==1 and sha(checkpoint)==plan['checkpoint_sha256']
    path=ROOT/f'runs/{mode}/summary.json';summary=json.loads(path.read_text());policy=loaded[0]
    assert len(summary['clips'])==1 and summary['post_initialization_resets']==0
    summary.update(controller_variant='action-output-bound-20-v1',reference_variant='raw' if variant=='raw' else 'ground-calibration-v1',
                   specialist_control_only=True,not_a_unified15_or55_result=True,training_updates_in_this_diagnostic=0,action_output_bound=[-20,20],
                   action_bound_observations=[{'clip':plan['clip'],'active_action_elements':int(policy.active_elements[0]),
                                               'clipped_elements':int(policy.clipped_elements[0]),'peak_raw_output':float(policy.peak_raw[0])}])
    path.write_text(json.dumps(summary,indent=2)+'\n')

if __name__=='__main__':main()
