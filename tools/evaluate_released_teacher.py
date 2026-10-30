"""Portable bound20 teacher controls with exact release and source provenance guards."""
import argparse,hashlib,importlib,json,os,sys
from pathlib import Path

def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--repository',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--name',choices=['stage2-full','acro88-08'],required=True)
    parser.add_argument('--checkpoint-file',type=Path)
    parser.add_argument('--reference-variant',choices=['raw','corrected'],default='raw')
    parser.add_argument('--mode');parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args();repository=args.repository.resolve();root=repository/'acro-library-20261027-b'
    provenance=json.loads((repository/'SOURCE_PROVENANCE.json').read_text(encoding='utf8'))
    lookup={row['path']:row for row in provenance['files']}
    names=[name for name in lookup if name.startswith('implementation/uni/')]
    names+=['acro-library-20261027-b/scripts/'+name for name in ['runtime.py','global_model.py','evaluate_library.py','evaluate_action_bound.py','stage2_references.py','assess_tracking.py','audit_physical_contacts.py','combine_acceptance.py']]
    for name in names:
        assert sha(repository/name)==lookup[name]['public_sha256'],name
    assert json.loads((repository/'UPSTREAM.lock.json').read_text())['commit']=='faf7818e8abac3ec6abb8f00e6e904d3577f800b'
    artifact=next(row for row in json.loads((repository/'artifacts/manifest.json').read_text())['checkpoints'] if row['name']==args.name)
    expected={'stage2-full':'807d5fac7ca128f892c950fa379133231cd5f741e83075ed017d105b2eec6eec','acro88-08':'0f4a7bc67ae9e3440711e401e37f8276102e59946177117d082a6d7aabe108b4'}
    assert artifact['sha256']==expected[args.name]
    checkpoint=(args.checkpoint_file or repository/artifact['path']).resolve()
    assert checkpoint.stat().st_size==artifact['bytes'] and sha(checkpoint)==artifact['sha256']
    os.environ.update(SUPERMIMIC_UNI=str(repository/'implementation/uni'),MUJOCO_GL='disable',HF_HUB_OFFLINE='1',WANDB_MODE='offline',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    import torch
    torch.set_num_threads(2)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    dimension=saved['pi']['obs_norm.mean'].numel();assert dimension==(101 if args.name=='stage2-full' else 99)
    verification={'status':'pinned_released_teacher_and_public_source_verified','name':args.name,'checkpoint_sha256':artifact['sha256'],'observation_dimension':dimension,'guarded_source_files':len(names),'physics_assessed':False}
    if args.verify_only:print(json.dumps(verification,indent=2));return
    assert args.mode and args.mode.replace('-','').replace('_','').isalnum()
    sys.path.insert(0,str(root/'scripts'));sys.path.insert(0,str(repository/'implementation/uni/src'))
    import runtime
    assert runtime.ROOT==root
    from evaluate_action_bound import BoundedPolicy
    evaluator=importlib.import_module('evaluate_library')
    if args.reference_variant=='corrected':
        from stage2_references import references
        evaluator.references=references
    native=evaluator.load_policy;loaded=[]
    def load(env,weights,device=None):
        policy=BoundedPolicy(native(env,weights,device),env.supermimic_mdp.motion.clip_len-1).eval();loaded.append(policy);return policy
    evaluator.load_policy=load
    sys.argv=[sys.argv[0],'--checkpoint',str(checkpoint),'--mode',args.mode,'--device','cpu','--group','new']
    if args.name=='acro88-08':sys.argv+=['--clips','88_08_stageii_unitree_g1']
    evaluator.main()
    assert len(loaded)==1 and sha(checkpoint)==artifact['sha256']
    path=root/f'runs/{args.mode}/summary.json';summary=json.loads(path.read_text());policy=loaded[0]
    summary.update(controller_variant='action-output-bound-20-v1',reference_variant='raw' if args.reference_variant=='raw' else 'ground-calibration-v1',
                   teacher_qualification_only=True,training_updates_in_this_diagnostic=0,action_output_bound=[-20,20],
                   observation_dimension=dimension,one_deployed_policy=True,release_verification=verification,
                   not_a_unified15_or55_result=args.name=='acro88-08')
    summary['action_bound_observations']=[{'clip':row['clip'],'active_action_elements':int(total),'clipped_elements':int(clipped),'peak_raw_output':float(peak)}
        for row,total,clipped,peak in zip(summary['clips'],policy.active_elements.cpu(),policy.clipped_elements.cpu(),policy.peak_raw.cpu())]
    path.write_text(json.dumps(summary,indent=2)+'\n')
    print('RELEASED_TEACHER_COMPLETE '+json.dumps({'name':args.name,'new_denominator':len(summary['clips']),'post_t0_resets':summary['post_initialization_resets']}))

if __name__=='__main__':main()
