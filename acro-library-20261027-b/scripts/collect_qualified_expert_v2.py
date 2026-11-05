"""Own-AFS finite terminal-result collection; preserve zero-update and partial failures."""
import argparse,ast,hashlib,json,sys,tarfile
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PLAN_SHA='bc5e8b397accd4f585732803385516b755538f924afebc9e73fbc205dc763476'

def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--platform-status',choices=['success','failed','stopped'],required=True)
    parser.add_argument('--job-uid',required=True);parser.add_argument('--label',required=True);args=parser.parse_args()
    assert ROOT.resolve().is_relative_to(Path('/mnt/afs/USER_ID'))
    assert args.job_uid.isalnum() and len(args.job_uid)==32 and args.label.replace('-','').isalnum()
    plan_path=ROOT/'qualified-expert-v2-plan.json';assert sha(plan_path)==PLAN_SHA;plan=json.loads(plan_path.read_text())
    for name,digest in plan['files'].items():assert sha(ROOT/name)==digest,name
    receipt_path=ROOT/'evidence/qualified-expert-v2-CUDA-execution.json'
    receipt=json.loads(receipt_path.read_text()) if receipt_path.exists() else {'status':'program_receipt_unavailable'}
    run=ROOT/'runs/unified-qualified-expert-v2-h100-1000';complete=receipt['status']=='finite_qualified_expert_v2_candidate_and_all_CUDA_assessments_complete'
    gate_failed=receipt['status']=='teacher_pretraining_control_failed_no_training'
    if gate_failed:assert receipt['new_training_updates']==0 and not run.exists()
    checkpoints=[]
    if run.exists():
        for path in sorted(run.glob('*.pt')):checkpoints.append({'path':path.relative_to(ROOT).as_posix(),'bytes':path.stat().st_size,'sha256':sha(path)})
    policy=None
    if complete:
        import torch
        proof=json.loads((run/'proof.json').read_text());assert proof['new_updates']==1000 and proof['normalizer_tensors_unchanged']
        assert proof['functional_bounded_policy_readback']['exact_output_equality'] and proof['retention']['teacher_parameters_and_norms_unchanged']
        assert sha(run/'final-full.pt')==receipt['checkpoint_sha256']
        saved=torch.load(run/'final-full.pt',map_location='cpu',weights_only=True)
        assert saved['iteration']==31000 and saved['pi']['obs_norm.mean'].numel()==101
        assert saved['variant']['qualified_expert_plan_sha256']==PLAN_SHA
        policy=ROOT/'policy/qualified-expert-v2-h100-1000.pt';assert not policy.exists(),'Preserve previous collection; inspect it before retry'
        policy.parent.mkdir(exist_ok=True)
        torch.save({key:saved[key] for key in ['pi','args','variant','iteration']},policy)
        exported=torch.load(policy,map_location='cpu',weights_only=True)
        assert all(torch.equal(value,exported['pi'][name]) for name,value in saved['pi'].items())
        fields_source=ROOT.parent/'lafan1-20261024/uni/src/unisupermimic/rl/algos/supermimic/learner.py'
        assert sha(fields_source)==plan['native_learner_sha256']
        fields=next(ast.literal_eval(node.value) for node in ast.parse(fields_source.read_text()).body if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id=='_LOG_FIELDS' for target in node.targets))
        raw=(run/'train_log.csv').read_text();lines=raw.splitlines();header=','.join(fields)
        assert lines and all(len(line.split(','))==len(fields) for line in lines)
        for name,payload in [('train_log_with_header.csv',('' if lines[0]==header else header+'\n')+raw),('train_log_schema.json',json.dumps({'fields':fields,'raw_header_present':lines[0]==header,'raw_preserved':True},indent=2)+'\n')]:
            target=run/name;assert not target.exists();target.write_text(payload)
    evidence=ROOT/'evidence';inventory=evidence/f'qualified-expert-v2-{args.label}-checkpoint-inventory.json';assert not inventory.exists()
    inventory.write_text(json.dumps({'all_checkpoints_retained_on_own_AFS':checkpoints,'program_complete':complete,'teacher_gate_failed_before_training':gate_failed,'policy_export':None if policy is None else {'path':policy.relative_to(ROOT).as_posix(),'bytes':policy.stat().st_size,'sha256':sha(policy)}},indent=2)+'\n')
    selected=set(plan['files'])|{'qualified-expert-v2-plan.json',inventory.relative_to(ROOT).as_posix(),'scripts/collect_qualified_expert_v2.py'}
    for name in ['deployment/qualified-expert-v2.command.txt','deployment/run_qualified_expert_v2_guarded.py']:
        if (ROOT/name).is_file():selected.add(name)
    for pattern in ['qualified-expert-v2-*.json','qualified-expert-v2-*.log']:
        selected.update(path.relative_to(ROOT).as_posix() for path in evidence.glob(pattern) if path.is_file())
    for folder in sorted((ROOT/'runs').glob('qualified-expert-v2-*')):
        if folder.is_dir():selected.update(path.relative_to(ROOT).as_posix() for path in folder.iterdir() if path.is_file() and path.suffix in {'.json','.npz','.csv','.log'})
    if run.exists():
        selected.update(path.relative_to(ROOT).as_posix() for path in run.iterdir() if path.is_file() and path.suffix in {'.json','.npz','.csv','.log'})
        for name in ['final-full.pt','interrupted-full.pt']:
            if (run/name).is_file():selected.add((run/name).relative_to(ROOT).as_posix())
    if policy is not None:selected.add(policy.relative_to(ROOT).as_posix())
    files=[]
    for name in sorted(selected):
        path=ROOT/name;assert path.resolve().is_relative_to(ROOT) and path.is_file() and not path.is_symlink(),name
        files.append({'path':name,'bytes':path.stat().st_size,'sha256':sha(path)})
    manifest={'status':'collected_not_yet_local_transfer_verified','at':datetime.now(timezone.utc).isoformat(),'job_uid':args.job_uid,
              'platform_terminal_status_from_attended_console':args.platform_status,'program_receipt':receipt,'program_complete':complete,
              'teacher_gate_failed_before_training':gate_failed,'candidate_adopted':False,'full_goal_complete':False,
              'all_partial_failure_outputs_retained':True,'intermediate_checkpoints_retained_on_own_AFS':checkpoints,
              'file_count':len(files),'files':files,'raw_motion_or_credentials_included':False}
    manifest_path=evidence/f'qualified-expert-v2-{args.label}-results-manifest.json';assert not manifest_path.exists();manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    archive=ROOT/f'deployment/qualified-expert-v2-{args.label}-results.tar.gz';assert not archive.exists()
    with tarfile.open(archive,'w:gz',compresslevel=3) as bundle:
        for name in [*sorted(selected),manifest_path.relative_to(ROOT).as_posix()]:bundle.add(ROOT/name,arcname=name,recursive=False)
    result={'status':'archive_ready_not_local_transfer_verified','archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),
            'manifest_files':len(files),'program_complete':complete,'teacher_gate_failed_before_training':gate_failed,'full_goal_complete':False,
            'CPU_CCI_manual_stop_empty_DNAT_unbind_original_restore_still_required':True}
    target=evidence/f'qualified-expert-v2-{args.label}-archive-receipt.json';assert not target.exists();target.write_text(json.dumps(result,indent=2)+'\n')
    print('QUALIFIED_EXPERT_V2_RESULTS_ARCHIVE_READY '+json.dumps(result),flush=True)

if __name__=='__main__':main()
