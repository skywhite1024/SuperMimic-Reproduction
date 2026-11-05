"""One bounded specialist-learning hypothesis; fixed repeats include every failure."""
import hashlib,json
from pathlib import Path

def execute(runner,python,env,root,budget,result,save):
    parent=Path('/mnt/afs/USER_ID/SuperMimic/lafan1-20261024/experiments/continuation-20261025-a/policy/full_016000.pt')
    child=root/'policy/specialist8807-h100-1000-1031d.pt'
    def run(label,script,args,cap):
        return runner.run(label,[str(python),'-u',str(root/'scripts'/script),*args],env,initialization=False,cap=cap)
    def single(role,checkpoint,variant,repetition):
        mode=f'specialist8807-1031d-{role}-{variant}-cuda-r{repetition}'
        run(mode,'evaluate_specialist8807_1031.py',['--role',role,'--reference-variant',variant,'--checkpoint',str(checkpoint),'--mode',mode,'--device','cuda','--group','new','--clips','88_07_stageii_unitree_g1'],180)
        run(mode+'-tracking','assess_tracking.py',['--run',mode],30)
        run(mode+'-contacts','audit_physical_contacts.py',['--run',mode],45)
        run(mode+'-strict','assess_specialist8807_1031.py',['--run',mode],15)
        row=json.loads((root/'runs'/mode/'specialist-assessment.json').read_text())
        row['repetition']=repetition;row['seed']=0;result[role].append(row)
        result['status']=role+'_fixed_repetition_complete';save()
    def boundary(role,checkpoint):
        mode=f'specialist8807-1031d-{role}-corrected-all55-cuda'
        run(mode,'evaluate_specialist8807_1031.py',['--role',role,'--reference-variant','corrected','--checkpoint',str(checkpoint),'--mode',mode,'--device','cuda','--group','all'],900)
        for label,script,cap in [('tracking','assess_tracking.py',120),('contacts','audit_physical_contacts.py',180),('strict','combine_acceptance.py',30)]:
            run(mode+'-'+label,script,['--run',mode],cap)
        summary=json.loads((root/'runs'/mode/'summary.json').read_text());combined=json.loads((root/'runs'/mode/'combined-assessment.json').read_text())
        result['boundary'][role]={'mode':mode,'groups':summary['groups'],'new_strict':combined['combined_engineering_passed'],
                                  'old_failed_clips':[x['clip'] for x in summary['clips'] if x['group']=='lafan' and not x['passed_original_hard_gate']],
                                  'summary_clips':summary['clips'],'independent99D_specialist_not_unified101D':True}
        save()
    result.update(parent=[],candidate=[],boundary={},new_training_updates=0)
    for variant in ['raw','corrected']:
        for repetition in range(3):single('parent',parent,variant,repetition)
    result.update(status='training_started',new_training_updates=None);save()
    training_cap=min(1800,budget.remaining(False)-1750)
    assert training_cap>300,'Insufficient remaining budget for the finite training phase'
    run('specialist8807-training1000','train_specialist8807_1031.py',['--initial',str(parent),'--device','cuda','--label','specialist8807-h100-1000-1031d','--steps','1000','--world-model-warmup','400','--num-envs','512'],training_cap)
    proof=json.loads((root/'runs/specialist8807-h100-1000-1031d/proof.json').read_text())
    assert proof['new_iterations']==1000 and proof['final_iteration']==17000 and proof['normalizer_tensors_unchanged'] and proof['functional_readback']['exact_output_equality']
    result.update(status='training1000_and_export_readback_verified',new_training_updates=1000,training_proof=proof);save()
    for variant in ['raw','corrected']:
        for repetition in range(3):single('candidate',child,variant,repetition)
    if budget.remaining(False)>2700:boundary('parent',parent)
    else:result['boundary']['parent']={'status':'not_run_remaining_budget'};save()
    if budget.remaining(False)>1450:boundary('candidate',child)
    else:result['boundary']['candidate']={'status':'not_run_remaining_budget'};save()
    result['counts']={role:{variant:sum(x['strict_passed'] for x in result[role] if x['reference_variant']==('raw' if variant=='raw' else 'ground-calibration-v1')) for variant in ['raw','corrected']} for role in ['parent','candidate']}
    result['worst']={role:{'penetration_m':min(x['actual_min_after_initial_100ms_m'] for x in result[role]),
                          'root_xy_rmse_m':max(x['tracking']['root_xy_rmse_m'] for x in result[role]),
                          'body_local_mean_m':max(x['tracking']['body_local_mean_m'] for x in result[role]),
                          'unexpected_trunk_contact_samples':max(x['unexpected_trunk_contact_samples'] for x in result[role])} for role in ['parent','candidate']}
    result['status']='training1000_fixed_pre_post_controls_complete_boundary_scope_recorded';save()
    if budget.remaining(False)>120:
        log=run('package-results','package_specialist8807_1031.py',[],110)
        result['archive']=json.loads(next(x.removeprefix('SPECIALIST8807_ARCHIVE ') for x in log.read_text().splitlines() if x.startswith('SPECIALIST8807_ARCHIVE ')))
    else:result['archive']={'status':'not_created_due_to_remaining_budget','AFS_artifacts_preserved':True}
    result['status']='finite_specialist8807_phase_complete_acceptance_and_collection_pending';save()
