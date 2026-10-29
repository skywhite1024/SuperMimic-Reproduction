"""Prepare one bounded1000-update retention trial and a matching6-update CPU check."""
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    target=ROOT/'retention-trial-v1-plan.json';assert not target.exists()
    proof=json.loads((ROOT/'runs/engineering-retention-cpu-6/proof.json').read_text())
    assert proof['status']=='updates_and_save_readback_verified' and proof['new_updates']==6 and proof['retention']['teacher_parameters_and_norms_unchanged']
    text=(ROOT/'scripts/train_replay_stage3.py').read_text(encoding='utf8')
    text=text.replace('from unisupermimic.rl.algos.supermimic.learner import SuperMimicConfig,SuperMimicLearner',
        'from unisupermimic.rl.algos.supermimic.learner import SuperMimicConfig\nfrom retention_learner import RetentionLearner as SuperMimicLearner\nfrom retention_components import install_action_bound,load_teacher')
    text=text.replace("default=4000)","default=1000)").replace("default=400);p.add_argument('--num-envs'","default=200);p.add_argument('--num-envs'")
    text=text.replace("ROOT/'stage3-plan.json'","ROOT/'retention-trial-v1-plan.json'")
    anchor="    assert not (out/'args.json').exists(),'Choose a fresh label; preserve every prior run'"
    assert text.count(anchor)==1
    text=text.replace(anchor,anchor+'''
    expected=(6,2,64,'cpu') if a.engineering else (1000,200,1024,'cuda')
    assert (a.steps,a.wm_warmup,a.num_envs,a.device)==expected
    for name,expected_digest in plan['files'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected_digest,name
    source=ROOT.parent/'implementation/uni/src/unisupermimic/rl/algos/supermimic/learner.py' if a.device=='cpu' else Path('/mnt/afs/USER_ID/SuperMimic/lafan1-20261024/uni/src/unisupermimic/rl/algos/supermimic/learner.py')
    assert hashlib.sha256(source.read_bytes()).hexdigest()==plan['native_learner_sha256']
    teacher_paths={key:(ROOT/plan[key+'_local'] if a.device=='cpu' else Path(plan[key+'_cloud'])) for key in ['old_teacher','new_teacher']}
    for key,path in teacher_paths.items():assert hashlib.sha256(path.read_bytes()).hexdigest()==plan[key+'_sha256']
''')
    anchor='    replay.install(learner)';assert text.count(anchor)==1
    text=text.replace(anchor,anchor+'''
    def optimizer_steps(optimizer):return {str(key):float(value['step']) for key,value in optimizer.state_dict()['state'].items() if 'step' in value}
    optimizers={'wm':learner.opt_wm,'pi':learner.opt_pi,'critic':learner.opt_v}
    steps_before={name:optimizer_steps(optimizer) for name,optimizer in optimizers.items()}
    learner.install_retention(teacher_paths['old_teacher'],teacher_paths['new_teacher'],[path.stem for path in references('new')],plan['protected_new'],plan['old_weight'],plan['new_weight'])
    install_action_bound(learner.pi)
''')
    text=text.replace("'name':'unified-55-reference-tracker-101d-balanced-replay-stage3'","'name':'unified-101d-bound20-training-only-retention-v1'")
    text=text.replace("'stage3_plan_sha256':", "'retention_trial_plan_sha256':")
    text=text.replace("'BONES_downloaded':False,", "'BONES_downloaded':False,'action_output_bound':[-20,20],'controller_variant':'consistent-output-bound-20-retention-v1','training_only_teacher_anchors':True,'deployed_teachers':0,'actor_objective_changed':True,")
    anchor='        sampling=sampler.receipt();replay_sampling=replay.receipt()';assert text.count(anchor)==1
    text=text.replace(anchor,anchor+'''
        joint_updates=a.steps-a.wm_warmup
        retention=learner.retention_receipt()
        assert retention['action_bound_calls']['grad_calls']==joint_updates*learner.cfg.horizon and retention['action_bound_calls']['no_grad_calls']>0
        assert len(retention['records'])==joint_updates and sum(row['old_samples'] for row in retention['records'])>0 and sum(row['protected_new_samples'] for row in retention['records'])>0
        step_deltas={}
        expected_steps={'wm':a.steps*learner.cfg.wm_batches,'pi':joint_updates,'critic':joint_updates*learner.cfg.critic_epochs*learner.cfg.critic_minibatches}
        for name,optimizer in optimizers.items():
            after_steps=optimizer_steps(optimizer);assert after_steps.keys()==steps_before[name].keys()
            deltas={key:after_steps[key]-steps_before[name][key] for key in after_steps}
            assert set(deltas.values())=={float(expected_steps[name])},(name,set(deltas.values()))
            step_deltas[name]={'all_parameter_step_deltas':expected_steps[name],'parameter_states':len(deltas)}
        reloaded=load_teacher(out/'final-full.pt',a.device);install_action_bound(reloaded)
        with torch.inference_mode():
            history=learner.obs_hist[:4];goal=learner.mdp.goal(learner._state_info()['f'][:4],learner.cfg.goal_frames)
            actual=learner.pi(history,goal);restored=reloaded(history,goal)
            assert torch.equal(actual,restored) and bool((actual.abs()<=20).all())
        functional_readback={'real_collected_history_rows':len(history),'bound_applied_after_loading':True,'exact_output_equality':True}
''')
    text=text.replace("'wall_seconds':time.perf_counter()-start,'sampling':sampling,", "'wall_seconds':time.perf_counter()-start,'retention':retention,'optimizer_step_deltas':step_deltas,'functional_bounded_policy_readback':functional_readback,'sampling':sampling,")
    trainer=ROOT/'scripts/train_retention_trial.py';assert not trainer.exists();trainer.write_text(text,encoding='utf8')
    engineering=json.loads((ROOT/'retention-engineering-plan.json').read_text())
    files=['scripts/retention_components.py','scripts/retention_learner.py','scripts/train_retention_trial.py','scripts/evaluate_declared_bound.py',
           'scripts/balanced_replay.py','scripts/balanced_sampling.py','scripts/global_model.py','scripts/stage2_references.py',
           'scripts/evaluate_library.py','scripts/evaluate_sequences.py','scripts/evaluate_sequences_corrected.py','scripts/evaluate_action_bound.py',
           'scripts/assess_tracking.py','scripts/audit_physical_contacts.py','scripts/combine_acceptance.py','scripts/assess_sequences.py',
           'ground-calibration-v1/manifest.json','acceptance-contract.json','contact-acceptance-contract.json','continuous-acceptance-contract-v1.json']
    for name in files:
        if name.endswith('.py'):compile((ROOT/name).read_text(encoding='utf8'),name,'exec')
    plan={'status':'frozen_before_v2_engineering_and_cuda_candidate_assessment','frozen_at':datetime.now(timezone.utc).isoformat(),
          'scope':'One1000-update candidate,200WM-only then800joint. No automatic extension; distinct controller/objective variant.',
          'initial_full_sha256':engineering['initial_full_sha256'],'initial_full':'runs/unified-h100-stage3-4000/final-full.pt',
          'old_teacher_local':engineering['old_teacher'],'new_teacher_local':engineering['new_teacher'],
          'old_teacher_cloud':'/mnt/afs/USER_ID/SuperMimic/lafan1-20261024/experiments/continuation-20261025-a/policy/full_016000.pt',
          'new_teacher_cloud':'/mnt/afs/USER_ID/SuperMimic/acro-library-20261027-b/runs/unified-h100-stage2-4000/final-full.pt',
          'old_teacher_sha256':engineering['old_teacher_sha256'],'new_teacher_sha256':engineering['new_teacher_sha256'],
          'native_learner_sha256':engineering['pinned_native_source_sha256'],'protected_new':engineering['protected_new'],
          'old_weight':.1,'new_weight':.05,'action_output_bound':[-20,20],
          'old_failure_replay_priorities':engineering['old_failure_replay_priorities'],
          'files':{name:sha(ROOT/name) for name in files},'gpu_count':1,'maximum_program_seconds':5400,'num_envs':1024,
          'new_denominator':15,'old_denominator':40,'fixed_continuous_trials':3,'seconds_per_continuous_trial':120,
          'replay_mass':{'new':.6,'lafan':.4},'teachers_training_only':True,'deployed_student_policies':1,
          'weights_vs_deployment_bound_confounding_control':'Before updates, assess the frozen stage3 parent with identicalbound20 on CUDA55 and raw15; retain unboundedCPU/CUDA and same-weightCPU ablations.',
          'reference_or_physics_changes':False,'frozen_eval_sequences_must_never_be_training_inputs':True,
          'new_cloud_action_prerequisites':['Current academy location/network condition','Fresh all-account running/queued GPU and CCI resource check','Same-source6real CPU updates, fixed teachers/norms, one PI optimizer step per joint update and exact bounded policy save-readback'],
          'continue_or_rollback_after_candidate':'No automatic second training block. Require strict correctedCUDA count not below the matched bound20 parent AND either more corrected strict passes or fewer failed old clips. Otherwise classify this candidate as unsuccessful and retain the previous weights. Goal completion still requires all original frozen acceptance criteria on all55 plus all3 full120s sequences, including matchedCPU regression checks.',
          'early_stop':'Abort on hash/contract mismatch, nonfinite losses, changed teachers/norms, incorrect optimizer deltas, missing replay groups/clips or failed readback. Save interrupted checkpoint and exit; never keep GPU alive.',
          'current_evidence':'Deployment bounds leave all3 first continuous failures unchanged; stage3 regressed flips/oldclips. Retention is a separate training hypothesis, not a proven cure for reference/contact issues.'}
    target.write_text(json.dumps(plan,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'plan_sha256':sha(target),'trainer_sha256':sha(trainer),'status':'prepared_not_submitted','maximum_new_updates':1000}),flush=True)

if __name__=='__main__':main()
