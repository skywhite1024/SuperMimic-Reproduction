"""Freeze one local engineering probe; this file never submits a cloud job."""
import ast
import hashlib
import json
import textwrap
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT.parent/'implementation/uni/src/unisupermimic/rl/algos/supermimic/learner.py'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    plan_path=ROOT/'retention-engineering-plan.json'
    assert not plan_path.exists()
    source=SOURCE.read_text(encoding='utf8');tree=ast.parse(source)
    learner=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='SuperMimicLearner')
    method=next(node for node in learner.body if isinstance(node,ast.FunctionDef) and node.name=='policy_update')
    native='\n'.join(source.splitlines()[method.lineno-1:method.end_lineno])+'\n'
    first='        obs_hist = self.mdp.add_noise(obs_hist)'
    second='            loss = -objective.mean()'
    assert native.count(first)==native.count(second)==1
    changed=native.replace(first,'        retention_history = obs_hist.detach()\n'+first).replace(second,second+' + self.retention_loss(start, retention_history, initial_goal)')
    assert changed.replace('        retention_history = obs_hist.detach()\n','').replace(' + self.retention_loss(start, retention_history, initial_goal)','')==native
    header='''"""Pinned native policy_update plus two explicit retention-objective edits."""
from __future__ import annotations
from typing import cast
import torch
import torch.nn as nn
from unisupermimic.rl.algos.supermimic.learner import SuperMimicLearner, _set_requires_grad
from unisupermimic.rl.algos.supermimic.geometry import unflatten_state
from unisupermimic.rl.contracts.supermimic import State
from retention_components import RetentionMixin


class RetentionLearner(RetentionMixin, SuperMimicLearner):
'''
    target=ROOT/'scripts/retention_learner.py';assert not target.exists();target.write_text(header+changed,encoding='utf8')
    audit={'pinned_learner_sha256':sha(SOURCE),'native_policy_update_sha256':hashlib.sha256(native.encode()).hexdigest(),
           'native_method_restored_byte_identically_by_reversing_two_edits':True,
           'changes':['Keep clean real replay observation history before native observation noise.','Add training-only frozen-teacher action MSE anchors to the existing actor loss, before its single backward/optimizer step.'],
           'native_world_model_critic_and_optimizer_step_logic_preserved':True,'derived_file_sha256':sha(target)}
    (ROOT/'evidence/retention-native-method-diff.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf8')
    training=(ROOT/'scripts/train_replay_stage3.py').read_text(encoding='utf8')
    training=training.replace('from unisupermimic.rl.algos.supermimic.learner import SuperMimicConfig,SuperMimicLearner',
        'from unisupermimic.rl.algos.supermimic.learner import SuperMimicConfig\nfrom retention_learner import RetentionLearner as SuperMimicLearner\nfrom retention_components import install_action_bound')
    training=training.replace("ROOT/'stage3-plan.json'","ROOT/'retention-engineering-plan.json'")
    anchor='    replay.install(learner)'
    assert training.count(anchor)==1
    training=training.replace(anchor,anchor+'''
    assert a.engineering and a.device=='cpu' and a.steps==6 and a.wm_warmup==2 and a.num_envs==64, 'Only the frozen local engineering probe is authorized by this script'
    def optimizer_steps(optimizer):return {str(key):float(value['step']) for key,value in optimizer.state_dict()['state'].items() if 'step' in value}
    optimizers={'wm':learner.opt_wm,'pi':learner.opt_pi,'critic':learner.opt_v}
    steps_before={name:optimizer_steps(optimizer) for name,optimizer in optimizers.items()}
    learner.install_retention(ROOT/plan['old_teacher'],ROOT/plan['new_teacher'],[path.stem for path in references('new')],plan['protected_new'],plan['old_weight'],plan['new_weight'])
    install_action_bound(learner.pi)
''')
    training=training.replace("'name':'unified-55-reference-tracker-101d-balanced-replay-stage3'","'name':'engineering-only-101d-bound20-teacher-retention'")
    training=training.replace("'BONES_downloaded':False,", "'BONES_downloaded':False,'action_output_bound':[-20,20],'training_only_teacher_anchors':True,'deployed_teachers':0,'actor_objective_changed':True,")
    anchor='        sampling=sampler.receipt();replay_sampling=replay.receipt()'
    assert training.count(anchor)==1
    training=training.replace(anchor,anchor+'''
        retention=learner.retention_receipt()
        assert retention['action_bound_calls']['grad_calls']==4*learner.cfg.horizon and retention['action_bound_calls']['no_grad_calls']>0
        assert len(retention['records'])==4 and sum(row['old_samples'] for row in retention['records'])>0 and sum(row['protected_new_samples'] for row in retention['records'])>0
        step_deltas={}
        for name,optimizer in optimizers.items():
            after_steps=optimizer_steps(optimizer);assert after_steps.keys()==steps_before[name].keys()
            deltas={key:after_steps[key]-steps_before[name][key] for key in after_steps}
            expected=6 if name=='wm' else 4
            assert set(deltas.values())=={float(expected)},(name,set(deltas.values()))
            step_deltas[name]={'all_parameter_step_deltas':expected,'parameter_states':len(deltas)}
''')
    training=training.replace("'wall_seconds':time.perf_counter()-start,'sampling':sampling,", "'wall_seconds':time.perf_counter()-start,'retention':retention,'optimizer_step_deltas':step_deltas,'sampling':sampling,")
    target_training=ROOT/'scripts/train_retention_engineering.py';assert not target_training.exists();target_training.write_text(training,encoding='utf8')
    initial='runs/unified-h100-stage3-4000/final-full.pt'
    old='../implementation/continuation-20261025-a/downloaded/policy/full_016000.pt'
    new='runs/unified-h100-stage2-4000/final-full.pt'
    assert sha(ROOT/initial)=='93bb8b830145c54092518eba8db1fb020d020eb72cdad1b619b4fc5470cc213d'
    assert sha(ROOT/old)=='0afcca9af97bd9db1f2dd083875898f962acab0b0a70c7313b221e575b70b05b'
    assert sha(ROOT/new)=='807d5fac7ca128f892c950fa379133231cd5f741e83075ed017d105b2eec6eec'
    files=['scripts/retention_components.py','scripts/retention_learner.py','scripts/train_retention_engineering.py',
           'scripts/balanced_replay.py','scripts/balanced_sampling.py','scripts/global_model.py','scripts/stage2_references.py',
           'ground-calibration-v1/manifest.json','acceptance-contract.json','contact-acceptance-contract.json','continuous-acceptance-contract-v1.json']
    for name in files:
        if name.endswith('.py'):compile((ROOT/name).read_text(encoding='utf8'),name,'exec')
    stage3=json.loads((ROOT/'stage3-plan.json').read_text())
    plan={'status':'frozen_before_one_local_engineering_probe','frozen_at':datetime.now(timezone.utc).isoformat(),
          'scope':'One CPU 6-update implementation/optimizer/readback check. No cloud resources or physical-success claim.',
          'initial_full':initial,'initial_full_sha256':sha(ROOT/initial),'old_teacher':old,'old_teacher_sha256':sha(ROOT/old),
          'new_teacher':new,'new_teacher_sha256':sha(ROOT/new),'protected_new':['88_07_stageii_unitree_g1','88_08_stageii_unitree_g1','88_09_stageii_unitree_g1'],
          'old_weight':.1,'new_weight':.05,'loss':'native imagined actor objective +0.1 mean raw-action MSE to bounded16K teacher on old replay starts +0.05 MSE to bounded stage2 teacher on three regressed flip starts; each group normalized separately',
          'action_bound':[-20,20],'teachers_training_only':True,'deployed_student_policies':1,
          'old_failure_replay_priorities':stage3['old_failure_replay_priorities'],
          'engineering_updates':6,'wm_only_updates':2,'device':'cpu','num_envs':64,
          'files':{name:sha(ROOT/name) for name in files},'pinned_native_source_sha256':sha(SOURCE),
          'basis':'All same-weight deployment bound comparisons preserve initial failures; stage3 forgets old clips and flip landings, while older frozen policies retain those skills. Test an explicit training-only retention mechanism; do not claim bound consistency alone fixes failure.',
          'stop':'Stop on failed source/hash guards, nonfinite loss, missing old/protected-new samples, teacher/norm changes, optimizer-step mismatch, failed exact save-readback, or more than6 updates. No automatic GPU escalation.'}
    plan_path.write_text(json.dumps(plan,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'plan_sha256':sha(plan_path),'native_method_diff':audit,'training_script':str(target_training)}),flush=True)


if __name__=='__main__':main()
