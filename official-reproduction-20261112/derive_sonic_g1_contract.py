"""Resolve published SONIC G1 gains/order and compare training/deploy tables.

Only configuration declarations execute, using plain Python value containers.
Isaac Lab, a simulator and robot connections are never initialized.
"""
import argparse
import ast
import hashlib
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

SOURCE_HASH='42f9b4394eb21512735da4b1ca78c2d0546c4221c7bdaf8b3a186d8dc5588ba9'
PARAMETER_HASH='b9332adf07c2c9b75c9b1e0756e57c7a1c2a890d8bb0aa53c3f1905fb739b791'


class Values(SimpleNamespace):
    pass


def arithmetic(expression,names):
    node=ast.parse(expression.strip(),mode='eval').body
    def value(node):
        if isinstance(node,ast.Constant) and type(node.value) in (int,float):return node.value
        if isinstance(node,ast.Name):return names[node.id]
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.USub):return -value(node.operand)
        if isinstance(node,ast.BinOp):
            left,right=value(node.left),value(node.right)
            if isinstance(node.op,ast.Add):return left+right
            if isinstance(node.op,ast.Sub):return left-right
            if isinstance(node.op,ast.Mult):return left*right
            if isinstance(node.op,ast.Div):return left/right
        raise ValueError('Unexpected C++ numeric expression: '+expression)
    return value(node)


def resolve(value,name,default=None):
    if not isinstance(value,dict):return value
    matches=[v for pattern,v in value.items() if re.fullmatch(pattern,name)]
    if not matches and default is not None:return default
    assert len(matches)==1,(name,matches)
    return matches[0]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=args.source_root.resolve()
    training=root/'gear_sonic/envs/manager_env/robots/g1.py'
    cpp=root/'gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp'
    assert hashlib.sha256(training.read_bytes()).hexdigest()==SOURCE_HASH
    assert hashlib.sha256(cpp.read_bytes()).hexdigest()==PARAMETER_HASH
    Values.InitialStateCfg=Values
    converter=Values(JointDriveCfg=Values)
    converter.JointDriveCfg.PDGainsCfg=Values
    sim=Values(UrdfFileCfg=Values,RigidBodyPropertiesCfg=Values,
               ArticulationRootPropertiesCfg=Values,UrdfConverterCfg=converter)
    namespace=dict(ImplicitActuatorCfg=Values,ArticulationCfg=Values,sim_utils=sim)
    module=ast.parse(training.read_text(encoding='utf-8'))
    module.body=[n for n in module.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
    exec(compile(module,str(training),'exec'),namespace)
    cfg=namespace['G1_CYLINDER_MODEL_12_DEX_CFG']
    original_cpp=cpp.read_text(encoding='utf-8')
    text=re.sub(r'//[^\n]*','',original_cpp)
    constants={}
    for name,expression in re.findall(r'const double (\w+)\s*=\s*([^;]+);',text):
        constants[name]=arithmetic(expression,constants)
    arrays={}
    for name,body in re.findall(r'const std::array<[^>]+>\s+(\w+)\s*=\s*\{([^}]+)\};',text,re.S):
        arrays[name]=[arithmetic(x,constants) for x in body.split(',') if x.strip()]
    # These published names are confusing: explicitly bind each array's use.
    # C++ output loop indexes MJ joint j, then reads action[isaaclab_to_mujoco[j]].
    MJ_TO_POLICY=arrays['isaaclab_to_mujoco']
    POLICY_TO_MJ=arrays['mujoco_to_isaaclab']
    assert sorted(MJ_TO_POLICY)==sorted(POLICY_TO_MJ)==list(range(29))
    assert all(MJ_TO_POLICY[POLICY_TO_MJ[i]]==i for i in range(29))
    assert namespace['G1_ISAACLAB_TO_MUJOCO_DOF']==MJ_TO_POLICY
    assert namespace['G1_MUJOCO_TO_ISAACLAB_DOF']==POLICY_TO_MJ
    # The Python variable G1_ISAACLAB_JOINTS actually contains30 body links.
    # Read29 named action-table entries and the published DOF permutation.
    action_table=re.search(r'g1_action_scale\s*=\s*\{(.*?)\};',original_cpp,re.S)[1]
    mj_names=re.findall(r'//\s*(\w+_joint)',action_table)
    assert len(mj_names)==len(set(mj_names))==29
    names=[mj_names[mj] for mj in POLICY_TO_MJ]
    rows=[]
    max_error=0.
    for i,name in enumerate(names):
        actuators=[a for a in cfg.actuators.values() if any(re.fullmatch(p,name) for p in a.joint_names_expr)]
        assert len(actuators)==1
        actuator=actuators[0];mj=POLICY_TO_MJ[i]
        kp=resolve(actuator.stiffness,name);kd=resolve(actuator.damping,name)
        effort=resolve(actuator.effort_limit_sim,name)
        scale=resolve(namespace['G1_MODEL_12_ACTION_SCALE'],name)
        default=resolve(cfg.init_state.joint_pos,name,0.)
        pairs=[(kp,arrays['kps'][mj]),(kd,arrays['kds'][mj]),
               (scale,arrays['g1_action_scale'][mj]),(default,arrays['default_angles'][mj])]
        for left,right in pairs:
            max_error=max(max_error,abs(left-right))
            assert math.isclose(left,right,rel_tol=1e-12,abs_tol=1e-12),(name,left,right)
        assert math.isclose(scale,.25*effort/kp,rel_tol=1e-12)
        rows.append(dict(policy_index=i,mujoco_index=mj,name=name,default_joint_position=default,
            stiffness=kp,damping=kd,action_scale=scale,effort_limit=effort,
            training_armature=resolve(actuator.armature,name),
            training_velocity_limit=resolve(actuator.velocity_limit_sim,name)))
    report=dict(status='all29_published_SONIC_training_and_deploy_gain_order_tables_agree',
        source_commit='1e1ec4f5f27e6c3e03048f693bd7c60cce04afda',
        source_hashes={'training_g1.py':SOURCE_HASH,'policy_parameters.hpp':PARAMETER_HASH},
        joints_in_policy_order=rows,mujoco_joint_to_policy_index=MJ_TO_POLICY,
        policy_joint_to_mujoco_index=POLICY_TO_MJ,maximum_table_absolute_difference=max_error,
        gain_comparison_precision='Evaluate declared arithmetic expressions in float64; C++ float-array serialization is not compared',
        joint_names_source='29 named C++ action-table entries remapped by the Python/C++ identical DOF permutations',
        misnamed_G1_ISAACLAB_JOINTS_contains30_body_links=True,
        source_configuration_expressions_evaluated=True,IsaacLab_initialized=False,simulation_executed=False,
        exact_checkpoint_training_revision_confirmed=False,paper_evaluation_actuator_protocol_confirmed=False,
        hip_pitch_uses7520_22=True,SuperMimic_default_gains_must_not_be_substituted=True,
        history_notes='C++ GetLatest uses oldest-to-newest order and pads insufficient ring history with zero entries. Exact benchmark reset history is not supplied.',
        sensor_port_verified=False,policy_quality_evaluated=False,paper_results_reproduced=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:report[k] for k in ('status','maximum_table_absolute_difference',
          'IsaacLab_initialized','simulation_executed','sensor_port_verified')},indent=2))


if __name__=='__main__':main()
