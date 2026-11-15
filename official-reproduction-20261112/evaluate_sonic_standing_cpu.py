"""Ten-second scalar-MuJoCo SONIC standing pilot, not a motion benchmark.

Builds the pinned SuperMimic/mjlab G1 scene with its published MuJoCo options.
Uses SONIC's own gains, effort bounds, default offsets and action scales. Common
G1 armatures remain unchanged. No physical robot, cloud or CUDA calls occur.
"""
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from prepare_sonic_onnx_batch import MODEL_HASHES,digest


def write(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8',newline='\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',type=Path,required=True)
    parser.add_argument('--batched',type=Path,required=True)
    parser.add_argument('--contract',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    import mujoco
    import onnxruntime as ort
    import yaml
    import supermimic.tasks
    from mjlab.scene import Scene
    from mjlab.tasks.registry import load_env_cfg
    models=args.models.resolve();output=args.output.resolve()
    assert all(digest(models/name)==value for name,value in MODEL_HASHES.items())
    contract=json.loads(args.contract.read_bytes())
    assert contract['status']=='all29_published_SONIC_training_and_deploy_gain_order_tables_agree'
    rows=contract['joints_in_policy_order'];names=[row['name'] for row in rows]
    assert len(set(names))==29
    training_cfg=models/'sonic_release/config.yaml'
    assert digest(training_cfg)=='f08187795fa16a839a28bc1c18e0555d38d9420e03733744341cdcb56ab629c7'
    action_clip=yaml.safe_load(training_cfg.read_bytes())['manager_env']['config']['action_clip_value']
    assert action_clip==20
    cfg=load_env_cfg('SuperMimic-GenCoord-G1',play=True);cfg.scene.num_envs=1
    scene=Scene(cfg.scene,'cpu');model=scene.compile();cfg.sim.mujoco.apply(model)
    assert (model.nq,model.nv,model.nu)==(36,35,29) and model.opt.timestep==.005
    model_names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,i) for i in range(model.njnt)]
    joint_ids=[model_names.index('robot/'+name) for name in names]
    assert len(set(joint_ids))==29
    q_indices=np.asarray([model.jnt_qposadr[i] for i in joint_ids])
    v_indices=np.asarray([model.jnt_dofadr[i] for i in joint_ids])
    actuator_ids=np.asarray([np.flatnonzero(model.actuator_trnid[:,0]==i).item() for i in joint_ids])
    assert len(set(actuator_ids.tolist()))==29
    assert np.all(model.actuator_gaintype==mujoco.mjtGain.mjGAIN_FIXED)
    assert np.all(model.actuator_biastype==mujoco.mjtBias.mjBIAS_AFFINE)
    assert np.all(model.actuator_dyntype==mujoco.mjtDyn.mjDYN_NONE)
    default=np.asarray([row['default_joint_position'] for row in rows])
    kp=np.asarray([row['stiffness'] for row in rows])
    kd=np.asarray([row['damping'] for row in rows])
    scales=np.asarray([row['action_scale'] for row in rows])
    effort=np.asarray([row['effort_limit'] for row in rows])
    armatures=model.dof_armature[v_indices].copy()
    model.actuator_gainprm[actuator_ids,0]=kp
    model.actuator_biasprm[actuator_ids,0]=0
    model.actuator_biasprm[actuator_ids,1]=-kp
    model.actuator_biasprm[actuator_ids,2]=-kd
    model.actuator_forcelimited[actuator_ids]=1
    model.actuator_forcerange[actuator_ids]=np.stack((-effort,effort),axis=-1)
    model.geom_friction[:,0]=1.
    if model.npair:model.pair_friction[:,:2]=1.
    assert np.array_equal(model.dof_armature[v_indices],armatures)
    data=mujoco.MjData(model)
    free=np.flatnonzero(model.jnt_type==mujoco.mjtJoint.mjJNT_FREE).item()
    root_address=model.jnt_qposadr[free];body_id=int(model.jnt_bodyid[free])
    data.qpos[root_address:root_address+7]=[0,0,.76,1,0,0,0]
    data.qpos[q_indices]=default;data.qvel[:]=0;data.ctrl[actuator_ids]=default
    mujoco.mj_forward(model,data)
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
    session=lambda path:ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
    encoder=session(models/'model_encoder.onnx');decoder=session(models/'model_decoder.onnx')
    batch_validation=json.loads((args.batched/'validation.json').read_bytes())
    for record in batch_validation['derived_files']:
        assert digest(args.batched/record['file'])==record['sha256']
    batch_encoder=session(args.batched/'g1-encoder-batched.onnx')
    batch_decoder=session(args.batched/'decoder-batched.onnx')
    history=None;last_action=np.zeros(29,dtype=np.float32)
    state_qpos=[data.qpos.copy()];state_qvel=[data.qvel.copy()]
    actions=[];encoder_inputs=[];decoder_inputs=[];heights=[];tilts=[]
    max_token_difference=0.;max_action_difference=0.;clipped=0;error=None
    output.mkdir(parents=True,exist_ok=False)
    protocol=dict(task='static SONIC default knees-bent standing pose, desired root orientation identity',
        source_commit=contract['source_commit'],HF_revision=batch_validation['HF_revision'],
        backend='Scalar MuJoCo on local CPU; not MuJoCo Warp or IsaacLab',mujoco_version=mujoco.__version__,
        control_seconds=.02,physics_substeps=4,target_steps=500,target_seconds=10,
        mujoco_options=dataclasses.asdict(cfg.sim.mujoco),seed=42,
        control_contract_sha256=digest(args.contract),policy_order=names,
        common_scene_armatures_retained=armatures.tolist(),
        SONIC_training_armatures=[row['training_armature'] for row in rows],
        effort_bounds=effort.tolist(),stiffness=kp.tolist(),damping=kd.tolist(),
        action_scales=scales.tolist(),action_clip=action_clip,initial_root_pose=[0,0,.76,1,0,0,0],
        initial_joint_positions=default.tolist(),friction=1.,domain_randomization=False,
        history='Ten oldest-to-newest frames, grouped omega/q-minus-SONIC-default/qd/last-raw-action/gravity. Initial standing sensor repeated; standing defaults make its first sensor equal zero-entry proprioception.',
        mode='G1 scalar0 with inactive modality inputs zero; q/default reference and qd0 held; relative full-base orientation as row-major first two rotation columns',
        stopping_rule='Root height differs from0.76 by>0.25 or1+body-gravity-z>0.8, or nonfinite state; retain failed prefix',
        exact_training_history_reset_confirmed=False,model_files=MODEL_HASHES,
        physical_robot_deployment=False,paper_results_reproduced=False)
    write(output/'protocol.json',protocol)
    started=time.monotonic()
    try:
        for step in range(500):
            rotation=np.empty(9);mujoco.mju_quat2Mat(rotation,data.xquat[body_id]);rotation=rotation.reshape(3,3)
            gravity=rotation.T@np.array([0.,0.,-1.])
            velocity=np.empty(6);mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,body_id,velocity,1)
            sensor=np.concatenate((velocity[:3],data.qpos[q_indices]-default,
                                   data.qvel[v_indices],last_action,gravity)).astype(np.float32)
            assert sensor.shape==(93,) and np.isfinite(sensor).all()
            history=(np.tile(sensor,(10,1)) if history is None else np.concatenate((history[1:],sensor[None])))
            features=np.zeros((1,1762),dtype=np.float32)
            features[:,4:294]=np.tile(default,10)
            features[:,601:661]=np.tile(rotation.T[:,:2].reshape(6),10)
            tokens=encoder.run(None,{'obs_dict':features})[0]
            policy=np.concatenate((tokens.reshape(-1),history[:,:3].reshape(-1),
                history[:,3:32].reshape(-1),history[:,32:61].reshape(-1),
                history[:,61:90].reshape(-1),history[:,90:].reshape(-1))).astype(np.float32)[None]
            assert policy.shape==(1,994)
            action=decoder.run(None,{'obs_dict':policy})[0].reshape(29)
            derived_tokens=batch_encoder.run(None,{'obs_dict':features})[0].reshape(1,64)
            derived_policy=policy.copy();derived_policy[:,:64]=derived_tokens
            derived_action=batch_decoder.run(None,{'obs_dict':derived_policy})[0].reshape(29)
            max_token_difference=max(max_token_difference,float(np.max(np.abs(tokens-derived_tokens))))
            max_action_difference=max(max_action_difference,float(np.max(np.abs(action-derived_action))))
            assert np.allclose(tokens,derived_tokens,atol=1e-6,rtol=1e-6)
            assert np.allclose(action,derived_action,atol=1e-4,rtol=1e-4)
            assert np.isfinite(action).all()
            applied=np.clip(action,-action_clip,action_clip)
            clipped+=int(np.count_nonzero(applied!=action))
            data.ctrl[actuator_ids]=default+applied*scales
            for _ in range(4):mujoco.mj_step(model,data)
            mujoco.mj_forward(model,data)
            assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
            last_action=action.copy()
            state_qpos.append(data.qpos.copy());state_qvel.append(data.qvel.copy())
            actions.append(action.copy());encoder_inputs.append(features[0]);decoder_inputs.append(policy[0])
            rotation=np.empty(9);mujoco.mju_quat2Mat(rotation,data.xquat[body_id]);rotation=rotation.reshape(3,3)
            height=float(data.qpos[root_address+2]);tilt=float(1.-rotation[2,2])
            heights.append(height);tilts.append(tilt)
            if abs(height-.76)>.25 or tilt>.8:
                error='Standing pose left native-like height/tilt bounds at step'+str(step+1);break
    except Exception as exception:error=type(exception).__name__+': '+str(exception)
    arrays=dict(qpos=np.asarray(state_qpos),qvel=np.asarray(state_qvel),
        actions=np.asarray(actions),encoder_inputs=np.asarray(encoder_inputs),
        decoder_inputs=np.asarray(decoder_inputs),root_height=np.asarray(heights),tilt=np.asarray(tilts))
    trace=output/'trace.npz';np.savez_compressed(trace,**arrays)
    with np.load(trace,allow_pickle=False) as saved:
        assert all(np.array_equal(saved[k],v) for k,v in arrays.items())
    result=dict(status='ten_second_scalar_MuJoCo_SONIC_standing_completed' if len(actions)==500 and not error else 'scalar_standing_pilot_failed_or_incomplete',
        control_steps=len(actions),simulated_seconds=len(actions)*.02,error=error,
        minimum_root_height=min(heights,default=None),maximum_tilt=max(tilts,default=None),
        clipped_action_components=clipped,wall_loop_seconds=time.monotonic()-started,
        real_released_model_inference=True,all_observed_G1_batch1_latents_and_actions_compared_to_original=True,
        max_token_difference=max_token_difference,max_action_difference=max_action_difference,
        protocol_sha256=digest(output/'protocol.json'),
        trace=dict(file=trace.name,bytes=trace.stat().st_size,sha256=digest(trace),readback_verified=True),
        original_models_unchanged=all(digest(models/name)==value for name,value in MODEL_HASHES.items()),
        full_motion_tracking_evaluated=False,Warp_or_IsaacLab_equivalence_verified=False,
        GPU_execution_verified=False,paper_results_reproduced=False)
    write(output/'result.json',result);print(json.dumps(result,indent=2))
    if error:raise RuntimeError(error)


if __name__=='__main__':main()
