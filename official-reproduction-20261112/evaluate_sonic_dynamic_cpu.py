"""Finite dynamic SONIC reference pilot in scalar MuJoCo; not a paper benchmark.

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
    parser.add_argument('--motions',type=Path,required=True)
    parser.add_argument('--motion-name',required=True)
    parser.add_argument('--steps',type=int,default=500)
    parser.add_argument('--corpus-contract',type=Path,required=True)
    args=parser.parse_args()
    assert 0<args.steps<=15000
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
    corpus=json.loads(args.corpus_contract.read_bytes())['motion_subset']
    for name,value in corpus['source_motion_files'].items():
        assert digest(args.motions/name)==value,(name,'canonical motion hash mismatch')
    motion_names=json.loads((args.motions/'motion_names.json').read_bytes())
    source_names=json.loads((args.motions/'joint_names.json').read_bytes())
    reorder=np.asarray([source_names.index(name) for name in names])
    loaded={name:np.load(args.motions/(name+'.npy'),mmap_mode='r') for name in
        ['joint_pos','joint_vel','root_pos_w','root_quat_w','root_lin_vel_w','root_ang_vel_w','motion_ends']}
    ends=loaded['motion_ends'];index=motion_names.index(args.motion_name)
    start=0 if index==0 else int(ends[index-1]);end=int(ends[index]);length=end-start
    assert args.steps<=length-1
    ref={name:np.asarray(value[start:end],dtype=np.float64) for name,value in loaded.items() if name!='motion_ends'}
    ref['joint_pos']=ref['joint_pos'][:,reorder];ref['joint_vel']=ref['joint_vel'][:,reorder]
    quat_norms=np.linalg.norm(ref['root_quat_w'],axis=1)
    assert np.allclose(quat_norms,1,atol=1e-5)
    ref['root_quat_w']/=quat_norms[:,None]
    soft_factor=cfg.scene.entities['robot'].articulation.soft_joint_pos_limit_factor
    limits=model.jnt_range[joint_ids]
    mid=limits.mean(axis=1);half=.5*(limits[:,1]-limits[:,0])*soft_factor
    initial_q=np.clip(ref['joint_pos'][0],mid-half,mid+half)
    data.qpos[root_address:root_address+3]=ref['root_pos_w'][0]
    data.qpos[root_address+3:root_address+7]=ref['root_quat_w'][0]
    data.qpos[q_indices]=initial_q
    root_v=model.jnt_dofadr[free]
    data.qvel[root_v:root_v+3]=ref['root_lin_vel_w'][0]
    initial_rotation=np.empty(9);mujoco.mju_quat2Mat(initial_rotation,ref['root_quat_w'][0])
    data.qvel[root_v+3:root_v+6]=initial_rotation.reshape(3,3).T@ref['root_ang_vel_w'][0]
    data.qvel[v_indices]=ref['joint_vel'][0]
    data.ctrl[actuator_ids]=initial_q
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
    root_errors=[];joint_errors=[];reference_frames=[]
    max_token_difference=0.;max_action_difference=0.;clipped=0;error=None
    output.mkdir(parents=True,exist_ok=False)
    protocol=dict(task='Dynamic canonical LAFAN reference prefix '+args.motion_name,
        source_commit=contract['source_commit'],HF_revision=batch_validation['HF_revision'],
        backend='Scalar MuJoCo on local CPU; not MuJoCo Warp or IsaacLab',gyro_frame='Regular pelvis body frame via mjOBJ_XBODY; not inertial principal axes',mujoco_version=mujoco.__version__,
        control_seconds=.02,physics_substeps=4,target_steps=args.steps,target_seconds=args.steps*.02,
        mujoco_options=dataclasses.asdict(cfg.sim.mujoco),seed=42,
        control_contract_sha256=digest(args.contract),policy_order=names,
        common_scene_armatures_retained=armatures.tolist(),
        SONIC_training_armatures=[row['training_armature'] for row in rows],
        effort_bounds=effort.tolist(),stiffness=kp.tolist(),damping=kd.tolist(),
        action_scales=scales.tolist(),action_clip=action_clip,initial_root_pose=data.qpos[root_address:root_address+7].tolist(),
        initial_joint_positions=initial_q.tolist(),initial_joint_velocities=ref['joint_vel'][0].tolist(),friction=1.,domain_randomization=False,
        motion_source_files=corpus['source_motion_files'],motion_index=index,reference_frames=length,
        source_frames_per_second=50,future_frame_offsets=[5*i for i in range(10)],future_clamp='Last frame of this motion only',
        root_quaternion_max_norm_correction=float(np.max(np.abs(quat_norms-1))),
        native_soft_limit_factor=soft_factor,initial_joint_clip_max=float(np.max(np.abs(initial_q-ref['joint_pos'][0]))),
        history='Ten oldest-to-newest frames, grouped omega/q-minus-SONIC-default/qd/last-raw-action/gravity. Exact first reference sensor repeated, matching corrected-v2 common evaluator; training history reset is not confirmed.',
        mode='G1 scalar0; inactive modalities zero; dynamic q and qd future10 at0.1s spacing; current-base inverse times future-reference rotation, first2 columns row-major. Initial reference/current heading coincide.',
        stopping_rule='Poststep root height differs from corresponding reference by>0.25 or1-Rzz>0.8, or nonfinite state; finite prefix pilot, not complete native tracking termination/reward evaluator',
        exact_training_history_reset_confirmed=False,model_files=MODEL_HASHES,
        physical_robot_deployment=False,paper_results_reproduced=False)
    write(output/'protocol.json',protocol)
    started=time.monotonic()
    try:
        for step in range(args.steps):
            rotation=np.empty(9);mujoco.mju_quat2Mat(rotation,data.xquat[body_id]);rotation=rotation.reshape(3,3)
            gravity=rotation.T@np.array([0.,0.,-1.])
            velocity=np.empty(6);mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_XBODY,body_id,velocity,1)
            sensor=np.concatenate((velocity[:3],data.qpos[q_indices]-default,
                                   data.qvel[v_indices],last_action,gravity)).astype(np.float32)
            assert sensor.shape==(93,) and np.isfinite(sensor).all()
            history=(np.tile(sensor,(10,1)) if history is None else np.concatenate((history[1:],sensor[None])))
            features=np.zeros((1,1762),dtype=np.float32)
            future=np.minimum(step+np.arange(10)*5,length-1)
            features[:,4:294]=ref['joint_pos'][future].reshape(1,290)
            features[:,294:584]=ref['joint_vel'][future].reshape(1,290)
            ref_rotations=[]
            for frame in future:
                ref_rotation=np.empty(9);mujoco.mju_quat2Mat(ref_rotation,ref['root_quat_w'][frame])
                ref_rotations.append((rotation.T@ref_rotation.reshape(3,3))[:,:2].reshape(6))
            features[:,601:661]=np.asarray(ref_rotations).reshape(1,60)
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
            next_frame=min(step+1,length-1)
            root_errors.append(float(np.linalg.norm(data.qpos[root_address:root_address+3]-ref['root_pos_w'][next_frame])))
            joint_errors.append(float(np.sqrt(np.mean((data.qpos[q_indices]-ref['joint_pos'][next_frame])**2))))
            reference_frames.append(next_frame)
            if (step+1)%250==0:print('SONIC_DYNAMIC_PILOT_PROGRESS',args.motion_name,step+1,flush=True)
            if abs(height-ref['root_pos_w'][next_frame,2])>.25 or tilt>.8:
                error='Dynamic reference left height/tilt pilot bounds at step'+str(step+1);break
    except Exception as exception:error=type(exception).__name__+': '+str(exception)
    arrays=dict(qpos=np.asarray(state_qpos),qvel=np.asarray(state_qvel),
        actions=np.asarray(actions),encoder_inputs=np.asarray(encoder_inputs),
        decoder_inputs=np.asarray(decoder_inputs),root_height=np.asarray(heights),tilt=np.asarray(tilts),root_distance=np.asarray(root_errors),
        joint_rms=np.asarray(joint_errors),reference_frame=np.asarray(reference_frames),
        reference_root_pos=ref['root_pos_w'][:len(actions)+1],reference_joint_pos=ref['joint_pos'][:len(actions)+1])
    trace=output/'trace.npz';np.savez_compressed(trace,**arrays)
    with np.load(trace,allow_pickle=False) as saved:
        assert all(np.array_equal(saved[k],v) for k,v in arrays.items())
    result=dict(status='finite_scalar_MuJoCo_SONIC_dynamic_prefix_completed' if len(actions)==args.steps and not error else 'scalar_dynamic_pilot_failed_or_incomplete',
        motion_name=args.motion_name,control_steps=len(actions),simulated_seconds=len(actions)*.02,error=error,
        mean_root_distance=float(np.mean(root_errors)) if root_errors else None,
        mean_joint_rms=float(np.mean(joint_errors)) if joint_errors else None,
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
