"""Kinematic diagnostics of every fixed bridge; no feasibility claim from pose alone."""
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from runtime import ROOT, make_env, references


def main():
    target=ROOT/'evidence/continuous-reference-bridge-diagnostic.json'
    assert not target.exists()
    env=make_env(references('new')[:1],observation_dim=99)
    versions=[]
    try:
        model=env.backend.model
        scratch=mujoco.MjData(model)
        floors=set(np.flatnonzero((model.geom_type==mujoco.mjtGeom.mjGEOM_PLANE)&(model.geom_bodyid==0)).tolist())
        limited=np.flatnonzero(model.jnt_limited)
        addresses=model.jnt_qposadr[limited]
        limits=model.jnt_range[limited]
        for version,mode in [('continuous-frozen-v1','stage3-frozen-continuous-v1'),('continuous-corrected-v1','stage3-corrected-continuous-v1')]:
            index_path=ROOT/version/'index.json';index=json.loads(index_path.read_text())
            summary=json.loads((ROOT/'runs'/mode/'summary.json').read_text())
            with np.load(ROOT/'runs'/mode/'trajectory.npz') as archive:states=archive['state']
            sequences=[]
            for slot,sequence in enumerate(index['sequences']):
                file=ROOT/version/f'seed-{sequence["seed"]}'/'sequence.csv'
                assert hashlib.sha256(file.read_bytes()).hexdigest()==sequence['reference_sha256']
                csv=np.loadtxt(file,delimiter=',')[:6001]
                qpos=csv.copy();qpos[:,3:7]=Rotation.from_quat(csv[:,3:7]).as_quat(scalar_first=True)
                com=[];floor_contact=[];minimum=[]
                for pose in qpos:
                    scratch.qpos[:]=pose;mujoco.mj_fwdPosition(model,scratch)
                    com.append(np.average(scratch.xipos,weights=model.body_mass,axis=0))
                    bodies=set();distance=0.
                    for item in scratch.contact:
                        first,second=int(item.geom1),int(item.geom2)
                        if (first in floors)==(second in floors) or item.dist>.001:continue
                        geom=second if first in floors else first
                        bodies.add(model.body(int(model.geom_bodyid[geom])).name);distance=min(distance,float(item.dist))
                    floor_contact.append(bool(bodies));minimum.append(distance)
                com=np.asarray(com);contact=np.asarray(floor_contact);minimum=np.asarray(minimum)
                acceleration=np.full_like(com,np.nan);acceleration[1:-1]=np.diff(com,n=2,axis=0)/.02**2
                unsupported=np.zeros(len(com),bool);unsupported[1:-1]=~(contact[:-2]|contact[1:-1]|contact[2:])
                residual=np.linalg.norm(acceleration-model.opt.gravity,axis=1)
                joint_velocity=np.gradient(qpos[:,7:],.02,axis=0)
                violation=np.maximum(np.maximum(limits[:,0]-qpos[:,addresses],qpos[:,addresses]-limits[:,1]),0)
                bridges=[]
                for segment in sequence['segments']:
                    start,stop=segment['transition_start_frame'],segment['source_start_in_sequence_frame']
                    if stop<=start:continue
                    mask=unsupported[start:stop]
                    values=residual[start:stop][mask]
                    bridges.append({'destination':segment['source'],'transition_start_s':start*.02,'source_start_s':stop*.02,
                        'bridge_frames':stop-start,'sampled_no_enabled_floor_contact_triplets':int(mask.sum()),
                        'sampled_unsupported_COM_acceleration_residual_to_gravity_peak_m_s2':float(values.max()) if len(values) else None,
                        'joint_speed_peak_rad_s':float(np.abs(joint_velocity[start:stop]).max()),
                        'reference_joint_limit_excess_peak_rad':float(violation[start:stop].max()),
                        'reference_geometry_min_contact_distance_m':float(minimum[start:stop].min())})
                trial=summary['trials'][slot];failure=trial['first_hard_failure_s'];frame=round(failure/.02) if failure is not None else None
                event=None
                if frame is not None:
                    actual=states[frame,slot];actual_rzz=1-2*(actual[4]**2+actual[5]**2)
                    ref_rzz=Rotation.from_quat(csv[frame,3:7]).as_matrix()[2,2]
                    segments=[segment for segment in sequence['segments'] if segment['transition_start_frame']<=frame<=segment['end_frame']]
                    event={'time_s':failure,'reference_pelvis_m':float(csv[frame,2]),'actual_pelvis_m':float(actual[2]),
                           'reference_Rzz':float(ref_rzz),'actual_Rzz':float(actual_rzz),
                           'absolute_root_height_error_m':float(abs(csv[frame,2]-actual[2])),
                           'absolute_root_tilt_Rzz_error':float(abs(ref_rzz-actual_rzz)),
                           'segment':segments[0] if segments else None,
                           'in_interpolation_bridge':any(segment['transition_start_frame']<=frame<segment['source_start_in_sequence_frame'] for segment in segments),
                           'sampled_reference_floor_contact':bool(contact[frame]),
                           'sampled_reference_COM_acceleration_residual_to_gravity_m_s2':float(residual[frame])}
                sequences.append({'seed':sequence['seed'],'reference_sha256':sequence['reference_sha256'],'all_bridges':bridges,'first_physical_failure':event})
                np.savez_compressed(ROOT/'evidence'/f'{version}-seed-{sequence["seed"]}-reference-kinematics.npz',
                                    com=com,reference_floor_contact=contact,COM_acceleration=acceleration,
                                    sampled_unsupported_triplets=unsupported,acceleration_residual_to_gravity=residual,
                                    reference_joint_limit_excess=violation)
            versions.append({'reference_version':version,'index_sha256':hashlib.sha256(index_path.read_bytes()).hexdigest(),'sequences':sequences})
        result={'status':'all_fixed_bridges_audited','reference_and_policy_weights_modified':False,
                'thresholds_or_denominators_modified':False,'contact_force_measured':False,
                'interpretation':'Kinematic sampled-pose diagnostic only. COM acceleration in sampled contact-free triplets can flag an inconsistent flight target, but contacts between samples and allowed tracking error prevent claiming a proof of dynamic impossibility. Joint-speed values are descriptive; no speed-limit acceptance threshold is introduced.',
                'gravity_m_s2':model.opt.gravity.tolist(),'versions':versions}
        target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
        print(json.dumps({'status':result['status'],'first_failures':[{'version':v['reference_version'],'events':[s['first_physical_failure'] for s in v['sequences']]} for v in versions]}),flush=True)
    finally:env.close();env.backend.close()


if __name__=='__main__':main()
