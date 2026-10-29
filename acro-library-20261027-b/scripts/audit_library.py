"""Complete schema, motion and collision-geometry audit of the frozen 15-clip denominator."""
import hashlib,json,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent/'interactive-g1-20261027-a/scripts'))
from runtime import make_env
from unisupermimic.tasks.motion_tracking.g1.supermimic.robot import G1_JOINT_NAMES


def main():
    manifest=json.loads((ROOT/'library-manifest.json').read_text())
    descriptions={x['name']:x for x in json.loads((ROOT/'sources/motion-descriptions.json').read_text())}
    expected=['left_hip_pitch','left_hip_roll','left_hip_yaw','left_knee','left_ankle_pitch','left_ankle_roll','right_hip_pitch','right_hip_roll','right_hip_yaw','right_knee','right_ankle_pitch','right_ankle_roll','waist_yaw','waist_roll','waist_pitch','left_shoulder_pitch','left_shoulder_roll','left_shoulder_yaw','left_elbow','left_wrist_roll','left_wrist_pitch','left_wrist_yaw','right_shoulder_pitch','right_shoulder_roll','right_shoulder_yaw','right_elbow','right_wrist_roll','right_wrist_pitch','right_wrist_yaw']
    assert [n.removesuffix('_joint') for n in G1_JOINT_NAMES]==expected
    env=make_env([ROOT/manifest['files'][0]['path']],fps=30,continuous=True)
    try:
        model=env.backend.model;assert model.nq==36
        indices=env.backend.get_joint_state_qpos_indices(G1_JOINT_NAMES);assert np.array_equal(indices,np.arange(7,36))
        limits=np.array([model.jnt_range[model.joint(name).id] for name in G1_JOINT_NAMES])
        floor=set(np.flatnonzero((model.geom_type==mujoco.mjtGeom.mjGEOM_PLANE)&(model.geom_bodyid==0)).tolist());assert floor
        rows=[]
        for source in manifest['files']:
            path=ROOT/source['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==source['sha256']
            raw=np.loadtxt(path,delimiter=',');assert raw.ndim==2 and raw.shape[1]==36 and len(raw)>2 and np.isfinite(raw).all()
            norms=np.linalg.norm(raw[:,3:7],axis=1);assert np.max(abs(norms-1))<1e-4
            q=raw[:,7:];violations=np.maximum(np.maximum(limits[:,0]-q,q-limits[:,1]),0)
            rotations=Rotation.from_quat(raw[:,3:7]);mat=rotations.as_matrix();rzz=mat[:,2,2]
            contacts=defaultdict(int);worst=defaultdict(lambda:0.);data=mujoco.MjData(model)
            for pose in raw:
                data.qpos[:3]=pose[:3];data.qpos[3:7]=pose[[6,3,4,5]];data.qpos[7:]=pose[7:]
                mujoco.mj_fwdPosition(model,data);seen=set()
                for c in data.contact:
                    g1,g2=int(c.geom1),int(c.geom2)
                    if (g1 in floor)==(g2 in floor):continue
                    other=g2 if g1 in floor else g1;name=model.body(int(model.geom_bodyid[other])).name
                    if c.dist<=.001:seen.add(name);worst[name]=min(worst[name],float(c.dist))
                for name in seen:contacts[name]+=1
            semantics=descriptions[path.stem]
            peak=float(violations.max());flags=[]
            if peak>1e-5:flags.append('joint_range_overshoot')
            if min(worst.values(),default=0)<-.03:flags.append('ground_fit_penetration_over_3cm')
            if 'object' in semantics['description'].lower():flags.append('body_pose_only_no_object_task')
            velocity=np.gradient(raw[:,:3],1/30,axis=0);qd=np.gradient(q,1/30,axis=0)
            rows.append({**source,'semantics':semantics,'frames':len(raw),'csv_fps':30,'duration_s':(len(raw)-1)/30,
                         'numeric_contract_passed':True,'joint_order_verified':True,'quaternion_norm_max_error':float(abs(norms-1).max()),
                         'joint_limit_peak_rad':peak,'joint_limit_violating_frames':int((violations.max(1)>1e-5).sum()),
                         'joint_violations':{name:float(violations[:,i].max()) for i,name in enumerate(G1_JOINT_NAMES) if violations[:,i].max()>1e-5},
                         'root_z_min_max_m':[float(raw[:,2].min()),float(raw[:,2].max())],'Rzz_min':float(rzz.min()),'inverted_frames_Rzz_below_minus_half':int((rzz<-.5).sum()),
                         'root_peak_speed_m_s':float(np.linalg.norm(velocity,axis=1).max()),'joint_peak_speed_rad_s':float(abs(qd).max()),
                         'root_orientation_travel_deg':float(np.rad2deg((rotations[1:]*rotations[:-1].inv()).magnitude()).sum()),
                         'ground_contact_geometry_frames':dict(contacts),'ground_min_signed_distance_by_body_m':dict(worst),
                         'flags':flags,'dynamic_feasibility_verified':False,'training_eligibility':'requires_review' if peak>.02 else 'numeric_compatible_pending_physical_trial'})
            print('LIBRARY_AUDIT_CLIP '+json.dumps({k:rows[-1][k] for k in ['path','duration_s','Rzz_min','joint_limit_peak_rad','flags']}),flush=True)
        result={'commit':manifest['commit'],'denominator':len(rows),'schema':'XYZ metres Z-up, xyzw unit quaternion, 29 SDK joint radians, CSV 30fps; original CMU fps retained separately',
                'method':'All raw frames checked, including ground geometry with runtime collision model. Kinematic audit does not establish dynamic tracking or recorded contact forces.',
                'readme_actual_count_discrepancy':manifest['readme_and_actual_count_discrepancy'],'clips':rows}
        (ROOT/'library-audit.json').write_text(json.dumps(result,indent=2)+'\n')
        print('LIBRARY_AUDIT_COMPLETE '+json.dumps({'denominator':len(rows),'numeric_passed':sum(x['numeric_contract_passed'] for x in rows),'joint_range_review_required':sum(x['training_eligibility']=='requires_review' for x in rows)}))
    finally:env.close();env.backend.close()


if __name__=='__main__':main()
