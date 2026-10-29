"""Reconstruct ground contact geometry from retained actual states, without replaying dynamics."""
import json
import hashlib
from collections import defaultdict
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from runtime import ROOT,make_env
from unisupermimic.tasks.motion_tracking.g1.supermimic.robot import G1_JOINT_NAMES


def main():
    run=ROOT/'runs/acro-h100-2000-retest';file=run/'trajectory.npz'
    with np.load(file) as x:state=x['state'].copy();ref=x['reference_qpos'].copy()
    actual=np.concatenate((state[:,:7],state[:,13:42]),1)
    reference=np.concatenate((ref[:,:3],Rotation.from_matrix(ref[:,3:12].reshape(-1,3,3)).as_quat(scalar_first=True),ref[:,12:]),1)
    env=make_env([ROOT/'assets/acro-candidates/88_08_stageii_unitree_g1.csv'],fps=30,continuous=True)
    try:
        model=env.backend.model;assert model.nq==36
        ids=env.backend.get_joint_state_qpos_indices(G1_JOINT_NAMES)
        assert np.array_equal(ids,np.arange(7,36))
        floor=set(np.flatnonzero((model.geom_type==mujoco.mjtGeom.mjGEOM_PLANE)&(model.geom_bodyid==0)).tolist())
        assert floor
        output={};rows=[]
        for label,poses in [('actual',actual),('reference',reference)]:
            data=mujoco.MjData(model);contacts=defaultdict(list);minimum={}
            for i,pose in enumerate(poses):
                data.qpos[:]=pose;mujoco.mj_fwdPosition(model,data)
                touched={}
                for c in data.contact:
                    g1,g2=int(c.geom1),int(c.geom2)
                    if (g1 in floor)==(g2 in floor):continue
                    other=g2 if g1 in floor else g1
                    name=model.body(int(model.geom_bodyid[other])).name
                    minimum[name]=min(minimum.get(name,float('inf')),float(c.dist))
                    if c.dist<=.001:touched[name]=min(touched.get(name,float('inf')),float(c.dist))
                for name,dist in touched.items():contacts[name].append(i);rows.append({'kind':label,'time_s':i*.02,'body':name,'signed_contact_distance_m':dist})
            output[label]={name:{'sampled_frames':len(indices),'first_s':indices[0]*.02,'last_s':indices[-1]*.02,'minimum_signed_distance_m':minimum[name]} for name,indices in sorted(contacts.items())}
        rzz=1-2*(state[:,4]**2+state[:,5]**2)
        result={'method':'MuJoCo mj_fwdPosition on a separate MjData using the exact runtime collision model; controlled simulator state was never edited',
                'interpretation':'Geometric ground-contact reconstruction at retained 20ms frames, not recorded contact forces or a new physical rollout',
                'trajectory_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'touch_threshold_m':.001,'ground_contacts':output,
                'actual_hand_ground_contact_observed':any('wrist' in name for name in output['actual']),
                'reference_hand_ground_contact_observed':any('wrist' in name for name in output['reference']),
                'actual_head_or_torso_ground_contact_frames':{name:value['sampled_frames'] for name,value in output['actual'].items() if any(part in name for part in ['head','torso','pelvis','logo'])},
                'end_state':{'pelvis_height_m':float(state[-1,2]),'Rzz':float(rzz[-1]),'last_half_second_min_pelvis_m':float(state[-25:,2].min()),'last_half_second_min_Rzz':float(rzz[-25:].min()),
                             'root_speed_m_s':float(np.linalg.norm(state[-1,7:10])),'root_angular_speed_rad_s':float(np.linalg.norm(state[-1,10:13])),'joint_velocity_rms_rad_s':float(np.sqrt(np.mean(state[-1,42:]**2)))},
                'post_initialization_resets':0,'scope':'single 88_08 hands-supported backward flip; no airborne-backflip or complete-video coverage claim'}
        (ROOT/'evidence/acro-h100-contact-audit.json').write_text(json.dumps(result,indent=2)+'\n')
        (ROOT/'evidence/acro-h100-contact-samples.json').write_text(json.dumps(rows,indent=2)+'\n')
        print('ACRO_CONTACT_GEOMETRY_AUDIT '+json.dumps(result))
    finally:env.close();env.backend.close()


if __name__=='__main__':main()
