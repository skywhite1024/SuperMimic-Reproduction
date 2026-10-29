"""Cold-path contract and joint-range audit of the small public reference."""
import hashlib,json
import numpy as np
import torch
from runtime import ROOT,UNI
from unisupermimic.tasks.motion_tracking.g1.supermimic.kinematics import G1Kinematics
from unisupermimic.tasks.motion_tracking.g1.supermimic.robot import G1_JOINT_NAMES
from unisupermimic.rl.algos.supermimic.geometry import quat_to_mat


def main():
    file=ROOT/'assets/acro-candidates/88_08_stageii_unitree_g1.csv';raw=np.loadtxt(file,delimiter=',')
    assert raw.shape==(143,36) and np.isfinite(raw).all()
    expected=['left_hip_pitch','left_hip_roll','left_hip_yaw','left_knee','left_ankle_pitch','left_ankle_roll','right_hip_pitch','right_hip_roll','right_hip_yaw','right_knee','right_ankle_pitch','right_ankle_roll','waist_yaw','waist_roll','waist_pitch','left_shoulder_pitch','left_shoulder_roll','left_shoulder_yaw','left_elbow','left_wrist_roll','left_wrist_pitch','left_wrist_yaw','right_shoulder_pitch','right_shoulder_roll','right_shoulder_yaw','right_elbow','right_wrist_roll','right_wrist_pitch','right_wrist_yaw']
    assert [n.removesuffix('_joint') for n in G1_JOINT_NAMES]==expected
    kin=G1Kinematics(str(UNI/'src/unisupermimic/assets/robots/g1/scene_supermimic.xml'),'cpu')
    a=torch.tensor(raw,dtype=torch.float32);p=a[:,:3];q=a[:,7:];quat=a[:,[6,3,4,5]];rotation=quat_to_mat(quat)
    violations=torch.maximum(kin.q_lo-q,q-kin.q_hi).clamp_min(0)
    bodies=kin.forward(p,rotation,torch.zeros_like(p),torch.zeros_like(p),q,torch.zeros_like(q),all_bodies=True)[0]
    result={'source':str(file),'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'frames':len(raw),'fps':30,
            'schema':'root XYZ metres, Z-up; quaternion xyzw; 29 SDK joint positions radians; no header',
            'explicit_joint_name_order_matches_local':True,'quaternion_norm_max_error':float(np.abs(np.linalg.norm(raw[:,3:7],axis=1)-1).max()),
            'joint_range_peak_violation_rad':float(violations.max()),'joint_range_violating_frames':int((violations.max(1).values>1e-5).sum()),
            'per_joint_peak_violation_rad':{n:float(violations[:,i].max()) for i,n in enumerate(G1_JOINT_NAMES) if violations[:,i].max()>1e-5},
            'minimum_link_origin_height_m':float(bodies[:,:,2].min()),'reference_Rzz_min':float(rotation[:,2,2].min()),
            'source_description':'https://github.com/lez666/chunwan-flip/blob/main/README.md',
            'limits':'A kinematic contract check does not prove dynamic feasibility, collision freedom, or a successful physical flip. Raw source remains unchanged.'}
    (ROOT/'evidence/acro-reference-contract.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
