"""Measure signed ground clearance on scratch kinematics, without altering references."""
import json
from pathlib import Path
import numpy as np
import mujoco
from runtime import ROOT,make_env,references
from unisupermimic.tasks.motion_tracking.g1.supermimic.robot import G1_JOINT_NAMES


def main():
    manifest=json.loads((ROOT/'library-manifest.json').read_text());env=make_env(references('new')[:1],observation_dim=99)
    rows=[]
    try:
        model=env.backend.model;data=mujoco.MjData(model)
        floor=[int(i) for i in np.flatnonzero((model.geom_type==mujoco.mjtGeom.mjGEOM_PLANE)&(model.geom_bodyid==0))];assert len(floor)==1
        collision=[int(i) for i in range(model.ngeom) if model.geom_bodyid[i]!=0 and (model.geom_contype[i] or model.geom_conaffinity[i])]
        assert collision
        for row in manifest['files']:
            raw=np.loadtxt(ROOT/row['path'],delimiter=',');minimum=[];com=[]
            for pose in raw:
                data.qpos[:3]=pose[:3];data.qpos[3:7]=pose[[6,3,4,5]];data.qpos[7:]=pose[7:]
                mujoco.mj_fwdPosition(model,data)
                minimum.append(min(float(mujoco.mj_geomDistance(model,data,floor[0],geom,10,None)) for geom in collision))
                com.append(data.subtree_com[1].copy())
            distance=np.array(minimum);com=np.array(com);air=distance>.02
            runs=[];begin=None
            for i,flag in enumerate(np.r_[air,False]):
                if flag and begin is None:begin=i
                if not flag and begin is not None:runs.append({'start_s':begin/30,'end_s':(i-1)/30,'duration_s':(i-begin)/30});begin=None
            result={'clip':Path(row['path']).stem,'frames':len(raw),'minimum_any_body_clearance_m':float(distance.min()),
                    'maximum_clearance_m':float(distance.max()),'first_half_second_clearance_m':distance[:15].tolist(),
                    'all_frames_collision_geometry_above_ground_2cm':bool(air.all()),'longest_no_near_ground_interval_s':max((x['duration_s'] for x in runs),default=0),
                    'no_near_ground_intervals':runs,'whole_body_COM_z_min_max_m':[float(com[:,2].min()),float(com[:,2].max())],
                    'raw_reference_unchanged':True,'diagnostic_only_not_dynamic_feasibility_proof':True}
            rows.append(result);print('GROUND_CLEARANCE '+json.dumps({key:result[key] for key in ['clip','minimum_any_body_clearance_m','longest_no_near_ground_interval_s']}),flush=True)
        out=ROOT/'evidence/raw-ground-clearance.json';assert not out.exists()
        out.write_text(json.dumps({'method':'MuJoCo mj_geomDistance between ground plane and all runtime collision geometries, each raw 30 Hz pose on independent scratch MjData; no simulation state or reference modified','denominator':15,'clips':rows},indent=2)+'\n')
    finally:env.close();env.backend.close()


if __name__=='__main__':main()
