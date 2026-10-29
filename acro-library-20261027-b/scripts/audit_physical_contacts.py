"""All-new-clip geometric contact review on retained physical states, not force inference."""
import argparse,hashlib,json
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import mujoco
from runtime import ROOT,make_env,references


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();run=ROOT/'runs'/a.run
    summary=json.loads((run/'summary.json').read_text());out=run/'contact-review.json';assert not out.exists()
    with np.load(run/'trajectory.npz') as archive:states=archive['state'];ref=archive['reference_qpos']
    env=make_env(references('new')[:1],observation_dim=99);rows=[];samples=[]
    try:
        model=env.backend.model;data=mujoco.MjData(model)
        floors=set(np.flatnonzero((model.geom_type==mujoco.mjtGeom.mjGEOM_PLANE)&(model.geom_bodyid==0)).tolist())
        def contact(pose):
            data.qpos[:]=pose;mujoco.mj_fwdPosition(model,data);bodies=set();minimum=0.
            for item in data.contact:
                first,second=int(item.geom1),int(item.geom2)
                if (first in floors)==(second in floors) or item.dist>.001:continue
                geom=second if first in floors else first;bodies.add(model.body(int(model.geom_bodyid[geom])).name);minimum=min(minimum,float(item.dist))
            return bodies,minimum
        for index,row in enumerate(summary['clips']):
            if row['group']!='new':continue
            length=row['full_clip_steps'];actual=states[:length+1,index];reference=ref[:length+1,index]
            qpos=np.concatenate([actual[:,:7],actual[:,13:42]],1)
            refq=np.concatenate([reference[:,:3],Rotation.from_matrix(reference[:,3:12].reshape(-1,3,3)).as_quat(scalar_first=True),reference[:,12:]],1)
            actual_counts=Counter();reference_counts=Counter();trunk=[];all_dist=[];detail=[]
            for frame,(first,second) in enumerate(zip(qpos,refq)):
                actual_bodies,distance=contact(first);reference_bodies,ref_distance=contact(second)
                actual_counts.update(actual_bodies);reference_counts.update(reference_bodies);all_dist.append(distance)
                trunk.append(any(any(word in body.lower() for word in ['torso','pelvis','head']) for body in actual_bodies))
                detail.append({'time_s':frame*.02,'actual_ground_bodies':sorted(actual_bodies),'reference_ground_bodies':sorted(reference_bodies),
                               'actual_min_contact_distance_m':distance,'reference_min_contact_distance_m':ref_distance})
            result={'clip':row['clip'],'full_clip_steps':length,'original_hard_gate_passed':row['passed_original_hard_gate'],
                    'actual_ground_contact_geometry_samples_by_body':dict(actual_counts),'reference_ground_contact_geometry_samples_by_body':dict(reference_counts),
                    'actual_min_contact_distance_m':min(all_dist),'actual_min_after_initial_100ms_m':min(all_dist[5:]),
                    'head_torso_or_pelvis_ground_contact_samples':int(sum(trunk)),'actual_ending_ground_bodies':detail[-1]['actual_ground_bodies'],
                    'ending_root_speed_m_s':row['end_root_speed_m_s'],'contact_force_measured':False,'geometric_review_complete':True,
                    'physical_contact_fidelity_inferred':False}
            rows.append(result);samples.append({'clip':row['clip'],'samples':detail});print('PHYSICAL_CONTACT_REVIEW '+json.dumps({key:result[key] for key in ['clip','actual_min_after_initial_100ms_m','head_torso_or_pelvis_ground_contact_samples']}),flush=True)
        result={'run':a.run,'checkpoint_sha256':summary['checkpoint_sha256'],'new_denominator':len(rows),
                'method':'20 ms retained physical and reference poses evaluated on separate scratch MuJoCo MjData using the runtime collision model. Neither simulation nor references modified; no contact force reconstruction.',
                'clips':rows,'all_failures_retained':True,'contact_force_measured':False}
        out.write_text(json.dumps(result,indent=2)+'\n');(run/'contact-samples.json').write_text(json.dumps(samples,indent=2)+'\n')
    finally:env.close();env.backend.close()


if __name__=='__main__':main()
