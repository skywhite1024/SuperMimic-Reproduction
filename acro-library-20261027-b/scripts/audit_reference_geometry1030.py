"""Check unchanged reference poses against the runtime joint limits and collision records."""
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import psutil
from runtime import ROOT, make_env, references

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    plan_path=ROOT/'evidence/reference-geometry-1030-plan.json'
    plan=json.loads(plan_path.read_text(encoding='utf8'))
    assert sha(Path(__file__))==plan['script_sha256']
    for name,digest in plan['source_sha256'].items():
        assert sha(ROOT/name)==digest,name
    out=ROOT/'evidence/reference-geometry-1030.json'
    assert not out.exists()
    process=psutil.Process();process.cpu_affinity(process.cpu_affinity()[:2]);start=time.monotonic()
    env=make_env(references('new')[:1],device='cpu',observation_dim=99,evaluation=True)
    try:
        model=env.backend.model
        joints=[i for i in range(model.njnt) if model.jnt_qposadr[i]>=7]
        assert len(joints)==29 and np.array_equal(model.jnt_qposadr[joints],np.arange(7,36))
        ranges=model.jnt_range[joints].copy();limited=model.jnt_limited[joints].astype(bool)
        rows=[]
        for variant,mode in plan['modes'].items():
            summary=json.loads((ROOT/f'runs/{mode}/summary.json').read_text())
            contacts=json.loads((ROOT/f'runs/{mode}/contact-samples.json').read_text());lookup={row['clip']:row['samples'] for row in contacts}
            with np.load(ROOT/f'runs/{mode}/trajectory.npz') as data:
                for i,row in enumerate(summary['clips']):
                    assert time.monotonic()-start<plan['maximum_seconds']
                    q=data['reference_qpos'][:row['full_clip_steps']+1,i,12:]
                    assert q.shape[1]==29 and np.isfinite(q).all()
                    violation=np.maximum(np.maximum(ranges[:,0]-q,q-ranges[:,1]),0)*limited
                    geometric=lookup[row['clip']][5:]
                    worst=int(violation.max(0).argmax())
                    rows.append({'reference':variant,'clip':row['clip'],'reference_frames':len(q),
                                 'reference_joint_limit_violating_frames':int((violation>1e-6).any(-1).sum()),
                                 'maximum_reference_hard_joint_limit_excess_rad':float(violation.max()),
                                 'worst_joint':model.joint(joints[worst]).name if violation.max()>1e-6 else None,
                                 'reference_ground_distance_min_after_100ms_m':min(sample['reference_min_contact_distance_m'] for sample in geometric),
                                 'reference_penetration_over_5cm_samples_after_100ms':sum(sample['reference_min_contact_distance_m']<-.05 for sample in geometric),
                                 'protected':row['clip'] in plan['protected_new'],'reference_modified':False})
        for name,digest in plan['source_sha256'].items():
            assert sha(ROOT/name)==digest,name
        result={'status':'complete_all30_unchanged_reference_pose_geometry_audited','checked_at':datetime.now(timezone.utc).isoformat(),
                'plan_sha256':sha(plan_path),'CPU_affinity':process.cpu_affinity(),'wall_seconds':time.monotonic()-start,'records':rows,
                'GPU':0,'training_updates':0,'new_physical_rollouts':0,'references_modified':False,'acceptance_thresholds_changed':False,
                'contact_force_measured':False,'dynamic_feasibility_proved':False,'full_goal_complete':False}
        assert len(rows)==30 and result['wall_seconds']<plan['maximum_seconds']
        out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
        print(json.dumps({'status':result['status'],'protected':[row for row in rows if row['protected']],'wall_seconds':result['wall_seconds']},indent=2))
    finally:
        env.close();env.backend.close()

if __name__=='__main__':
    main()
