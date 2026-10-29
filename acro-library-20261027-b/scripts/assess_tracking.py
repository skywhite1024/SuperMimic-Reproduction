"""Apply frozen quality criteria to every complete physical trajectory, without exclusions."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
    folder=ROOT/'runs'/a.run;summary=json.loads((folder/'summary.json').read_text())
    contract_path=ROOT/'acceptance-contract.json';contract=json.loads(contract_path.read_text());limits=contract['new_clip_tracking']
    trace=np.load(folder/'trajectory.npz');all_states=trace['state'];all_references=trace['reference_qpos'];rows=[]
    for index,row in enumerate(summary['clips']):
        length=row['full_clip_steps'];states=all_states[:length+1,index];ref=all_references[:length+1,index]
        quat=states[:,3:7].astype(np.float64);quat/=np.linalg.norm(quat,axis=1,keepdims=True)
        w,x,y,z=quat.T
        rotations=np.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],axis=-1).reshape(-1,3,3)
        references=ref[:,3:12].reshape(-1,3,3).astype(np.float64)
        angle=np.arccos(np.clip((np.sum(rotations*references,axis=(1,2))-1)/2,-1,1))
        ref_inverted=references[:,2,2]<-.5
        coverage=float(np.mean(rotations[ref_inverted,2,2]<-.5)) if ref_inverted.any() else None
        ending=min(25,len(states));upright_reference=bool(np.all(references[-ending:,2,2]>.8) and np.all(ref[-ending:,2]>.55))
        metrics={**row,'root_orientation_geodesic_rmse_deg':float(np.rad2deg(np.sqrt(np.mean(angle**2)))),
                 'inversion_frame_coverage':coverage,'reference_ending_upright':upright_reference,
                 'end_half_second_pelvis_min_m':float(states[-ending:,2].min()),'end_half_second_Rzz_min':float(rotations[-ending:,2,2].min())}
        checks={'no_original_hard_failure':row['passed_original_hard_gate'],'no_resets':row['post_initialization_resets']==0}
        for key in ['root_xy_rmse_m','root_xy_peak_m','body_global_mean_m','body_local_mean_m','yaw_rmse_deg','root_orientation_geodesic_rmse_deg']:
            checks[key]=metrics[key]<=limits[key+'_max']
        checks['norm_return_original']=row['norm_return_original']>=limits['norm_return_original_min']
        checks['inversion_coverage']=coverage is None or coverage>=limits['reference_inversion_frame_coverage_min']
        checks['upright_ending']=not upright_reference or (metrics['end_half_second_pelvis_min_m']>=.55 and metrics['end_half_second_Rzz_min']>=.8)
        metrics.update(quality_checks=checks,tracking_quality_passed=all(checks.values()),failed_checks=[key for key,value in checks.items() if not value])
        rows.append(metrics)
    new=[row for row in rows if row['group']=='new'];result={'run':a.run,'checkpoint_sha256':summary['checkpoint_sha256'],'device':summary['device'],
        'contract_sha256':hashlib.sha256(contract_path.read_bytes()).hexdigest(),'new_denominator':len(new),
        'new_original_hard_gate_passed':sum(row['passed_original_hard_gate'] for row in new),'new_tracking_quality_passed':sum(row['tracking_quality_passed'] for row in new),'clips':rows}
    target=folder/'tracking-assessment.json';assert not target.exists();target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:value for key,value in result.items() if key!='clips'}))


if __name__=='__main__':main()
