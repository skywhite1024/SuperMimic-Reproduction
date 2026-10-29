"""Continuous tracking review with per-new-segment checks and a fixed terminal hold."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args()
    folder=ROOT/'runs'/args.run;summary=json.loads((folder/'summary.json').read_text())
    contract_file=ROOT/'continuous-acceptance-contract-v1.json';limits=json.loads(contract_file.read_text())
    index_file=ROOT/summary['reference_version']/'index.json';assert hashlib.sha256(index_file.read_bytes()).hexdigest()==summary['frozen_reference_index_sha256']
    index=json.loads(index_file.read_text());rows=[]
    with np.load(folder/'trajectory.npz') as source:trace={key:source[key] for key in source.files}
    assert np.isfinite(trace['state']).all()
    for i,(trial,sequence) in enumerate(zip(summary['trials'],index['sequences'])):
        state=trace['state'][:,i];reference=trace['reference_qpos'][:,i];quat=state[:,3:7].astype(np.float64);quat/=np.linalg.norm(quat,axis=1,keepdims=True)
        w,x,y,z=quat.T
        matrices=np.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],axis=-1).reshape(-1,3,3)
        desired=reference[:,3:12].reshape(-1,3,3);angles=np.arccos(np.clip((np.sum(matrices*desired,axis=(1,2))-1)/2,-1,1))
        def window_metrics(start,stop):
            window=slice(start,stop);inverse=desired[window,2,2]<-.5
            metrics={'root_xy_rmse_m':float(np.sqrt(np.mean(trace['xy_error'][window,i]**2))),
                     'root_xy_peak_m':float(trace['xy_error'][window,i].max()),'body_global_mean_m':float(trace['body_global'][window,i].mean()),
                     'body_local_mean_m':float(trace['body_local'][window,i].mean()),'yaw_rmse_deg':float(np.rad2deg(np.sqrt(np.mean(trace['yaw_error'][window,i]**2)))),
                     'root_orientation_geodesic_rmse_deg':float(np.rad2deg(np.sqrt(np.mean(angles[window]**2)))),
                     'inversion_frame_coverage':float(np.mean(matrices[window,2,2][inverse]<-.5)) if inverse.any() else None,
                     'hard_failure_samples':int(trace['hard_failure'][window,i].sum())}
            checks={key:value<=limits[key+'_max'] for key,value in metrics.items() if key not in ['inversion_frame_coverage','hard_failure_samples']}
            checks['inversion_coverage']=metrics['inversion_frame_coverage'] is None or metrics['inversion_frame_coverage']>=limits['reference_inversion_frame_coverage_min']
            checks['no_hard_failure']=metrics['hard_failure_samples']==0
            return {**metrics,'checks':checks,'passed':all(checks.values())}
        entire=window_metrics(0,len(state));segments=[]
        for segment in sequence['segments']:
            if segment['source'] not in index['new_full_clips_in_union']:continue
            start=segment['source_start_in_sequence_frame'];stop=segment['end_frame']+1
            segments.append({'source':segment['source'],'physical_start_s':start*.02,'physical_end_s':(stop-1)*.02,**window_metrics(start,stop)})
        terminal={'upright':trial['terminal_last_second_pelvis_min_m']>=limits['terminal_pelvis_min_m'] and trial['terminal_last_second_Rzz_min']>=limits['terminal_Rzz_min'],
                  'root_speed':trial['terminal_last_second_root_speed_peak_m_s']<=limits['terminal_root_speed_peak_m_s_max'],
                  'joint_speed':trial['terminal_last_second_joint_speed_rms_rad_s']<=limits['terminal_joint_speed_rms_rad_s_max']}
        checks={'complete_120_seconds':trial['elapsed_seconds']==120,'no_resets':trial['post_initialization_resets']==0,
                'whole_sequence_tracking':entire['passed'],'all_complete_new_segments_tracking':all(row['passed'] for row in segments),
                'terminal_hold':all(terminal.values())}
        rows.append({'seed':trial['seed'],'checks':checks,'continuous_engineering_passed':all(checks.values()),'whole_sequence':entire,
                     'new_full_segments':segments,'terminal_checks':terminal,'original_trial':trial})
    result={'run':args.run,'checkpoint_sha256':summary['checkpoint_sha256'],'reference_version':summary['reference_version'],
            'frozen_index_sha256':summary['frozen_reference_index_sha256'],'contract_sha256':hashlib.sha256(contract_file.read_bytes()).hexdigest(),
            'denominator':3,'passed_original_hard_gate':summary['passed_original_hard_gate'],'continuous_engineering_passed':sum(row['continuous_engineering_passed'] for row in rows),
            'stage1_contract_application_retrospective':args.run.startswith('unified4000'),'trials':rows}
    target=folder/'continuous-assessment.json';assert not target.exists();target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:value for key,value in result.items() if key!='trials'}),flush=True)

if __name__=='__main__':main()
