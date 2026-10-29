"""Compare every new score and the complete traces of all stage3 failed new clips."""
import hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]


def main():
    modes=['unified-stage2-corrected-cuda-all55','unified-stage3-corrected-cuda-all55']
    summaries=[json.loads((ROOT/f'runs/{mode}/summary.json').read_text()) for mode in modes]
    lookups=[{r['clip']:(i,r) for i,r in enumerate(s['clips'])} for s in summaries]
    names=[r['clip'] for r in summaries[1]['clips'] if r['group']=='new' and not r['passed_original_hard_gate']]
    assert set(names)=={'88_07_stageii_unitree_g1','88_08_stageii_unitree_g1','88_09_stageii_unitree_g1'}
    traces=[np.load(ROOT/f'runs/{mode}/trajectory.npz') for mode in modes]
    rows=[];fig,axes=plt.subplots(len(names),3,figsize=(12,9),layout='constrained')
    try:
        for row_index,name in enumerate(names):
            indices=[lookup[name][0] for lookup in lookups];length=lookups[1][name][1]['full_clip_steps'];time=np.arange(length+1)*.02
            reference=traces[1]['reference_qpos'][:length+1,indices[1]]
            assert np.array_equal(reference,traces[0]['reference_qpos'][:length+1,indices[0]])
            axes[row_index,0].plot(time,reference[:,2],color='#333333',ls='--',label='Reference')
            axes[row_index,1].plot(time,reference[:,11],color='#333333',ls='--',label='Reference')
            result={'clip':name,'complete_window_s':float(time[-1]),'reference_bytes_equal_between_stages':True,'stages':[]}
            for j,(trace,index,color,label) in enumerate(zip(traces,indices,['#557f9b','#ba5f45'],['Stage2','Stage3'])):
                state=trace['state'][:length+1,index];rzz=1-2*(state[:,4]**2+state[:,5]**2)
                xy=trace['xy_error'][:length+1,index];height=state[:,2];original=lookups[j][name][1];failure=original['first_original_hard_failure_s']
                for axis,values in zip(axes[row_index],[height,rzz,xy]):
                    axis.plot(time,values,color=color,lw=1.4,label=label)
                    if failure is not None:axis.axvline(failure,color=color,ls=':',alpha=.65)
                invert=reference[:,11]<-.5
                item={'stage':j+2,'score_until_original_failure':original['norm_return_original'],'original_hard_gate_passed':original['passed_original_hard_gate'],
                      'first_original_hard_failure_s':failure,'full_trace_xy_rmse_m':float(np.sqrt(np.mean(xy**2))),
                      'full_trace_body_local_mean_m':float(trace['body_local'][:length+1,index].mean()),
                      'full_trace_body_global_mean_m':float(trace['body_global'][:length+1,index].mean()),
                      'inverted_reference_frames':int(invert.sum()),'inversion_coverage':float((rzz[invert]<-.5).mean()) if invert.any() else None,
                      'height_error_max_m':float(abs(height-reference[:,2]).max()),'Rzz_error_max':float(abs(rzz-reference[:,11]).max())}
                if failure is not None:
                    f=int(round(failure/.02));item['aligned_pose_at_first_original_failure']={'time_s':float(time[f]),'actual_z_m':float(height[f]),'reference_z_m':float(reference[f,2]),
                        'actual_Rzz':float(rzz[f]),'reference_Rzz':float(reference[f,11]),'stored_original_gate':float(trace['hard_failure_original'][max(0,f-1),index]),
                        'stored_aligned_gate':float(trace['hard_failure_aligned'][f,index])}
                result['stages'].append(item)
            rows.append(result)
            axes[row_index,0].set_ylabel(name.removesuffix('_stageii_unitree_g1')+'\nRoot height (m)')
            axes[row_index,1].set_ylabel('Root Rzz')
            axes[row_index,2].set_ylabel('World XY error (m)')
            for axis in axes[row_index]:axis.grid(alpha=.15);axis.set_xlabel('Complete clip time (s)');axis.spines[['top','right']].set_visible(False)
        axes[0,0].legend(frameon=False);fig.suptitle('All3 stage3 failed new clips: complete retained physical traces')
        fig.supxlabel('Dotted vertical lines: first original hard failure; no reset or tail removal. Same fixed calibrated references.',fontsize=9)
        for ext in ['png','svg']:fig.savefig(ROOT/f'figures/stage3-failure-profiles.{ext}',dpi=160)
        result={'status':'all_stage3_failed_new_clips_diagnosed_from_full_traces','selected_by':'all new clips failing original hard gate in stage3 calibrated CUDA; no success selection',
                'reference_variant':'ground-calibration-v1','thresholds_unchanged':True,'no_causal_failure_mechanism_inferred_from_pose_only':True,'clips':rows,
                'all15_score_changes':[{'clip':name,'stage2':lookups[0][name][1]['norm_return_original'],'stage3':row['norm_return_original']} for name,(_,row) in lookups[1].items() if row['group']=='new']}
        (ROOT/'evidence/stage3-failure-diagnosis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    finally:
        for trace in traces:trace.close()
        plt.close(fig)


if __name__=='__main__':main()
