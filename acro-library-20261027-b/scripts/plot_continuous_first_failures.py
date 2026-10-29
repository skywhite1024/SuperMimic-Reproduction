"""Show all three initial failures, with full unabridged traces retained separately."""
import hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def main():
    modes=['stage3-corrected-continuous-v1','stage3-actionbound20-corrected-continuous-v1']
    traces=[];summaries=[]
    for mode in modes:
        folder=ROOT/'runs'/mode;summaries.append(json.loads((folder/'summary.json').read_text()))
        with np.load(folder/'trajectory.npz') as archive:traces.append({key:archive[key] for key in ['state','reference_qpos']})
    assert np.array_equal(traces[0]['reference_qpos'],traces[1]['reference_qpos'])
    index=json.loads((ROOT/'continuous-corrected-v1/index.json').read_text())
    fig,axes=plt.subplots(3,2,figsize=(12.8,8.3),sharex='row',layout='constrained')
    time=np.arange(6001)*.02
    for slot,(trial,sequence) in enumerate(zip(summaries[0]['trials'],index['sequences'])):
        failure=trial['first_hard_failure_s'];reference=traces[0]['reference_qpos'][:,slot]
        for column in [0,1]:
            ax=axes[slot,column]
            for segment in sequence['segments']:
                start,stop=segment['transition_start_frame']*.02,segment['source_start_in_sequence_frame']*.02
                if stop>start and start<failure+2 and stop>failure-2:ax.axvspan(start,stop,color='#a5bed3',alpha=.25,label='reference bridge' if slot==0 else None)
            ref=reference[:,2] if column==0 else reference[:,11]
            ax.plot(time,ref,color='#269264',linewidth=2,label='reference')
            for trace,label,color,style in zip(traces,['unbounded','bound20'],['#d05d39','#315fc0'],['-','--']):
                state=trace['state'][:,slot];value=state[:,2] if column==0 else 1-2*(state[:,4]**2+state[:,5]**2)
                ax.plot(time,value,color=color,linestyle=style,linewidth=1.6,label=label)
            ax.axvline(failure,color='#343b43',linestyle=':',linewidth=1.3)
            ax.set_xlim(max(0,failure-2),failure+2);ax.grid(alpha=.22)
            ax.set_ylabel('pelvis height (m)' if column==0 else 'root Rzz')
            if column==1:ax.set_ylim(-1.1,1.1)
            ax.set_title(f'seed {trial["seed"]}: first failure {failure:.2f} s',fontsize=11,loc='left')
            if slot==2:ax.set_xlabel('physical time (s)')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncols=4,frameon=False)
    fig.suptitle('Same weights and references: output bounding leaves all three initial failures unchanged',fontsize=13)
    target=ROOT/'figures/continuous-first-failure-diagnostic'
    files=[]
    for extension in ['png','svg','pdf']:
        path=target.with_suffix('.'+extension);fig.savefig(path,dpi=160);files.append({'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    plt.close(fig)
    (target.with_suffix('.json')).write_text(json.dumps({'status':'all_three_failures_included','scope':'Local diagnostic windows only; full120s traces and complete videos retained. This is not a success montage.','matched_device':'cpu','reference_variant':'ground-calibration-v1','files':files,
        'input_trajectory_sha256':{mode:hashlib.sha256((ROOT/'runs'/mode/'trajectory.npz').read_bytes()).hexdigest() for mode in modes}},indent=2)+'\n')
    print(json.dumps({'figure':str(target.with_suffix('.png'))}),flush=True)

if __name__=='__main__':main()
