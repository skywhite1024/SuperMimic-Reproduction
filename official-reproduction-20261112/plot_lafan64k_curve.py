"""Plot a verified nine-point native curve with the completed H32 no-critic endpoint."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--curve',type=Path,required=True,help='Recovered main-receipts folder')
    parser.add_argument('--without-critic',type=Path,default=Path(__file__).parent/'critic-free-main-1113ah')
    args=parser.parse_args();folder=args.curve.resolve();no=args.without_critic.resolve()
    validation=json.loads((folder/'validation.json').read_bytes())
    assert validation['status']=='complete_native64001_ninepoint_full40_curve_originals_verified'
    points=validation['points'];assert len(points)==9
    no_accounting=json.loads((no/'eval-000016001/accounting.json').read_bytes())
    no_timing=json.loads((no/'training/checkpoint-times.json').read_bytes())['checkpoints'][-1]
    with_protocol=json.loads((folder/'eval-000016001/protocol.json').read_bytes())
    no_protocol=json.loads((no/'eval-000016001/protocol.json').read_bytes())
    assert with_protocol['motion_files']==no_protocol['motion_files']
    assert no_timing['collection_steps']==16001 and no_accounting['clips']==40
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'savefig.facecolor':'white'})
    fig,axes=plt.subplots(1,2,figsize=(12.8,5.4),sharey=True)
    ys=[row['mean_normalized_full_motion_return'] for row in points]
    xs=[[row['real_simulator_steps']/1e6 for row in points],
        [row['seconds_since_runner_run_entry']/60 for row in points]]
    no_xs=[16001*4096/1e6,no_timing['retained_snapshot_complete']['seconds_since_runner_run_entry']/60]
    no_y=no_accounting['mean_normalized_full_motion_return']
    for column,(axis,xs_column,no_x) in enumerate(zip(axes,xs,no_xs,strict=True)):
        axis.plot(xs_column,ys,'o-',color='#275EA8',lw=2,ms=5,label='With critic: 9 sampled checkpoints')
        axis.scatter([no_x],[no_y],s=90,marker='D',color='#D16A17',edgecolor='white',zorder=5,
                     label='Without critic: 16k endpoint')
        axis.axhline(.82,color='#777777',ls='--',lw=1,label='0.82 sampled-score threshold')
        for index,(x,y,row) in enumerate(zip(xs_column,ys,points,strict=True)):
            axis.annotate(f"{row['successes']}/40",(x,y),xytext=(0,8 if index%2==0 else -15),
                          textcoords='offset points',ha='center',fontsize=8,color='#275EA8')
        axis.annotate(f"{no_accounting['successes']}/40",(no_x,no_y),xytext=(7,-12),
                      textcoords='offset points',fontsize=9,color='#AD520B')
        axis.grid(alpha=.2);axis.set_ylim(-.055,1.055)
        axis.margins(x=.055)
        axis.set_xlabel('Real simulator interactions (millions)' if column==0
                        else 'Snapshot availability since native runner entry (min)')
    axes[0].set_ylabel('Mean normalized full-motion return')
    axes[0].legend(loc='lower right',fontsize=8,framealpha=.92)
    fig.suptitle('LAFAN40, H32: fixed-start performance across finite training',fontsize=15,y=.97)
    fig.text(.5,.035,'Single seed 42; 40 motions; failed tails contribute zero. Timing includes warmup and snapshot I/O.',
             ha='center',fontsize=9,color='#444444')
    fig.subplots_adjust(left=.08,right=.985,bottom=.16,top=.87,wspace=.08)
    for extension in ('png','svg'):
        fig.savefig(folder/f'lafan64k-learning-curve.{extension}',dpi=180)
    plt.close(fig)
    matched=next(row for row in points if row['collection_steps']==16001)
    result=dict(status='derived_plot_from_verified_originals',fixed_start_protocol='manual-reset-v2',
        same_canonical_motion_hashes=True,seed=42,critic_points=len(points),
        no_critic_endpoint=dict(collection_steps=16001,successes=no_accounting['successes'],mean=no_y),
        matched_cadence_16k_critic=dict(successes=matched['successes'],
            mean=matched['mean_normalized_full_motion_return']),
        first_observed_sample_at_or_above_082=validation['first_observed_sample_at_or_above_082'],
        exact_crossing_time_or_multiseed_effect_claimed=False,paper_results_reproduced=False,
        sources={str(path):digest(path) for path in [folder/'validation.json',
            no/'eval-000016001/accounting.json',no/'training/checkpoint-times.json']},
        plots={extension:dict(file=f'lafan64k-learning-curve.{extension}',
                sha256=digest(folder/f'lafan64k-learning-curve.{extension}')) for extension in ('png','svg')})
    (folder/'learning-curve-analysis.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({key:value for key,value in result.items() if key not in ('sources','plots')},indent=2))


if __name__=='__main__':main()
