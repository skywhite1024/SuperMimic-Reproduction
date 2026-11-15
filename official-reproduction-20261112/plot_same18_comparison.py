"""Plot verified fixed18 evaluation points without claiming paper speedups."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    data=json.loads((args.root/'finite-same18-comparison-1116.json').read_bytes())
    assert data['identical_physical_motion_file_hashes'] and data['identical_native_termination_rules']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator,FuncFormatter,NullFormatter
    fig,axes=plt.subplots(1,2,figsize=(13,5.4),sharey=True,layout='constrained')
    curves=[('Native first-order: saved pre-stop weights',data['first_order'],'black')]
    colors=['#0072b2','#e69f00','#009e73','#cc79a7']
    curves += [('PPO-WM '+row['label'],row['points'],color) for row,color in zip(data['PPO_WM'],colors,strict=True)]
    for label,points,color in curves:
        values=[p['mean_normalized_full_motion_return'] for p in points]
        for ax,key,scale in [(axes[0],'real_simulator_steps',1e6),(axes[1],'checkpoint_clock_seconds',60)]:
            ax.plot([p[key]/scale for p in points],values,'o-',label=label,color=color,
                    lw=2.3 if color=='black' else 1.5,markersize=5)
    for ax in axes:
        ax.set_xscale('log');ax.set_ylim(-.015,1.025);ax.grid(alpha=.22)
        ax.axhline(.82,color='#777777',ls=':',lw=1)
        ax.xaxis.set_minor_formatter(NullFormatter())
    axes[0].set_xlabel('Real simulator transitions (millions; log scale)')
    axes[1].set_xlabel('Reported training checkpoint clock (minutes; log scale)')
    axes[0].set_ylabel('Mean normalized full-motion return')
    axes[0].xaxis.set_major_locator(FixedLocator([.004,.04,.4,4,40,100]))
    axes[0].xaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{x:g}'))
    axes[1].xaxis.set_major_locator(FixedLocator([.3,1,3,10,30,100]))
    axes[1].xaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{x:g}'))
    axes[0].legend(loc='upper left',fontsize=8.5)
    axes[1].text(.98,.03,'Dotted line: local full-motion return 0.82\nClock origins differ; offline evaluation excluded',
                 transform=axes[1].transAxes,ha='right',va='bottom',fontsize=9)
    fig.suptitle('Fixed18 locomotion clips · seed42 · complete native evaluations at every shown point\n'
                 'Interrupted first-order parent remains interrupted; author Figure5 protocol and speedups unconfirmed',fontsize=11)
    for suffix in ('png','svg'):fig.savefig(args.root/('finite-same18-comparison-1116.'+suffix),dpi=160)
    plt.close(fig)
    print('Rendered six first-order points and four finite PPO-WM curves.')


if __name__=='__main__':main()
