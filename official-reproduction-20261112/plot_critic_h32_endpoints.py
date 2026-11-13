"""Plot every canonical40 fixed-start endpoint; no multi-seed inference."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path(__file__).parent
    after_root=root/'critic-free-main-1113ah'
    before=json.loads((root/'manual-reset-v2-lafan40-result.json').read_bytes())
    after=json.loads((after_root/'eval-000016001/result.json').read_bytes())
    before_by_name={row['name']:row for row in before['records']}
    rows=[]
    for row in after['records']:
        old=before_by_name[row['name']]
        assert row['reference_frames']==old['reference_frames']
        x=old['reward_sum']/old['reference_frames']/.1
        y=row['reward_sum']/row['reference_frames']/.1
        rows.append(dict(name=row['name'],with_critic=x,without_critic=y,delta=y-x,
            with_critic_success=old['success'],without_critic_success=row['success'],
            reference_frames=row['reference_frames'],with_critic_saved_steps=old['steps'],
            without_critic_saved_steps=row['steps']))
    assert len(rows)==40 and len({row['name'] for row in rows})==40
    with (after_root/'critic-endpoint-per-clip.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    cases=[(True,True,'Both succeed','#2471a3'),(True,False,'Critic only succeeds','#d95f02'),
        (False,True,'No critic only succeeds','#238b45'),(False,False,'Both fail','#777777')]
    fig,ax=plt.subplots(figsize=(7.1,7.6))
    fig.subplots_adjust(left=.13,right=.98,top=.84,bottom=.24)
    for first,second,label,color in cases:
        group=[row for row in rows if row['with_critic_success']==first and row['without_critic_success']==second]
        if group:ax.scatter([row['with_critic'] for row in group],[row['without_critic'] for row in group],
            s=49,c=color,alpha=.85,edgecolors='white',linewidths=.5,label=f'{label} ({len(group)})')
    low=min(0.,min(min(row['with_critic'],row['without_critic']) for row in rows)-.01)
    high=max(1.,max(max(row['with_critic'],row['without_critic']) for row in rows)+.01)
    ax.plot([low,high],[low,high],ls='--',c='#888888',lw=1)
    ax.set(xlim=(low,high),ylim=(low,high),xlabel='With critic: normalized full-motion return',
        ylabel='Without critic: normalized full-motion return')
    ax.set_aspect('equal');ax.grid(alpha=.2);ax.set_axisbelow(True)
    means=[sum(row[key] for row in rows)/40 for key in ['with_critic','without_critic']]
    ax.scatter(*means,c='black',marker='*',s=160,zorder=6,label='Mean of all 40 clips')
    for index,row in enumerate(sorted(rows,key=lambda row:row['delta'])[:3]):
        ax.annotate(row['name'],(row['with_critic'],row['without_critic']),xytext=(.27,.22-index*.07),
            textcoords='axes fraction',fontsize=8,ha='left',va='center',
            bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=1),
            arrowprops=dict(arrowstyle='-',color='#999999',lw=.6))
    fig.suptitle('H32 critic ablation: 16,001 collections, seed 42',fontsize=13,y=.975)
    fig.text(.5,.925,'With critic: 39/40, mean 0.7971  |  Without critic: 25/40, mean 0.6355',
        ha='center',fontsize=10)
    handles,labels=ax.get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.115),ncol=2,fontsize=9,frameon=False)
    fig.text(.5,.065,'Same canonical 40 and v2 fixed-start protocol; failed tails contribute zero.',ha='center',fontsize=9)
    fig.text(.5,.037,'One seed. Prior critic save cadence differs. This is not training-episode return.',ha='center',fontsize=8.5,color='#555555')
    plt.rcParams['svg.fonttype']='none'
    for extension in ['png','svg']:
        fig.savefig(after_root/f'critic-endpoint-comparison.{extension}',dpi=180)
    plt.close(fig)
    analysis=dict(scope='Single seed fixed-start full40 endpoint; all failures retained',
        with_critic_mean=means[0],without_critic_mean=means[1],mean_difference=means[1]-means[0],
        clips_with_lower_without_critic=sum(row['delta']<0 for row in rows),
        clips_with_higher_without_critic=sum(row['delta']>0 for row in rows),
        success_transitions={label:sum(row['with_critic_success']==first and row['without_critic_success']==second for row in rows)
            for first,second,label,color in cases},
        most_negative_deltas=sorted(rows,key=lambda row:row['delta'])[:5],
        uncertainty_or_causal_effect_established=False,paper_results_reproduced=False)
    (after_root/'endpoint-analysis.json').write_text(json.dumps(analysis,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(analysis,indent=2))


if __name__=='__main__':main()
