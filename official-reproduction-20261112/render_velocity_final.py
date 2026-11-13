"""Plot all final standing commands with shared error scale and fall markers."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

folder=Path(__file__).parent
planes=[('vx-vy',0,1),('vx-wz',0,2),('vy-wz',1,2)]
labels=[r'$v_x$ (m/s)',r'$v_y$ (m/s)',r'$\omega_z$ (rad/s)']
fig,axes=plt.subplots(2,3,figsize=(11.4,8.6),layout='constrained')
cmap=plt.colormaps['YlOrRd'].copy()
cmap.set_bad('#e5e7eb')
metrics={}
for n,mode in enumerate(('rl','distill')):
    result=json.loads((folder/f'velocity-{mode}-100k-step-result.json').read_bytes())
    metrics[mode]=result
    assert len(result['records'])==672 and result['total_trials']==2688
    for j,(plane,yi,xi) in enumerate(planes):
        records=[r for r in result['records'] if r['plane']==plane]
        assert len(records)==224
        values=np.full((15,15),np.nan)
        for record in records:
            values[record['row'],record['column']]=record['display_relative_error']
        ax=axes[n,j]
        im=ax.imshow(values,origin='lower',cmap=cmap,vmin=0,vmax=.9,interpolation='nearest')
        xs=sorted({r['command'][xi] for r in records})
        ys=sorted({r['command'][yi] for r in records})
        assert len(xs)==len(ys)==15
        xticks=[0,3,7,11,14]
        yticks=[0,3,6,10,14] if yi==0 else xticks
        fmt=lambda value: f'{value:.2f}'.rstrip('0').rstrip('.')
        ax.set_xticks(xticks,[fmt(xs[k]) for k in xticks],fontsize=9)
        ax.set_yticks(yticks,[fmt(ys[k]) for k in yticks],fontsize=9)
        ax.set_xlabel(labels[xi]);ax.set_ylabel(labels[yi])
        ax.set_title(f"{'RL' if mode=='rl' else 'Distillation'}: {plane}")
        fall=[r for r in records if r['non_falling_trials']<4]
        if fall:
            ax.scatter([r['column'] for r in fall],[r['row'] for r in fall],marker='^',s=80,
                facecolors='none',edgecolors='black',linewidths=1.6)
headline='   |   '.join(f"{label}: median {metrics[mode]['median_defined_command_error']:.2%}, "
    f"{metrics[mode]['fallen_trials']}/2688 falls" for mode,label in [('rl','RL'),('distill','Distillation')])
fig.suptitle('100k-trained velocity interfaces: all standing step-test commands\n'+headline,fontsize=14,fontweight='bold')
bar=fig.colorbar(im,ax=axes,shrink=.85,pad=.02,format=PercentFormatter(1))
bar.set_label('Command median relative error (display capped at 90%)')
fig.supxlabel('Norm of final-2s mean velocity error / command norm; median across non-falling trials.\n'
    'Triangles: at least one falling trial. Gray: zero command excluded. One training seed, nominally identical reset poses.\n'
    'Local standing protocol; exact author library/pose and strict paper equivalence remain unconfirmed.',fontsize=9)
for extension in ('png','svg'):
    fig.savefig(folder/f'velocity-100k-step-errors.{extension}',dpi=160,bbox_inches='tight')
plt.close(fig)
