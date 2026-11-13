"""Render all eight measured horizons from the exact original forecast JSON."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

folder=Path(__file__).parent
result=json.loads((folder/'saved-wm-forecast-result.json').read_bytes())
fig,ax=plt.subplots(figsize=(8.9,5.6),layout='constrained')
for label,title,color in [('1k','1,001 training collections','#b55e29'),('16k','16,001 training collections','#24688d')]:
    case=result['models'][label]
    values=case['normalized_state_mse_by_horizon']
    ax.plot(range(1,9),values,marker='o',lw=2,color=color,label=title)
    ax.annotate(f'{values[-1]:.5g}',(8,values[-1]),xytext=(-8,8),textcoords='offset points',ha='right',color=color,fontweight='bold')
ax.set_yscale('log')
ax.set_xticks(range(1,9))
ax.set_xlabel('Predicted transitions (20 ms per transition)')
ax.set_ylabel('Native nominal-weighted state MSE (70 channels)')
ax.set_title('Frozen world models on identical saved physical trajectories',loc='left',fontweight='bold')
ax.grid(True,which='major',alpha=.2)
ax.legend(loc='upper left')
fig.supxlabel('All 437,276 complete windows from 40 corrected LAFAN trajectories, including the failed clip.\n'
    'Nine states + eight actual actions per window; open-loop forecasts with no teacher forcing.\n'
    'Offline fixed-policy diagnostic: not the paper online dataset-size sweep or held-out generalization.',fontsize=9)
for extension in ('png','svg'):
    fig.savefig(folder/f'saved-wm-forecast-horizons.{extension}',dpi=180,bbox_inches='tight')
plt.close(fig)
