"""Render the recovered original timing measurements with explicit path scopes."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

folder=Path(__file__).parent
with (folder/'benchmark-sweep-measurements.csv').open(encoding='utf-8',newline='') as stream:
    rows=list(csv.DictReader(stream))
x=[int(row['num_envs']) for row in rows]
fig,ax=plt.subplots(figsize=(9.3,5.7),layout='constrained')
for key,label,color in [
    ('simulator_actor_sps','Native env.step + actor','#235789'),
    ('wm_grad_recording_sps','WM dynamics forward (grad recording)','#c76d2c'),
    ('wm_inference_sps','WM dynamics forward (inference)','#26866e')]:
    ax.plot(x,[float(row[key]) for row in rows],marker='o',lw=1.8,ms=4,label=label,color=color)
ax.set_xscale('log',base=2)
ax.set_yscale('log')
ax.set_xticks(x,[f'{v:,}' for v in x],rotation=35,ha='right')
ax.yaxis.set_major_formatter(FuncFormatter(lambda value,pos:f'{value:,.0f}'))
ax.set_xlabel('Parallel batch size')
ax.set_ylabel('Batched state / control transitions per second')
ax.set_title('H100 throughput diagnostic: 11 completed batch sizes',loc='left',fontweight='bold')
ax.grid(True,which='major',alpha=.2)
ax.legend(loc='upper left',fontsize=9)
fig.supxlabel('20 warmups per path; median of 3 × 500 timed calls; CUDA synchronized.\nUnequal computation scopes: WM uses fixed random inputs; no actor, reward or backward pass.\nThis figure does not establish paper full-control speedup or tracking quality.',fontsize=9)
for extension in ('png','svg'):
    fig.savefig(folder/f'benchmark-sweep-throughput.{extension}',dpi=180)
plt.close(fig)
