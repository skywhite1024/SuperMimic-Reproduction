"""Plot the recovered, original stage-one training log without inferring acceptance."""
import csv,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def main():
    source=ROOT/'runs/unified-h100-4000/train_log_with_header.csv'
    with source.open(encoding='utf8',newline='') as stream:rows=list(csv.DictReader(stream))
    x=np.array([int(row['iteration'])-18000 for row in rows]);assert len(rows)==80 and x[0]==50 and x[-1]==4000
    fig,axes=plt.subplots(2,2,figsize=(11,6.4),sharex=True,layout='constrained')
    for axis,key,label,color in zip(axes.flat,['wm_loss','pi_loss','sim_rew','ep_ret_norm'],
                                    ['World-model loss','Policy loss','Normalized training reward','Training episode return'],
                                    ['#325a86','#a6532e','#2f7b62','#785591']):
        values=np.array([float(row[key]) for row in rows])
        if key=='pi_loss':values[x<=800]=np.nan
        axis.axvspan(0,800,color='#e9eef4',label='WM-only warmup')
        axis.plot(x,values,color=color,lw=1.6)
        axis.set_ylabel(label);axis.grid(alpha=.18);axis.spines[['top','right']].set_visible(False)
        axis.set_xlim(0,4000)
    for axis in axes[-1]:axis.set_xlabel('New updates after the 18K parent checkpoint')
    axes[0,0].legend(loc='upper right',frameon=False,fontsize=9)
    fig.suptitle('Unified 15-new + 40-old policy: first 4000 updates',fontsize=15)
    fig.supxlabel('Training metrics include the XY reward variant; physical acceptance is evaluated separately.',fontsize=9)
    output=ROOT/'figures';output.mkdir(exist_ok=True);files=[]
    for suffix in ['png','svg','pdf']:
        target=output/f'stage1-training-curves.{suffix}';fig.savefig(target,dpi=180)
        files.append({'path':str(target.relative_to(ROOT)).replace('\\','/'),'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'bytes':target.stat().st_size})
    plt.close(fig)
    result={'source':str(source.relative_to(ROOT)).replace('\\','/'),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'rows':len(rows),'updates_min':int(x[0]),'updates_max':int(x[-1]),'warmup_policy_updates_not_applicable':800,
            'all_original_rows_preserved':True,'acceptance_inferred_from_training_metrics':False,'files':files}
    (ROOT/'evidence/training-plot-receipt.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
