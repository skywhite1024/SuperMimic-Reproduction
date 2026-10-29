"""Plot all fixed new-clip denominators; no filtering by success or backend."""
import csv,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def main():
    manifest=json.loads((ROOT/'library-manifest.json').read_text())
    names=[Path(row['path']).stem for row in manifest['files']]
    datasets=[('Stage1 raw CUDA','unified4000-cuda-all55','#7699b6'),
              ('Stage2 raw CUDA','unified-stage2-raw-cuda-new15','#e0a06c'),
              ('Stage2 calibrated CUDA','unified-stage2-corrected-cuda-all55','#4a8e78'),
              ('Stage2 calibrated CPU','unified-stage2-corrected-cpu-all55','#866aa5')]
    if (ROOT/'runs/unified-stage3-corrected-cuda-all55/summary.json').exists():
        datasets.append(('Stage3 calibrated CUDA','unified-stage3-corrected-cuda-all55','#bc5364'))
    fig,axis=plt.subplots(figsize=(11,11),layout='constrained')
    rows=[];sources=[];width=.78/len(datasets)
    for i,(label,run,color) in enumerate(datasets):
        path=ROOT/f'runs/{run}/summary.json';summary=json.loads(path.read_text());lookup={row['clip']:row for row in summary['clips']}
        values=[lookup[name]['norm_return_original'] for name in names]
        y=np.arange(15)+(i-(len(datasets)-1)/2)*width
        axis.barh(y,values,height=width*.9,color=color,label=label)
        failed=np.array([not lookup[name]['passed_original_hard_gate'] for name in names])
        axis.scatter(np.array(values)[failed],y[failed],marker='x',color='#b52424',s=40,zorder=5)
        rows.extend({'run':run,'clip':name,'original_reward_score':lookup[name]['norm_return_original'],
                     'original_hard_gate_passed':lookup[name]['passed_original_hard_gate']} for name in names)
        sources.append({'file':str(path.relative_to(ROOT)).replace('\\','/'),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    axis.axvline(.65,color='#535353',ls='--',lw=1,label='Fixed score threshold (.65)')
    axis.set_yticks(np.arange(15),[name.replace('_stageii_unitree_g1','') for name in names]);axis.invert_yaxis()
    axis.set_xlim(-.12,1.06);axis.set_xlabel('Original reward / full-clip denominator; score alone is not strict acceptance')
    axis.set_title('All15 new motions: raw/calibrated reference and backend differences')
    axis.grid(axis='x',alpha=.15);axis.spines[['top','right']].set_visible(False)
    axis.legend(loc='lower right',fontsize=8,framealpha=.95)
    fig.supxlabel('Red x: original hard-gate failure. One fixed seed per assessment; no uncertainty estimate.',fontsize=9)
    out=ROOT/'figures';out.mkdir(exist_ok=True)
    for ext in ['png','svg']:fig.savefig(out/f'new-library-stage-comparison.{ext}',dpi=160)
    plt.close(fig)
    with (out/'new-library-stage-comparison.csv').open('w',encoding='utf8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (ROOT/'evidence/stage-comparison-plot.json').write_text(json.dumps({'status':'plotted_full_fixed_denominator','new_denominator':15,'rows':len(rows),
        'sources':sources,'hard_gate_only_and_score_only_do_not_imply_strict_acceptance':True},indent=2)+'\n')
    print(json.dumps({'rows':len(rows),'clips':15,'datasets':len(datasets)}))


if __name__=='__main__':main()
