"""Compare original manager early-stop evaluation with retained full physical traces."""
import argparse,json,subprocess,sys,time
from pathlib import Path
import torch
from runtime import ROOT,make_env,references
from unisupermimic.tasks.motion_tracking.g1.supermimic.evaluate import evaluate


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--clips',nargs='+',required=True);p.add_argument('--label',required=True);a=p.parse_args()
    checkpoint=a.checkpoint.resolve();saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    dimension=saved['pi']['obs_norm.mean'].numel();torch.set_num_threads(2);rows=[]
    for name in a.clips:
        mode=f'{a.label}-{name}';torch.manual_seed(0)
        env=make_env([x for x in references('all') if x.stem==name],observation_dim=dimension,full_trace=False)
        try:original=evaluate(env,saved,device='cpu')['per_clip'][name]
        finally:env.close();env.backend.close()
        subprocess.run([sys.executable,str(ROOT/'scripts/evaluate_library.py'),'--checkpoint',str(checkpoint),
                        '--mode',mode,'--group','all','--clips',name],check=True)
        traced=json.loads((ROOT/'runs'/mode/'summary.json').read_text())['clips'][0]
        row={'clip':name,'dimension':dimension,'original':original,
             'full_trace':{key:traced[key] for key in ['norm_return_original','scored_steps','passed_original_hard_gate','post_initialization_resets']},
             'absolute_score_difference':abs(original['norm_return']-traced['norm_return_original'])}
        row['passed']=row['absolute_score_difference']<=1e-6 and original['steps']==traced['scored_steps'] and original['failed']==(not traced['passed_original_hard_gate']) and traced['post_initialization_resets']==0
        rows.append(row)
    result={'label':a.label,'checkpoint':str(checkpoint),'contract_version':2,'tolerance':1e-6,
            'gate':'native post-physics state at pre-action f+1','normalization':'original float32','cases':rows,'passed':all(x['passed'] for x in rows)}
    out=ROOT/'evidence'/f'{a.label}.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n')
    print('SCORE_CONTRACT '+json.dumps(result),flush=True);assert result['passed']


if __name__=='__main__':main()
