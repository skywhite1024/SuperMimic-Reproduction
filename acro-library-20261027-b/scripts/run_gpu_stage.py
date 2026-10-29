"""One bounded H100 stage, matched full-library evaluation before and after training."""
import hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    started=time.monotonic();ready=json.loads((ROOT/'evidence/preparation-ready.json').read_text())
    assert ready['status']=='cpu_preparation_passed' and ready['original_score_contract']['passed']
    initial=Path(ready['initial_101d_checkpoint']);assert hashlib.sha256(initial.read_bytes()).hexdigest()==ready['initial_sha256']
    out=ROOT/'runs/unified-h100-4000';assert not (out/'args.json').exists(),'Inspect existing stage before any repeated submission'
    import torch
    assert torch.cuda.device_count()==1 and 'H100' in torch.cuda.get_device_name(0)
    root_cgroup=Path('/sys/fs/cgroup')
    config={'gpu_count':torch.cuda.device_count(),'gpu_name':torch.cuda.get_device_name(0),'cpu_max':(root_cgroup/'cpu.max').read_text().strip(),
            'memory_max':(root_cgroup/'memory.max').read_text().strip(),'new_updates':4000,'wm_only_updates':800,'num_envs':1024,
            'maximum_program_seconds':7200,'target_policy_count':1,'new_denominator':15,'old_lafan_denominator':40,'BONES_route':'paused'}
    print('UNIFIED_H100_STAGE_START '+json.dumps(config),flush=True)
    receipt={'status':'running','config':config,'initial_sha256':ready['initial_sha256']}
    path=ROOT/'evidence/gpu-stage-4000.json';path.write_text(json.dumps(receipt,indent=2)+'\n')
    def run(script,args,limit):
        remaining=6900-(time.monotonic()-started);assert remaining>0,'Finite stage wall budget exhausted'
        subprocess.run([sys.executable,'-u',str(ROOT/'scripts'/script),*args],check=True,timeout=min(limit,remaining))
    try:
        run('evaluate_library.py',['--checkpoint',str(initial),'--mode','initial101d-cuda-all55','--group','all','--device','cuda'],1200)
        before=json.loads((ROOT/'runs/initial101d-cuda-all55/summary.json').read_text())
        print('UNIFIED_MATCHED_CUDA_BASELINE '+json.dumps(before['groups']),flush=True)
        run('train_unified.py',['--initial',str(initial),'--label','unified-h100-4000','--device','cuda','--steps','4000','--wm-warmup','800','--num-envs','1024'],5700)
        proof=json.loads((out/'proof.json').read_text());assert proof['new_updates']==4000 and proof['normalizer_tensors_unchanged']
        print('UNIFIED_H100_UPDATE_PROOF '+json.dumps(proof),flush=True)
        full=out/'final-full.pt'
        run('evaluate_library.py',['--checkpoint',str(full),'--mode','unified4000-cuda-all55','--group','all','--device','cuda'],1200)
        after=json.loads((ROOT/'runs/unified4000-cuda-all55/summary.json').read_text())
        receipt.update(status='training_and_full55_evaluation_complete',wall_seconds=time.monotonic()-started,
                       before=before['groups'],after=after['groups'],full_checkpoint_sha256=hashlib.sha256(full.read_bytes()).hexdigest(),
                       new_clips_all_learned_inferred_from_update_count=False,full_goal_complete=False)
    except BaseException as error:
        receipt.update(status='failed_or_interrupted',error=repr(error),traceback=traceback.format_exc(),wall_seconds=time.monotonic()-started)
        path.write_text(json.dumps(receipt,indent=2)+'\n');raise
    path.write_text(json.dumps(receipt,indent=2)+'\n');print('UNIFIED_H100_STAGE_COMPLETE '+json.dumps(receipt),flush=True)


if __name__=='__main__':main()
