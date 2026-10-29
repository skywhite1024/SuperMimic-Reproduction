"""Bounded third stage: balanced replay continuation plus fixed separate assessments."""
import hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    started=time.monotonic();plan=json.loads((ROOT/'stage3-plan.json').read_text())
    assert plan['status']=='frozen_before_engineering_and_stage3_evaluation'
    first=json.loads((ROOT/'evidence/gpu-stage2-4000.json').read_text());assert first['status']=='updates_corrected55_and_raw15_evaluation_complete'
    initial=ROOT/'runs/unified-h100-stage2-4000/final-full.pt';assert sha(initial)==plan['initial_full_sha256']==first['full_checkpoint_sha256']
    for name,expected in plan['files'].items():assert sha(ROOT/name)==expected,name
    out=ROOT/'runs/unified-h100-stage3-4000';assert not (out/'args.json').exists(),'Inspect the existing stage before retrying'
    import torch
    assert torch.cuda.device_count()==1 and 'H100' in torch.cuda.get_device_name(0)
    config={'gpu_count':1,'gpu_name':torch.cuda.get_device_name(0),'cpu_max':Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
            'memory_max':Path('/sys/fs/cgroup/memory.max').read_text().strip(),'new_updates':4000,'wm_only_updates':400,'num_envs':1024,
            'maximum_program_seconds':7200,'reference_variant':'ground-calibration-v1','new_denominator':15,'old_denominator':40,
            'thresholds_unchanged':True,'one_policy':True,'actual_replay_group_mass':{'new':.6,'lafan':.4},'old_failure_replay_priorities':plan['old_failure_replay_priorities']}
    path=ROOT/'evidence/gpu-stage3-4000.json';receipt={'status':'running','config':config,'initial_full_sha256':sha(initial),'plan_sha256':sha(ROOT/'stage3-plan.json')}
    path.write_text(json.dumps(receipt,indent=2)+'\n');print('UNIFIED_STAGE3_START '+json.dumps(receipt),flush=True)
    def run(script,args,limit):
        remaining=6900-(time.monotonic()-started);assert remaining>0
        subprocess.run([sys.executable,'-u',str(ROOT/'scripts'/script),*args],check=True,timeout=min(limit,remaining))
    try:
        run('train_replay_stage3.py',['--initial',str(initial),'--label','unified-h100-stage3-4000','--device','cuda','--steps','4000','--wm-warmup','400','--num-envs','1024'],5700)
        proof=json.loads((out/'proof.json').read_text());assert proof['new_updates']==4000 and proof['normalizer_tensors_unchanged']
        print('UNIFIED_STAGE3_UPDATE_PROOF '+json.dumps(proof),flush=True)
        full=out/'final-full.pt'
        run('evaluate_ground_variant.py',['--checkpoint',str(full),'--mode','unified-stage3-corrected-cuda-all55','--group','all','--device','cuda'],1200)
        corrected=json.loads((ROOT/'runs/unified-stage3-corrected-cuda-all55/summary.json').read_text())
        run('evaluate_library.py',['--checkpoint',str(full),'--mode','unified-stage3-raw-cuda-new15','--group','new','--device','cuda'],400)
        raw=json.loads((ROOT/'runs/unified-stage3-raw-cuda-new15/summary.json').read_text())
        receipt.update(status='updates_corrected55_and_raw15_evaluation_complete',wall_seconds=time.monotonic()-started,
                       full_checkpoint_sha256=sha(full),corrected=corrected['groups'],raw_new=raw['groups'],full_goal_complete=False,
                       physical_quality_and_continuous_assessment='pending_local_full_trajectory_review')
    except BaseException as error:
        receipt.update(status='failed_or_interrupted',error=repr(error),traceback=traceback.format_exc(),wall_seconds=time.monotonic()-started)
        path.write_text(json.dumps(receipt,indent=2)+'\n');raise
    path.write_text(json.dumps(receipt,indent=2)+'\n');print('UNIFIED_STAGE3_COMPLETE '+json.dumps(receipt),flush=True)

if __name__=='__main__':main()
