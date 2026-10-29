"""CPU-only deserialization and identity audit before freezing the result archive."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import os
import sys

assert os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'Explicitly disable CUDA for this readback'
import torch

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from audit_common import file_hash,state_hash,write_json,lr_at


def snapshot_digest(mapping):
    digest=hashlib.sha256()
    for name,value in mapping.items():
        assert bool(torch.isfinite(value).all())
        digest.update(name.encode())
        digest.update(value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def main():
    torch.set_num_threads(2)
    destination=ROOT/'evidence/full-final-readback.json'
    assert not destination.exists(), 'Preserve an existing readback receipt'
    selection=json.loads((ROOT/'selection.json').read_text())
    iteration=selection['best']['iteration']
    full=ROOT/'policy'/f'full_{iteration:06d}.pt'
    inference=ROOT/'policy'/f'ckpt_{iteration:06d}.pt'
    saved=torch.load(full,map_location='cpu',weights_only=True)
    policy=torch.load(inference,map_location='cpu',weights_only=True)
    update=json.loads((ROOT/'policy'/f'update-state-{iteration:06d}.json').read_text())
    assert saved['iteration']==policy['iteration']==update['state']['iteration']==16000
    assert saved['sim_steps']==policy['sim_steps']==update['state']['sim_steps']
    assert file_hash(inference)==selection['best']['checkpoint_sha256']
    hashes={key:state_hash(saved[key]) for key in ['wm','pi','critic','opt_wm','opt_pi','opt_v']}
    for key,snapshot_key in [('wm','wm'),('pi','policy'),('critic','critic')]:
        assert snapshot_digest(saved[key])==update['state'][snapshot_key]['sha256_state']
    for key in ['wm','pi']:
        assert hashes[key]==state_hash(policy[key])
    counters={}
    for key in ['opt_wm','opt_pi','opt_v']:
        assert hashes[key]==update['optimizer_hashes'][key]
        assert all(abs(group['lr']-lr_at(iteration-1))<1e-15 for group in saved[key]['param_groups'])
        counters[key]=sorted({int(row['step']) for row in saved[key]['state'].values() if 'step' in row})
    assert counters=={'opt_wm':[160000],'opt_pi':[15500],'opt_v':[310000]}
    receipt={'passed':True,'verified_at':dt.datetime.now(dt.timezone.utc).isoformat(),
             'device':'cpu','GPU_operations_executed':False,'torch':torch.__version__,
             'iteration':iteration,'sim_steps':saved['sim_steps'],'full_checkpoint_sha256':file_hash(full),
             'inference_checkpoint_sha256':file_hash(inference),'state_hashes':hashes,
             'optimizer_steps':counters,'optimizer_moments_match_training_snapshot':True,
             'models_match_training_snapshot':True,'inference_models_match_full_checkpoint':True,
             'checkpoint_lr_matches_absolute_iteration':True,'all_state_tensors_finite':True}
    write_json(destination,receipt)
    print('FULL_FINAL_CPU_READBACK_PASS '+json.dumps({key:receipt[key] for key in ['iteration','sim_steps','full_checkpoint_sha256','optimizer_steps']}),flush=True)


if __name__=='__main__':main()
