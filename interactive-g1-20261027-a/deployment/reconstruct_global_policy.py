"""Reconstruct the exact accepted inference policy from the pinned baseline and small delta."""
import hashlib,json,os,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from global_model import expand_decoder


def state_hash(state):
    h=hashlib.sha256()
    for key,value in sorted(state.items()):h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    delta=torch.load(ROOT/'deployment/global-adapter-600-delta.pt',map_location='cpu',weights_only=True)
    full=Path(os.environ.get('SUPERMIMIC_FULL',str(ROOT.parent/'implementation/continuation-20261025-a/downloaded/policy/full_016000.pt')))
    assert hashlib.sha256(full.read_bytes()).hexdigest()==delta['parent_full_sha256']
    parent=torch.load(full,map_location='cpu',weights_only=True);pi=parent['pi']
    pi['dec.0.weight']=expand_decoder(pi['dec.0.weight'],delta['args']['hist'],delta['args']['z_dim'])
    pi['obs_norm.mean']=torch.cat((pi['obs_norm.mean'],torch.zeros(2)))
    pi['obs_norm.var']=torch.cat((pi['obs_norm.var'],torch.ones(2)))
    pi['dec.0.weight'][:,delta['columns']]=delta['decoder_xy_columns']
    assert state_hash(pi)==delta['policy_state_sha256']
    out=ROOT/'policy';out.mkdir(exist_ok=True);p=out/'global-adapter-600-deployed.pt'
    if p.exists():
        old=torch.load(p,map_location='cpu',weights_only=True);assert state_hash(old['pi'])==delta['policy_state_sha256']
    else:torch.save({'pi':pi,'args':delta['args'],'variant':{'name':'global-position-101d-inference','canonical_pi_sha256':delta['policy_state_sha256']}},p)
    loaded=torch.load(p,map_location='cpu',weights_only=True)
    assert all(torch.equal(value,loaded['pi'][key]) for key,value in pi.items())
    receipt={'status':'exact_policy_reconstruction_passed','policy':str(p),'policy_state_sha256':state_hash(loaded['pi']),
             'policy_bytes':p.stat().st_size,'delta_bytes':(ROOT/'deployment/global-adapter-600-delta.pt').stat().st_size,'optimizer_or_WM_reconstruction_claimed':False}
    (out/'reconstruction-proof.json').write_text(json.dumps(receipt,indent=2)+'\n');print('GLOBAL_POLICY_RECONSTRUCTION_PASS '+json.dumps(receipt))


if __name__=='__main__':main()
