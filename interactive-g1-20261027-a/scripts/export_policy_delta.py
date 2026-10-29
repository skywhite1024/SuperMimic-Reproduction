"""Export only the learned 20 decoder columns, with a strict legacy-weight identity check."""
import hashlib,json
import torch
from runtime import ROOT
from global_model import expand_decoder


def state_hash(state):
    h=hashlib.sha256()
    for key,value in sorted(state.items()):h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    original=torch.load(ROOT/'runs/global-position/initial-full.pt',map_location='cpu',weights_only=True)
    trained=torch.load(ROOT/'runs/global-adapter-600/final-full.pt',map_location='cpu',weights_only=True)
    before,after=original['pi'],trained['pi'];hist=trained['args']['hist'];z=trained['args']['z_dim']
    columns=torch.tensor([z+i*101+j for i in range(hist) for j in [99,100]])
    mask=torch.ones(after['dec.0.weight'].shape[1],dtype=torch.bool);mask[columns]=False
    for key in before:
        if key=='dec.0.weight':assert torch.equal(before[key][:,mask],after[key][:,mask])
        else:assert torch.equal(before[key],after[key]),key
    payload={'decoder_xy_columns':after['dec.0.weight'][:,columns].clone(),'columns':columns,
             'args':trained['args'],'policy_state_sha256':state_hash(after),
             'parent_full_sha256':'0afcca9af97bd9db1f2dd083875898f962acab0b0a70c7313b221e575b70b05b',
             'trained_full_sha256':hashlib.sha256((ROOT/'runs/global-adapter-600/final-full.pt').read_bytes()).hexdigest(),
             'scope':'Exact inference policy reconstruction only; does not reconstruct changed WM normalizers or optimizer history.'}
    out=ROOT/'deployment';out.mkdir(exist_ok=True);target=out/'global-adapter-600-delta.pt'
    assert not target.exists(),'Preserve the existing verified export'
    torch.save(payload,target)
    receipt={k:v for k,v in payload.items() if not isinstance(v,torch.Tensor) and k!='args'}
    receipt.update(bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),legacy_policy_tensors_unchanged=True,changed_columns=columns.tolist())
    (out/'delta-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))


if __name__=='__main__':main()
