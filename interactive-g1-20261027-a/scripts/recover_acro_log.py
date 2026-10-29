"""Recover only this job's explicitly logged result, checking every trajectory byte."""
import argparse
import base64
import hashlib
import json
import re
import numpy as np
from runtime import ROOT


def messages(path):
    text=path.read_text(encoding='utf-8-sig')
    result=[]
    for line in text.splitlines():
        try:value=json.loads(line)
        except json.JSONDecodeError:result.append(line);continue
        if isinstance(value,str):result.append(value)
        elif isinstance(value,dict):
            # Official JSONL export represents each visible log row as Message.
            found=next((value[k] for k in ['Message','message','Body','body','Log','log'] if k in value),None)
            if isinstance(found,str):result.append(found)
            else:result.extend(v for v in value.values() if isinstance(v,str) and 'ACRO_' in v)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('log',type=__import__('pathlib').Path);a=p.parse_args()
    rows=messages(a.log);chunks={};count=None;receipt=None;proof=None
    for row in rows:
        match=re.search(r'ACRO_TRAJECTORY_CHUNK (\d+)/(\d+) ([A-Za-z0-9+/=]+)',row)
        if match:
            i,n,part=match.groups();i=int(i);n=int(n)
            assert count in (None,n);count=n
            assert i not in chunks or chunks[i]==part;chunks[i]=part
        for marker,kind in [('ACRO_GPU_RESULT_RECEIPT ','receipt'),('ACRO_H100_TRAINING_PROOF ','proof')]:
            if marker in row:
                item=json.JSONDecoder().raw_decode(row.split(marker,1)[1])[0]
                if kind=='receipt':assert receipt in (None,item);receipt=item
                else:assert proof in (None,item);proof=item
    assert receipt is not None and proof is not None,'Complete result and training proof required'
    assert count==receipt['chunks'] and set(chunks)==set(range(1,count+1)),'Incomplete trajectory chunks'
    raw=base64.b64decode(''.join(chunks[i] for i in range(1,count+1)),validate=True)
    assert len(raw)==receipt['trajectory_bytes'] and hashlib.sha256(raw).hexdigest()==receipt['trajectory_sha256']
    out=ROOT/'runs/acro-h100-2000-retest';out.mkdir(exist_ok=True)
    files={'trajectory.npz':raw,'summary.json':(json.dumps(receipt['summary'],indent=2)+'\n').encode()}
    for name,data in files.items():
        target=out/name
        assert not target.exists() or target.read_bytes()==data,'Preserve conflicting existing evidence'
        target.write_bytes(data)
    with np.load(out/'trajectory.npz') as traj:
        assert traj['state'].shape[1]==71 and traj['reference_qpos'].shape[1]==41
        assert all(np.isfinite(traj[k]).all() for k in traj.files)
        assert int(traj['hard_failure'].sum())==receipt['summary']['hard_failure_frames']
    (ROOT/'evidence/acro-gpu-result-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (ROOT/'evidence/acro-h100-training-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    print('ACRO_LOG_RECOVERY_VERIFIED '+json.dumps(receipt))


if __name__=='__main__':main()
