"""Freeze the public G1 library before continuation; retain the complete denominator."""
import concurrent.futures
import base64
import datetime
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
REPO='lez666/chunwan-flip'
API='https://api.github.com/repos/'+REPO
OLD=ROOT.parent/'interactive-g1-20261027-a/assets/acro-candidates'


def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'G1-acro-reproduction-audit','Accept':'application/vnd.github+json'}),timeout=40) as r:return r.read()


def main():
    manifest=ROOT/'library-manifest.json';assert not manifest.exists(),'Preserve the frozen library version'
    docs=ROOT/'sources';docs.mkdir(parents=True,exist_ok=True)
    raw=(docs/'commit.json').read_bytes() if (docs/'commit.json').exists() else fetch(API+'/commits/main')
    commit=json.loads(raw)['sha'];(docs/'commit.json').write_bytes(raw)
    tree_raw=(docs/'tree.json').read_bytes() if (docs/'tree.json').exists() else fetch(API+'/git/trees/'+commit+'?recursive=1')
    tree=json.loads(tree_raw);assert not tree.get('truncated')
    (docs/'tree.json').write_bytes(tree_raw)
    files=sorted([x for x in tree['tree'] if x['type']=='blob' and x['path'].startswith('data/') and x['path'].endswith('.csv')],key=lambda x:x['path'])
    assert files,'No public CSV references found'
    def blob_bytes(row):
        item=json.loads(fetch(API+'/git/blobs/'+row['sha']));assert item['encoding']=='base64'
        return base64.b64decode(item['content'])
    def copy_one(row):
        name=Path(row['path']).name;target=ROOT/row['path'];target.parent.mkdir(exist_ok=True)
        old=OLD/name
        raw=old.read_bytes() if old.is_file() else blob_bytes(row)
        blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        if blob!=row['sha']:
            raw=blob_bytes(row)
            blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        assert blob==row['sha'] and len(raw)==row['size']
        assert not target.exists() or target.read_bytes()==raw;target.write_bytes(raw)
        return {'path':row['path'],'git_blob':blob,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),
                'training_status_before_phase':'trained_single_clip_specialist' if name=='88_08_stageii_unitree_g1.csv' else 'not_trained',
                'unified_policy_acceptance':'not_evaluated','single_clip_018000_acceptance':'passed_original_inversion_upright_gate' if name=='88_08_stageii_unitree_g1.csv' else 'not_evaluated',
                'semantics_status':'pending_primary_source_audit','source_url':'https://github.com/'+REPO+'/blob/'+commit+'/'+row['path']}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(copy_one,files))
    for row in tree['tree']:
        if row['path'].lower() in ['readme.md','license','license.md','license.txt']:
            raw=blob_bytes(row);(docs/Path(row['path']).name).write_bytes(raw)
    result={'frozen_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'repository':'https://github.com/'+REPO,'commit':commit,
            'csv_count_actual':len(rows),'readme_count_claim_reported':16,'readme_and_actual_count_discrepancy':len(rows)!=16,
            'source_contract_to_verify':{'fps':30,'columns':36,'root_position_units':'metres','up_axis':'Z','quaternion_order':'xyzw','joint_order':'Unitree G1 SDK 29 joints'},
            'BONES_route':'paused; no BONES files requested','files':rows,
            'denominator_policy':'All frozen CSVs retained in manifest, including unsuitable or failed references. No unrecorded filtering.'}
    manifest.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('PUBLIC_ACRO_LIBRARY_FROZEN '+json.dumps({'commit':commit,'csv_count':len(rows),'total_bytes':sum(x['bytes'] for x in rows),'already_trained':sum(x['training_status_before_phase']!='not_trained' for x in rows),'remaining':sum(x['training_status_before_phase']=='not_trained' for x in rows)}),flush=True)


if __name__=='__main__':main()
