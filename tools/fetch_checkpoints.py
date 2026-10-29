"""Retrieve only explicitly selected named release artifacts and verify every byte."""
import argparse,hashlib,json,os,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--name');p.add_argument('--list',action='store_true');a=p.parse_args()
    rows=json.loads((ROOT/'artifacts/manifest.json').read_text())['checkpoints']
    if a.list:
        print(json.dumps([{k:r[k] for k in ['name','kind','bytes','path']} for r in rows],indent=2));return
    if not a.name:p.error('Select --name or --list')
    row=next((r for r in rows if r['name']==a.name),None)
    if row is None:p.error('Unknown checkpoint name')
    dest=ROOT/row['path'];assert dest.resolve().is_relative_to(ROOT.resolve())
    if dest.exists():
        if hashlib.file_digest(dest.open('rb'),'sha256').hexdigest()!=row['sha256']:raise FileExistsError('Conflicting checkpoint; preserve it and choose a clean checkout')
        print('Verified existing checkpoint:',row['path']);return
    dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_suffix('.download')
    with urllib.request.urlopen(row['url'],timeout=60) as response,tmp.open('wb') as out:
        while chunk:=response.read(1024**2):out.write(chunk)
    assert tmp.stat().st_size==row['bytes'] and hashlib.file_digest(tmp.open('rb'),'sha256').hexdigest()==row['sha256'],'Download hash mismatch; original destination unchanged'
    os.replace(tmp,dest);print('Verified downloaded checkpoint:',row['path'])
if __name__=='__main__':main()
