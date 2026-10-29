"""Verify cloud archive and every manifest entry before publishing run files locally."""
import hashlib,json,shutil,tarfile
from datetime import datetime,timezone
from pathlib import Path,PurePosixPath

ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    folder=ROOT/'downloaded/unified4000';archive=folder/'unified-h100-4000-results.tar.gz'
    receipt=json.loads((folder/'archive-receipt-4000.json').read_text(encoding='utf8'))
    assert archive.stat().st_size==receipt['bytes'] and sha(archive)==receipt['sha256']
    destination=folder/'extracted';assert not destination.exists();destination.mkdir()
    with tarfile.open(archive,'r:gz') as bundle:
        for member in bundle:
            parts=PurePosixPath(member.name).parts
            assert member.isfile() and not member.issym() and not member.islnk()
            assert parts and not member.name.startswith('/') and '..' not in parts and ':' not in member.name
            target=destination.joinpath(*parts);assert target.resolve().is_relative_to(destination.resolve()) and not target.exists()
            target.parent.mkdir(parents=True,exist_ok=True)
            with bundle.extractfile(member) as source,target.open('xb') as output:shutil.copyfileobj(source,output)
    manifest=json.loads((destination/'evidence/results-manifest-4000.json').read_text(encoding='utf8'))
    assert manifest['file_count']==len(manifest['files'])
    for entry in manifest['files']:
        target=destination/entry['path'];assert target.stat().st_size==entry['bytes'] and sha(target)==entry['sha256'],entry['path']
    published=[]
    for entry in manifest['files']:
        name=entry['path']
        if name.startswith(('runs/','policy/','evidence/')):
            source=destination/name;target=ROOT/name;target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():assert sha(target)==entry['sha256'],f'Existing local artifact differs: {name}'
            else:shutil.copyfile(source,target)
            published.append(name)
    result={'status':'archive_and_all_manifest_files_verified','verified_at':datetime.now(timezone.utc).isoformat(),
            'archive_bytes':receipt['bytes'],'archive_sha256':receipt['sha256'],'file_count':manifest['file_count'],
            'cloud_source_preserved':True,'published_run_policy_evidence':published,'archive':str(archive.relative_to(ROOT)),
            'manual_resource_release_verified_separately':True}
    (ROOT/'evidence/local-transfer-verification-4000.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps({key:value for key,value in result.items() if key!='published_run_policy_evidence'}),flush=True)

if __name__=='__main__':main()
