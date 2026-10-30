"""Safe extraction and independent all-file verification before publishing new results."""
import argparse,hashlib,json,shutil,tarfile
from datetime import datetime,timezone
from pathlib import Path,PurePosixPath
ROOT=Path(__file__).resolve().parents[1]
def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--attempt',choices=['attempt-a','resume-b'],required=True);args=parser.parse_args()
    label='retention-v1-1000-'+args.attempt
    folder=ROOT/'downloaded'/label;archive=folder/f'{label}-results.tar.gz'
    receipt=json.loads((folder/f'archive-receipt-{label}.json').read_text(encoding='utf8'))
    assert archive.stat().st_size==receipt['bytes'] and sha(archive)==receipt['sha256']
    destination=folder/'extracted';assert not destination.exists();destination.mkdir()
    with tarfile.open(archive,'r:gz') as bundle:
        for member in bundle:
            parts=PurePosixPath(member.name).parts
            assert member.isfile() and not member.issym() and not member.islnk()
            assert parts and not member.name.startswith('/') and '..' not in parts and ':' not in member.name and '\\' not in member.name
            target=destination.joinpath(*parts);assert target.resolve().is_relative_to(destination.resolve()) and not target.exists()
            target.parent.mkdir(parents=True,exist_ok=True)
            with bundle.extractfile(member) as source,target.open('xb') as output:shutil.copyfileobj(source,output)
    manifest=json.loads((destination/f'evidence/results-manifest-{label}.json').read_text())
    assert manifest['file_count']==len(manifest['files'])
    listed=set()
    for entry in manifest['files']:
        name=entry['path'];assert name not in listed;listed.add(name)
        target=destination/name;assert target.resolve().is_relative_to(destination.resolve())
        assert target.stat().st_size==entry['bytes'] and sha(target)==entry['sha256'],name
    assert len(list(destination.rglob('*')))>len(listed)
    published=[]
    for entry in manifest['files']:
        name=entry['path']
        if name.startswith(('runs/','policy/','evidence/')):
            source=destination/name;target=ROOT/name;target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():assert sha(target)==entry['sha256'],f'Preserve differing existing evidence: {name}'
            else:shutil.copyfile(source,target)
            published.append(name)
        elif (ROOT/name).exists():assert sha(ROOT/name)==entry['sha256'],f'Original source/config changed: {name}'
    result={'status':'archive_and_all_manifest_files_verified','verified_at':datetime.now(timezone.utc).isoformat(),
            'archive_bytes':receipt['bytes'],'archive_sha256':receipt['sha256'],'file_count':manifest['file_count'],
            'program_complete':manifest['program_complete'],'platform_status_from_attended_controller':manifest['platform_status_from_attended_controller'],
            'cloud_source_preserved':True,'published_run_policy_evidence':published,'archive':str(archive.relative_to(ROOT)),
            'resource_release_requires_separate_actual_platform_check':True}
    target=ROOT/f'evidence/local-transfer-verification-{label}.json';assert not target.exists();target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:value for key,value in result.items() if key!='published_run_policy_evidence'}),flush=True)

if __name__=='__main__':main()
