"""Verify the cloud receipt and per-file hashes before accepting local results."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile


def sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--receipt',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    receipt=json.loads(args.receipt.read_text(encoding='utf-8'))
    assert args.archive.stat().st_size==receipt['bytes']
    assert sha256(args.archive)==receipt['sha256']
    destination=args.output.resolve()
    destination.mkdir(parents=True,exist_ok=True)
    assert not any(destination.iterdir()), 'Use an empty destination; no overwrites'
    verified=[]
    with tarfile.open(args.archive,'r:gz') as tar:
        members=tar.getmembers()
        names=[m.name for m in members]
        assert len(names)==len(set(names))==receipt['files']
        for member in members:
            name=PurePosixPath(member.name)
            assert member.isfile() and not name.is_absolute()
            assert '..' not in name.parts and '\\' not in member.name and ':' not in member.name
            assert (destination/member.name).resolve().is_relative_to(destination)
        manifest_file=tar.extractfile('result-manifest.json')
        assert manifest_file is not None
        manifest=json.load(manifest_file)
        assert set(names)==set(manifest)|{'result-manifest.json'}
        assert sum(row['bytes'] for row in manifest.values())==receipt['payload_bytes']
        for member in members:
            output=destination/member.name
            output.parent.mkdir(parents=True,exist_ok=True)
            source=tar.extractfile(member)
            assert source is not None
            digest=hashlib.sha256()
            size=0
            with output.open('xb') as stream:
                for chunk in iter(lambda:source.read(1024*1024),b''):
                    stream.write(chunk)
                    digest.update(chunk)
                    size+=len(chunk)
            assert size==member.size
            if member.name in manifest:
                assert manifest[member.name]=={'bytes':size,'sha256':digest.hexdigest()}
            verified.append({'name':member.name,'bytes':size,'sha256':digest.hexdigest()})
    record={'status':'archive_and_every_file_verified','archive':str(args.archive.resolve()),
            'archive_sha256':receipt['sha256'],'files':verified,'executable_code_run':False}
    (destination.parent/'local-transfer-verification.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':record['status'],'files':len(verified),'archive_bytes':receipt['bytes']}))


if __name__=='__main__':
    main()
