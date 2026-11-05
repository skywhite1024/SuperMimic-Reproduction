"""Attended, strictly pinned CCI transfer; never starts or releases cloud resources."""
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLOUD = '/mnt/afs/USER_ID/SuperMimic/acro-library-20261027-b'
HOST = 'root@203.0.113.1'


def main():
    window = json.loads((ROOT / 'evidence/cci-retention-results-1030-retrieval-window.json').read_text())
    assert time.time() < window['deadline_epoch'] - 600, 'Reserve ten minutes for actual UI closure'
    key = Path.home() / '.ssh/sensecore_cci_ed25519'
    known = ROOT / 'evidence/cci-retention-results-1030-known-hosts'
    assert key.is_file() and known.is_file()
    options = ['-i', str(key), '-o', 'StrictHostKeyChecking=yes', '-o', f'UserKnownHostsFile={known}',
               '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10']
    ssh = ['ssh', '-p', '10076', *options, HOST]
    scp = ['scp', '-P', '10076', *options]
    transcript = ROOT / 'evidence/retention-resume1030b-retrieval-console.txt'
    assert not transcript.exists(), 'Preserve previous transfer attempt'

    def run(args, *, data=None, timeout=300):
        result = subprocess.run(args, input=data, text=True, encoding='utf8', errors='replace',
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
        with transcript.open('a', encoding='utf8') as stream:
            stream.write(result.stdout + '\n')
        print(result.stdout[-4500:], flush=True)
        assert result.returncode == 0, f'Transfer operation failed with {result.returncode}'
        return result.stdout

    runtime = run([*ssh, 'python3 -'], data="""from pathlib import Path
assert Path('/sys/fs/cgroup/cpu.max').read_text().strip() == '200000 100000'
assert int(Path('/sys/fs/cgroup/memory.max').read_text()) == 8589934592
assert Path('/mnt/afs/USER_ID/SuperMimic/acro-library-20261027-b').is_dir()
print('STRICT_SSH_OWN_AFS_2CPU_8GIB_PASS')
""")
    bundle = ROOT / 'deployment/retention-resume1030b-collection-inputs.tar.gz'
    expected = json.loads((ROOT / 'evidence/retention-resume1030b-collection-inputs.json').read_text())
    assert bundle.stat().st_size == expected['bytes']
    assert hashlib.sha256(bundle.read_bytes()).hexdigest() == expected['sha256']
    incoming = f'{CLOUD}/deployment/retention-collection-inputs-1030-{int(time.time())}.tar.gz'
    run([*ssh, f'test ! -e {incoming}'])
    run([*scp, str(bundle), f'{HOST}:{incoming}'])
    payload = f"""import hashlib,tarfile
from pathlib import Path
root=Path({CLOUD!r})
archive=Path({incoming!r})
assert archive.resolve().is_relative_to(root)
assert hashlib.sha256(archive.read_bytes()).hexdigest()=={expected['sha256']!r}
with tarfile.open(archive,'r:gz') as bundle:
    members=bundle.getmembers()
    assert len(members)==1 and members[0].isfile()
    member=members[0]
    assert member.name=='scripts/collect_retention_resume1030b.py'
    data=bundle.extractfile(member).read()
    assert hashlib.sha256(data).hexdigest()=={expected['files']['scripts/collect_retention_resume1030b.py']!r}
    target=root/member.name
    assert target.resolve().is_relative_to(root)
    if target.exists(): assert target.read_bytes()==data
    else:
        with target.open('xb') as stream: stream.write(data)
print('COLLECTION_INPUT_ARCHIVE_AND_SOURCE_HASH_PASS')
"""
    run([*ssh, 'python3 -'], data=payload)
    run([*ssh, f'/mnt/afs/USER_ID/SuperMimic/lafan1-20261024/uni/.venv/bin/python -u {CLOUD}/scripts/collect_retention_resume1030b.py --platform-status success'], timeout=600)
    folder = ROOT / 'downloaded/retention-v1-1000-resume-b'
    folder.mkdir(exist_ok=True)
    files = ['deployment/retention-v1-1000-resume-b-results.tar.gz',
             'evidence/archive-receipt-retention-v1-1000-resume-b.json',
             'deployment/retention-resume1030b-provenance.tar.gz',
             'evidence/retention-resume1030b-provenance-receipt.json']
    for name in files:
        assert time.time() < window['deadline_epoch'] - 480, 'Reserve eight minutes for UI closure'
        target = folder / Path(name).name
        assert not target.exists(), 'Do not overwrite existing download'
        run([*scp, f'{HOST}:{CLOUD}/{name}', str(target)], timeout=300)
    receipt = {'status': 'four_result_files_downloaded_pending_independent_validation',
               'at': datetime.now(timezone.utc).isoformat(), 'private_key_body_read': False,
               'strict_host_key_checking': True, 'cpu_cores': 2, 'memory_gib': 8, 'GPU': 0,
               'files': [{'path': str((folder / Path(name).name).relative_to(ROOT)),
                          'bytes': (folder / Path(name).name).stat().st_size} for name in files],
               'manual_stop_empty_unbind_restore_still_required': True}
    (ROOT / 'evidence/retention-resume1030b-download-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
