"""Finite collection of the original results plus operational recovery provenance."""
import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--platform-status', choices=['success', 'failed', 'stopped'], required=True)
    args = parser.parse_args()
    assert ROOT.resolve().is_relative_to(Path('/mnt/afs/USER_ID'))
    operation = json.loads((ROOT / 'retention-resume-1030b-plan.json').read_text())
    for name, digest in operation['files'].items():
        assert sha(ROOT / name) == digest, name
    label = 'retention-v1-1000-resume-b'
    original_archive = ROOT / f'deployment/{label}-results.tar.gz'
    if original_archive.exists():
        receipt = json.loads((ROOT / f'evidence/archive-receipt-{label}.json').read_text())
        assert original_archive.stat().st_size == receipt['bytes'] and sha(original_archive) == receipt['sha256']
        manifest = json.loads((ROOT / f'evidence/results-manifest-{label}.json').read_text())
        assert manifest['platform_status_from_attended_controller'] == args.platform_status
        for row in manifest['files']:
            assert sha(ROOT / row['path']) == row['sha256'], row['path']
        print('REUSE_VERIFIED_EXISTING_RESULTS_ARCHIVE', flush=True)
    else:
        subprocess.run([sys.executable, '-u', str(ROOT / 'scripts/collect_retention_trial.py'),
                        '--platform-status', args.platform_status, '--attempt', 'resume-b'], check=True, timeout=300)
    names = ['retention-resume-1030b-plan.json', 'retention-recovery-1030-plan.json',
             'scripts/run_retention_resume1030b.py', 'scripts/collect_retention_resume1030b.py',
             'evidence/gpu-retention-v1-1000.json', 'evidence/retention-1030-dependency-preflight.json',
             'evidence/retention-1030-cpu-recovery-console.txt', 'evidence/retention-1030-cpu-recovery-console-v2.txt']
    files = [{'path': name, 'bytes': (ROOT / name).stat().st_size, 'sha256': sha(ROOT / name)} for name in names]
    manifest_name = 'evidence/retention-resume1030b-provenance-manifest.json'
    manifest = ROOT / manifest_name
    assert not manifest.exists()
    manifest.write_text(json.dumps({'files': files, 'original_failed_receipt_preserved': True,
                                    'credentials_or_reference_CSVs_included': False}, indent=2) + '\n')
    archive = ROOT / 'deployment/retention-resume1030b-provenance.tar.gz'
    assert not archive.exists()
    with tarfile.open(archive, 'w:gz', compresslevel=3) as bundle:
        for name in names + [manifest_name]:
            path = ROOT / name
            assert path.resolve().is_relative_to(ROOT) and path.is_file()
            bundle.add(path, arcname=name, recursive=False)
    result = {'archive': archive.name, 'bytes': archive.stat().st_size, 'sha256': sha(archive),
              'file_count': len(files), 'platform_status_from_attended_controller': args.platform_status}
    (ROOT / 'evidence/retention-resume1030b-provenance-receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    print('RETENTION_RESUME_PROVENANCE_ARCHIVE_READY ' + json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
