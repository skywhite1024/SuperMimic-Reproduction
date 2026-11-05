"""Verify every archive member before importing only this trial's new result paths."""
import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('archive', type=Path)
    args = parser.parse_args()
    folder = ROOT/'downloaded/specialist8807-1031d'
    result = json.loads((folder/'recovered.json').read_text())
    expected = result['archive']
    assert args.archive.stat().st_size == expected['bytes']
    assert digest(args.archive) == expected['sha256']
    extracted = folder/'extracted'
    assert not extracted.exists(), 'Inspect existing extraction before any retry'
    with tarfile.open(args.archive, 'r:gz') as archive:
        members = archive.getmembers()
        names = [item.name for item in members]
        assert len(set(names)) == len(names)
        for item in members:
            name = PurePosixPath(item.name)
            assert item.isfile() and not name.is_absolute() and '..' not in name.parts
            assert (extracted/item.name).resolve().is_relative_to(extracted.resolve())
        archive.extractall(extracted, filter='data')
    manifest_name = 'evidence/specialist8807-1031d-results-manifest.json'
    manifest = json.loads((extracted/manifest_name).read_text())
    assert set(names) == set(manifest) | {manifest_name}
    for name, row in manifest.items():
        path = extracted/name
        assert path.stat().st_size == row['bytes'], name
        assert digest(path) == row['sha256'], name
    plan = extracted/'specialist8807-1031d-plan.json'
    assert digest(plan) == '80a2ae20d00b403a873f973bd6bc5cd056e49c725c672631b428f33ea3cc9f4a'
    assert plan.read_bytes() == (ROOT/plan.name).read_bytes()
    frozen = json.loads(plan.read_text())['files']
    for name, expected_sha in frozen.items():
        assert digest(extracted/name) == expected_sha == digest(ROOT/name), name
    imported = []
    for name in sorted(manifest):
        if not (name.startswith('runs/specialist8807-1031d-') or
                name.startswith('runs/specialist8807-h100-1000-1031d/') or
                name == 'policy/specialist8807-h100-1000-1031d.pt'):
            continue
        source = extracted/name
        destination = ROOT/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            assert digest(destination) == manifest[name]['sha256'], name
        else:
            shutil.copy2(source, destination)
        imported.append(name)
    import torch
    full = torch.load(ROOT/'runs/specialist8807-h100-1000-1031d/final-full.pt',
                      map_location='cpu', weights_only=True)
    exported = torch.load(ROOT/'policy/specialist8807-h100-1000-1031d.pt',
                          map_location='cpu', weights_only=True)
    assert full['iteration'] == exported['iteration'] == 17000
    assert exported['pi']['obs_norm.mean'].numel() == 99
    assert full['variant'] == exported['variant']
    assert set(full['pi']) == set(exported['pi'])
    assert all(torch.equal(value, exported['pi'][name]) for name, value in full['pi'].items())
    verification = {'status': 'archive_all_members_and_full_export_readback_verified',
        'archive': str(args.archive.resolve()), 'bytes': expected['bytes'],
        'sha256': expected['sha256'], 'manifest_files': len(manifest),
        'archive_members': len(members), 'frozen_files_verified': len(frozen),
        'imported_result_paths': imported, 'final_iteration': 17000,
        'observation_dim': 99, 'full_and_exported_PI_tensors_equal': True,
        'no_existing_different_file_overwritten': True, 'full_goal_complete': False}
    (ROOT/'evidence/specialist8807-1031d-local-transfer-verification.json').write_text(
        json.dumps(verification, indent=2)+'\n')
    print(json.dumps(verification), flush=True)


if __name__ == '__main__':
    main()
