"""Archive only this finite trial's result artifacts; no credentials or raw data CSVs."""
import hashlib,json,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def main():
    selected=set()
    for directory in (ROOT/'runs').glob('specialist8807-1031d-*'):
        selected.update(p for p in directory.rglob('*') if p.is_file())
    training=ROOT/'runs/specialist8807-h100-1000-1031d'
    selected.update(p for p in training.rglob('*') if p.is_file() and (p.suffix in ['.json','.csv','.log'] or p.name=='final-full.pt'))
    selected.add(ROOT/'policy/specialist8807-h100-1000-1031d.pt')
    plan=ROOT/'specialist8807-1031d-plan.json';selected.add(plan)
    for name in json.loads(plan.read_text())['files']:selected.add(ROOT/name)
    receipt=ROOT/'evidence/specialist8807-1031d-execution.json';selected.add(receipt)
    selected.update(p for p in (ROOT/'deployment/specialist8807-1031d-work').glob('*.log') if p.name!='package-results.log')
    assert all(p.resolve().is_relative_to(ROOT.resolve()) and p.is_file() for p in selected)
    manifest={p.relative_to(ROOT).as_posix():{'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(selected)}
    manifest_path=ROOT/'evidence/specialist8807-1031d-results-manifest.json';assert not manifest_path.exists();manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    archive=ROOT/'deployment/specialist8807-1031d-results.tar.gz';assert not archive.exists()
    with tarfile.open(archive,'w:gz',compresslevel=1) as target:
        for p in sorted(selected):target.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False)
        target.add(manifest_path,arcname=manifest_path.relative_to(ROOT).as_posix(),recursive=False)
    print('SPECIALIST8807_ARCHIVE '+json.dumps({'path':str(archive),'bytes':archive.stat().st_size,'sha256':digest(archive),'files':len(manifest),'raw_dataset_CSV_included':False}),flush=True)
if __name__=='__main__':main()
