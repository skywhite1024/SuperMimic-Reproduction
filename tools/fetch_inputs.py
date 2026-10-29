"""Fetch pinned motion inputs from their original sources; do not bundle raw datasets in Git."""
import argparse,hashlib,json,os,subprocess,sys,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',action='store_true');p.add_argument('--new-library',action='store_true');a=p.parse_args()
    if not (a.baseline or a.new_library):p.error('Select an input library')
    if a.baseline:subprocess.run([sys.executable,str(ROOT/'implementation/prepare_assets.py')],check=True)
    if a.new_library:
        directory=ROOT/'acro-library-20261027-b';manifest=json.loads((directory/'library-manifest.json').read_text())
        for row in manifest['files']:
            dest=directory/row['path'];assert dest.resolve().is_relative_to(directory.resolve())
            if dest.exists():
                assert hashlib.sha256(dest.read_bytes()).hexdigest()==row['sha256'];continue
            # Fetch the raw file at the recorded commit, never a moving default branch.
            url=row['source_url'].replace('https://github.com/','https://raw.githubusercontent.com/').replace('/blob/','/')
            dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_suffix('.download')
            with urllib.request.urlopen(url,timeout=45) as r:tmp.write_bytes(r.read())
            assert hashlib.sha256(tmp.read_bytes()).hexdigest()==row['sha256'],'Pinned input hash mismatch'
            os.replace(tmp,dest)
        print('Verified all 15 pinned additional motions')
if __name__=='__main__':main()
