"""No-dependency checks for a clean clone of the public reproduction source and reports."""
from pathlib import Path
import ast,hashlib,json,posixpath,re,sys
from html.parser import HTMLParser
from urllib.parse import unquote,urlsplit
ROOT=Path(__file__).resolve().parents[1]
SITE=ROOT/'docs/site'
errors=[];counts={'python':0,'json':0,'local_links':0,'videos':0}
keys={p.relative_to(SITE).as_posix() for p in SITE.rglob('*')}
class Links(HTMLParser):
    def handle_starttag(self,tag,attrs):
        for key,value in attrs:
            if key not in {'src','href','poster'} or not value:continue
            u=urlsplit(value)
            if u.scheme or u.netloc or not u.path:continue
            target=posixpath.normpath(posixpath.join(self.parent,unquote(u.path)))
            counts['local_links']+=1
            if target not in keys:errors.append(f'Missing site link {self.file}: {value}')
for p in ROOT.rglob('*'):
    if not p.is_file() or any(x in {'.git','.venv','__pycache__','downloads','release'} for x in p.relative_to(ROOT).parts):continue
    if p.stat().st_size>100*1024**2:errors.append('File exceeds normal Git limit: '+p.relative_to(ROOT).as_posix())
    if p.suffix=='.py':
        try:ast.parse(p.read_text(encoding='utf-8-sig'));counts['python']+=1
        except SyntaxError as e:errors.append(str(e))
    if p.suffix=='.json':
        try:json.loads(p.read_text(encoding='utf-8-sig'));counts['json']+=1
        except ValueError as e:errors.append(str(e))
    if p.suffix=='.html' and p.is_relative_to(SITE):
        parser=Links();parser.file=p.relative_to(SITE).as_posix();parser.parent=posixpath.dirname(parser.file);parser.feed(p.read_text(encoding='utf-8'))
manifest=json.loads((SITE/'public-manifest.json').read_text(encoding='utf-8'))
for v in manifest['videos']:
    p=SITE/'reports'/v['file'];counts['videos']+=1
    if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=v['sha256']:errors.append('Video hash mismatch: '+v['file'])
    if not v['full_decode_passed']:errors.append('Missing original full-decode receipt: '+v['file'])
source=json.loads((ROOT/'SOURCE_PROVENANCE.json').read_text())
for row in source['files']:
    p=ROOT/row['path']
    if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=row['public_sha256']:errors.append('Source provenance mismatch: '+row['path'])
print(json.dumps({'status':'passed' if not errors else 'failed','checks':counts,'errors':errors},ensure_ascii=False,indent=2))
sys.exit(bool(errors))
