"""Dependency-free integrity checks for the curated research and static reports."""
import ast
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import posixpath
import re
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / 'docs/site'
errors = []
counts = {'python': 0, 'json': 0, 'links': 0, 'provenance': 0, 'videos': 0, 'experiments': 0}
skip = {'.git', '__pycache__', '.venv', '.venv310', 'downloads', 'release'}
secrets = re.compile(r'(?:ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class Links(HTMLParser):
    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key not in {'href', 'src', 'poster'} or not value: continue
            url = urlsplit(value.replace('\\', '/'))
            if url.scheme or url.netloc or not url.path: continue
            target = (self.file.parent / unquote(url.path)).resolve()
            counts['links'] += 1
            if not target.is_relative_to(SITE) or not target.exists():
                errors.append('Missing/escaping link: ' + str(self.file.relative_to(SITE)) + ' -> ' + value)


for path in ROOT.rglob('*'):
    if not path.is_file() or any(part in skip for part in path.relative_to(ROOT).parts): continue
    if path.stat().st_size > 100 * 1024**2: errors.append('Exceeds regular Git file limit: ' + str(path.relative_to(ROOT)))
    if path.suffix not in {'.py', '.json', '.jsonl', '.html', '.md', '.txt', '.yml', '.yaml', '.sh', '.toml'}: continue
    try: text = path.read_text(encoding='utf-8-sig')
    except UnicodeDecodeError:
        errors.append('Non-UTF8 text: ' + str(path.relative_to(ROOT))); continue
    if secrets.search(text): errors.append('Potential credential material: ' + str(path.relative_to(ROOT)))
    try:
        if path.suffix == '.py': ast.parse(text); counts['python'] += 1
        if path.suffix == '.json': json.loads(text); counts['json'] += 1
    except (ValueError, SyntaxError) as error: errors.append(str(path.relative_to(ROOT)) + ': ' + str(error))
    if path.suffix == '.html' and path.is_relative_to(SITE):
        parser = Links(); parser.file = path; parser.feed(text)

for row in json.loads((ROOT / 'SOURCE_PROVENANCE.json').read_text(encoding='utf-8'))['files']:
    path = ROOT / row['path']; counts['provenance'] += 1
    if not path.exists() or sha(path) != row['public_sha256']: errors.append('Source hash: ' + row['path'])
manifest = json.loads((SITE / 'public-manifest.json').read_text(encoding='utf-8'))
for row in manifest['videos']:
    path = SITE / 'reports' / row['file']; counts['videos'] += 1
    if not path.exists() or sha(path) != row['sha256'] or not row['full_decode_passed']: errors.append('Video integrity: ' + row['file'])
for row in manifest['reports']:
    path = SITE / 'reports' / row['report']
    if not path.exists() or sha(path) != row['public_sha256']: errors.append('Report hash: ' + row['report'])
registry = json.loads((ROOT / 'EXPERIMENTS.json').read_text(encoding='utf-8'))
site_registry = json.loads((SITE / 'experiment-registry.json').read_text(encoding='utf-8'))
if registry != site_registry: errors.append('Source/site registries differ')
ids = set()
for row in registry['experiments']:
    counts['experiments'] += 1
    if row['id'] in ids: errors.append('Duplicate experiment: ' + row['id'])
    ids.add(row['id'])
    if not (SITE / 'reports' / row['report']).exists(): errors.append('Missing experiment report: ' + row['id'])
print(json.dumps({'status': 'failed' if errors else 'passed', 'checks': counts, 'errors': errors}, ensure_ascii=True, indent=2))
sys.exit(bool(errors))
