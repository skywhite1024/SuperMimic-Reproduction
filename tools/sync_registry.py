"""Rebuild the static experiment table from the canonical EXPERIMENTS.json."""
from pathlib import Path
import html
import json
import re

root = Path(__file__).resolve().parents[1]
site = root / 'docs/site'
data = json.loads((root / 'EXPERIMENTS.json').read_text(encoding='utf-8'))
rows = data['experiments']
body = ''.join(
    '<tr data-route="' + html.escape(row['route']) + '"><td>' + html.escape(row['id']) +
    '</td><td>' + html.escape(row['route']) + '</td><td><a href="reports/' +
    html.escape(row['report']) + '">' + html.escape(row['title']) +
    '</a></td><td><span class="badge">' + html.escape(row['status']) +
    '</span></td><td>' + html.escape(row['finding']) + '</td></tr>' for row in rows)
path = site / 'index.html'
text = path.read_text(encoding='utf-8')
text, count = re.subn(r'<tbody>.*?</tbody>', lambda match: '<tbody>' + body + '</tbody>', text, count=1, flags=re.S)
assert count == 1
options = '<option value="">全部路线</option>' + ''.join('<option>' + html.escape(route) + '</option>' for route in sorted({row['route'] for row in rows}))
text, count = re.subn(r'(<select id="route">).*?(</select>)', lambda match: match[1] + options + match[2], text, count=1, flags=re.S)
assert count == 1
path.write_text(text, encoding='utf-8', newline='\n')
(site / 'experiment-registry.json').write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
print(json.dumps({'experiments': len(rows), 'source': 'EXPERIMENTS.json', 'site_table_updated': True}))
