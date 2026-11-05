"""Recover the complete finite-job receipt from the console's JSONL log export."""
import argparse
import base64
import gzip
import hashlib
import json
import re
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('console_export', type=Path)
    args = parser.parse_args()
    metadata = []
    parts = {}
    events = []
    complete = False
    with gzip.open(args.console_export, 'rt', encoding='utf8') as stream:
        logs = [json.loads(line) for line in stream if line.strip()]
    for row in logs:
        body = row['Body']
        if 'S8807_META ' in body:
            metadata.append(json.loads(body.split('S8807_META ', 1)[1]))
        match = re.search(r'S8807_PART (\d+)/(\d+) ([A-Za-z0-9+/=]+)', body)
        if match:
            index, total = map(int, match.group(1, 2))
            item = (total, match.group(3))
            assert index not in parts or parts[index] == item, 'Conflicting console parts'
            parts[index] = item
        if 'S8807_EVT ' in body:
            events.append({'timestamp': row['Timestamp'],
                           **json.loads(body.split('S8807_EVT ', 1)[1])})
        complete |= 'S8807_COMPLETE' in body
    assert complete and len(metadata) == 1, 'Final console receipt is not complete'
    meta = metadata[0]
    assert set(parts) == set(range(1, meta['parts']+1))
    assert all(total == meta['parts'] for total, _ in parts.values())
    raw = zlib.decompress(base64.b64decode(''.join(parts[i][1] for i in sorted(parts)),
                                         validate=True))
    assert len(raw) == meta['bytes']
    assert hashlib.sha256(raw).hexdigest() == meta['sha256']
    result = json.loads(raw)
    target = ROOT/'downloaded/specialist8807-1031d'
    target.mkdir(parents=True, exist_ok=True)
    receipt = target/'recovered.json'
    assert not receipt.exists() or receipt.read_bytes() == raw
    receipt.write_bytes(raw)
    verification = {'status': 'complete_console_receipt_independently_verified',
        'console_export': str(args.console_export.resolve()),
        'console_export_bytes': args.console_export.stat().st_size,
        'console_export_sha256': hashlib.sha256(args.console_export.read_bytes()).hexdigest(),
        'log_rows': len(logs), 'complete_marker': complete, 'chunks': len(parts),
        'receipt_bytes': len(raw), 'receipt_sha256': meta['sha256'],
        'receipt': receipt.relative_to(ROOT).as_posix(), 'events': events,
        'new_training_updates': result['new_training_updates'],
        'actual_platform_terminal_requires_separate_UI_evidence': True}
    (ROOT/'evidence/specialist8807-1031d-console-recovery.json').write_text(
        json.dumps(verification, indent=2)+'\n')
    print(json.dumps({'status': result['status'], 'updates': result['new_training_updates'],
                      'counts': result.get('counts'), 'worst': result.get('worst'),
                      'archive': result.get('archive'), 'error': result.get('error')}), flush=True)


if __name__ == '__main__':
    main()
