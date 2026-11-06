"""Recover SHA-verified JSON exports from a console JSONL(.gz) log download."""
import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path


def recover(lines):
    metadata = {}; chunks = {}; done = {}
    for line in lines:
        body = json.loads(line).get('Body', '')
        if body.startswith('REPLAY_EXPORT_META '):
            row = json.loads(body.split(' ', 1)[1]); label = row['label']
            if label in metadata: assert metadata[label] == row
            metadata[label] = row
        elif body.startswith('REPLAY_EXPORT_CHUNK '):
            _, label, index, data = body.split(' ', 3)
            bucket = chunks.setdefault(label, {}); index = int(index)
            if index in bucket: assert bucket[index] == data
            bucket[index] = data
        elif body.startswith('REPLAY_EXPORT_DONE '):
            _, label, digest = body.split(' ')
            if label in done: assert done[label] == digest
            done[label] = digest
    results = {}
    for label, meta in metadata.items():
        if label not in done: continue  # A snapshot can end during an active export.
        assert label in chunks and set(chunks[label]) == set(range(meta['chunks']))
        packed = base64.b64decode(''.join(chunks[label][i] for i in range(meta['chunks'])), validate=True)
        assert len(packed) == meta['bytes'] and hashlib.sha256(packed).hexdigest() == meta['sha256'] == done[label]
        results[label] = json.loads(gzip.decompress(packed))
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('log', type=Path); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    stream = gzip.open if a.log.suffix == '.gz' else open
    with stream(a.log, 'rt', encoding='utf-8-sig') as handle: recovered = recover(handle)
    assert recovered, 'No complete, verified export in this snapshot'
    a.output.mkdir(parents=True, exist_ok=False)
    for label, payload in recovered.items():
        assert label and all(c.isalnum() or c in '-_' for c in label)
        (a.output/f'{label}.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps({'verified_exports': list(recovered), 'output': str(a.output)}, indent=2))


if __name__ == '__main__': main()
