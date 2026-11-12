"""Emit bounded JSON artifacts in short, hash-verifiable log records.

The cloud offline log service truncates long individual messages. This helper
reads explicitly named JSON files only and keeps every output line below 512
characters. It never reads environment variables or discovers additional files.
"""

import argparse
import base64
import hashlib
import json
import zlib
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    for index, path in enumerate(args.paths):
        if path.suffix != ".json" or path.stat().st_size > 2_000_000:
            raise ValueError("Only explicitly named JSON receipts up to 2 MB are accepted")
        data = path.read_bytes()
        json.loads(data)
        encoded = base64.b64encode(zlib.compress(data, level=9)).decode("ascii")
        chunks = [encoded[i:i + 360] for i in range(0, len(encoded), 360)]
        metadata = {"i": index, "path": str(path), "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(), "chunks": len(chunks)}
        line = "RECEIPT_META " + json.dumps(metadata, separators=(",", ":"))
        if len(line) >= 512:
            raise ValueError("Receipt path is too long for the log transport")
        print(line, flush=True)
        for chunk_index, chunk in enumerate(chunks):
            print(f"RECEIPT_DATA {index} {chunk_index} {chunk}", flush=True)
    print("RECEIPT_EXPORT_COMPLETED", flush=True)


if __name__ == "__main__":
    main()
