"""Resume BONES using bounded HTTP ranges when whole-object delivery stalls.

Normal Hugging Face login is used implicitly. Each completed range is hashed;
the full archive is accepted only after matching the pinned upstream SHA256.
"""

import argparse
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from download_bones import FILENAME, REPO, REVISION, SHA256, SIZE
from huggingface_hub import get_hf_file_metadata, hf_hub_url
from huggingface_hub.utils import build_hf_headers, get_session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk-mib", type=int, default=32)
    parser.add_argument("--max-seconds", type=int, default=7200)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8 or not 1 <= args.chunk_mib <= 128:
        parser.error("Use 1-8 workers and 1-128 MiB chunks")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    url = hf_hub_url(REPO, FILENAME, repo_type="dataset", revision=REVISION)
    metadata = get_hf_file_metadata(url, timeout=30)
    if (metadata.commit_hash, metadata.size, metadata.etag) != (REVISION, SIZE, SHA256):
        raise RuntimeError("RemoteMetadataMismatch")
    chunk_size = args.chunk_mib * 1024**2
    total = (SIZE + chunk_size - 1) // chunk_size
    target = root / FILENAME
    partial = root / (FILENAME + ".ranges.partial")
    ledger = root / "range-progress.json"
    started = time.monotonic()
    stop = threading.Event()
    completed = {}
    if target.exists():
        with target.open("rb") as stream:
            if target.stat().st_size == SIZE and hashlib.file_digest(stream, "sha256").hexdigest() == SHA256:
                print("BONES_ARCHIVE_ALREADY_VERIFIED", flush=True)
                return
        raise RuntimeError("ExistingDestinationHashMismatch")
    if ledger.exists() and partial.exists():
        saved = json.loads(ledger.read_text(encoding="utf-8"))
        if saved["revision"] != REVISION or saved["chunk_size"] != chunk_size:
            raise RuntimeError("ResumeConfigurationMismatch")
        with partial.open("rb") as stream:
            for key, digest in saved["completed"].items():
                offset = int(key) * chunk_size
                stream.seek(offset)
                block = stream.read(min(chunk_size, SIZE - offset))
                if hashlib.sha256(block).hexdigest() == digest:
                    completed[key] = digest
    if not partial.exists():
        with partial.open("xb") as stream:
            stream.truncate(SIZE)

    def report(status):
        received = sum(min(chunk_size, SIZE - int(key) * chunk_size) for key in completed)
        state = {
            "status": status, "revision": REVISION, "expected_sha256": SHA256,
            "expected_bytes": SIZE, "chunk_size": chunk_size, "completed": completed,
            "verified_range_bytes": received, "completed_chunks": len(completed),
            "total_chunks": total, "elapsed_seconds": round(time.monotonic()-started, 1),
            "updated_at": datetime.now(UTC).isoformat(),
        }
        temp = ledger.with_suffix(".tmp")
        temp.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        temp.replace(ledger)
        print(json.dumps({key: value for key, value in state.items() if key != "completed"}), flush=True)

    def fetch(index):
        left = index * chunk_size
        right = min(SIZE, left + chunk_size) - 1
        for attempt in range(3):
            if stop.is_set() or time.monotonic() - started > args.max_seconds:
                raise TimeoutError("BoundedDownloadWindowEnded")
            try:
                headers = build_hf_headers()
                headers["Range"] = f"bytes={left}-{right}"
                with get_session().stream("GET", url, headers=headers, follow_redirects=True, timeout=30) as response:
                    if response.status_code != 206:
                        raise RuntimeError("ExpectedPartialContent")
                    if response.headers.get("content-range") != f"bytes {left}-{right}/{SIZE}":
                        raise RuntimeError("UnexpectedContentRange")
                    digest = hashlib.sha256()
                    count = 0
                    with partial.open("r+b") as stream:
                        stream.seek(left)
                        for block in response.iter_bytes(chunk_size=1024**2):
                            if stop.is_set() or time.monotonic()-started > args.max_seconds:
                                raise TimeoutError("BoundedDownloadWindowEnded")
                            if count + len(block) > right-left+1:
                                raise RuntimeError("RangeOverflow")
                            stream.write(block)
                            digest.update(block)
                            count += len(block)
                        if count != right-left+1:
                            raise RuntimeError("IncompleteRange")
                        stream.flush()
                        os.fsync(stream.fileno())
                return index, digest.hexdigest()
            except Exception:
                if attempt == 2:
                    raise
                stop.wait(2 ** attempt)
        raise RuntimeError("RetriesExhausted")

    report("downloading")
    last_report = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch, index) for index in range(total) if str(index) not in completed]
        try:
            for future in as_completed(futures):
                index, digest = future.result()
                completed[str(index)] = digest
                if time.monotonic()-last_report >= 30 or len(completed) == total:
                    report("downloading")
                    last_report = time.monotonic()
        except BaseException as exc:  # noqa: BLE001 - persist progress on interrupt and redact URLs
            stop.set()
            for future in futures:
                future.cancel()
            report("interrupted")
            print(json.dumps({"error_type": type(exc).__name__}), flush=True)
            raise SystemExit(1) from None
    report("verifying")
    with partial.open("rb") as stream:
        actual_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual_hash != SHA256:
        report("hash_mismatch")
        raise SystemExit(1)
    partial.replace(target)
    report("verified")


if __name__ == "__main__":
    main()
