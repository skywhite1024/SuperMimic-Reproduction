"""Download the authorized fixed BONES G1 archive and verify its content hash.

Uses the user's normal Hugging Face login implicitly; never accepts or logs tokens.
Interrupted downloads are resumed by huggingface_hub in the same output directory.
"""

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import get_hf_file_metadata, get_token, hf_hub_download, hf_hub_url

REPO = "bones-studio/seed"
REVISION = "2f59b2077b9da34dd4e43618e705c7cb962c9a66"
FILENAME = "g1.tar.gz"
SIZE = 23_499_973_647
SHA256 = "52580ea8bced72ea9e2ff1e7b68f01c51c7f1099581e9a46b7c87e1dec106d8a"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "download-status.json"
    state = {
        "repository": REPO,
        "revision": REVISION,
        "filename": FILENAME,
        "expected_bytes": SIZE,
        "expected_sha256": SHA256,
        "started_at": datetime.now(UTC).isoformat(),
    }

    def record(status, **fields):
        state.update(status=status, updated_at=datetime.now(UTC).isoformat(), **fields)
        temporary = state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        temporary.replace(state_path)
        print(json.dumps({"status": status, **fields}), flush=True)

    try:
        if get_token() is None:
            raise RuntimeError("HuggingFaceLoginRequired")
        metadata = get_hf_file_metadata(
            hf_hub_url(REPO, FILENAME, repo_type="dataset", revision=REVISION), timeout=30
        )
        if (metadata.commit_hash, metadata.size, metadata.etag) != (REVISION, SIZE, SHA256):
            raise RuntimeError("RemoteMetadataMismatch")
        # A complete file can be reverified even when less than 24 GB remains.
        destination = output / FILENAME
        if not destination.exists() and shutil.disk_usage(output).free < SIZE + 2_000_000_000:
            raise RuntimeError("InsufficientDiskSpace")
        record("downloading", metadata_verified=True)
        downloaded = Path(hf_hub_download(
            REPO, FILENAME, repo_type="dataset", revision=REVISION, local_dir=output
        ))
        record("verifying")
        if downloaded.stat().st_size != SIZE:
            raise RuntimeError("DownloadedSizeMismatch")
        with downloaded.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        if digest != SHA256:
            raise RuntimeError("DownloadedHashMismatch")
        record("verified", bytes=SIZE, sha256=digest, path=str(downloaded))
    except Exception as exc:  # noqa: BLE001 - redact all upstream error messages
        # Exception messages may contain temporary signed URLs. Keep them out of logs.
        record("failed", error_type=type(exc).__name__)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
