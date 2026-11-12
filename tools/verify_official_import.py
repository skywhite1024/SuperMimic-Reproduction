"""Verify vendored official sources without importing or installing their runtime."""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    if sys.version_info < (3, 13):
        raise SystemExit("Use Python 3.13+ for the official source syntax check.")
    manifest = json.loads((ROOT / "upstream/SOURCES.lock.json").read_text("utf-8"))
    errors = []
    python_files = 0
    total_bytes = 0
    for source in manifest["sources"]:
        expected = set()
        for row in source["files"]:
            relative = f"{source['path']}/{row['path']}"
            expected.add(row["path"])
            path = ROOT / relative
            if not path.is_file() and not path.is_symlink():
                errors.append(f"Missing: {relative}")
                continue
            data = os.readlink(path).encode("utf-8") if path.is_symlink() else path.read_bytes()
            total_bytes += len(data)
            if hashlib.sha256(data).hexdigest() != row["sha256"]:
                errors.append(f"SHA256 mismatch: {relative}")
            if path.suffix == ".py":
                try:
                    ast.parse(data, filename=relative)
                    python_files += 1
                except SyntaxError as error:
                    errors.append(f"Syntax error: {error}")
        actual = {
            p.relative_to(ROOT / source["path"]).as_posix()
            for p in (ROOT / source["path"]).rglob("*")
            if (p.is_file() or p.is_symlink())
            and not any(part in {"__pycache__", ".venv", ".pytest_cache", ".ruff_cache"} for part in p.parts)
        }
        errors.extend(f"Unexpected source file: {source['path']}/{p}" for p in sorted(actual - expected))
    result = {
        "status": "passed" if not errors else "failed",
        "source_files": sum(len(s["files"]) for s in manifest["sources"]),
        "python_syntax_files": python_files,
        "bytes": total_bytes,
        "runtime_imports_or_training_tested": False,
        "errors": errors,
    }
    print(json.dumps(result, indent=2))
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
