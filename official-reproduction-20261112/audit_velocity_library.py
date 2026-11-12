"""Inspect cross-machine packed-file differences without changing either pack.

Write the complete audit before reporting whether raw array payloads match.
Whole-file and payload identity are separate evidence; header differences do
not silently qualify as whole-file identity. No dataset files are rewritten.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path, offset=0):
    with path.open("rb") as stream:
        stream.seek(offset)
        return hashlib.file_digest(stream, "sha256").hexdigest()


def audit(folder, expected_manifest, expected_payloads, output):
    if output.exists():
        raise FileExistsError("A fresh audit output is required")
    actual = json.loads((folder / "library-manifest.json").read_text(encoding="utf-8"))
    expected = json.loads(expected_manifest.read_text(encoding="utf-8"))
    payloads = json.loads(expected_payloads.read_text(encoding="utf-8"))
    expected_files = {entry["file"]: entry for entry in expected["packed_files"]}
    actual_files = {entry["file"]: entry for entry in actual["packed_files"]}
    if set(actual_files) != set(expected_files):
        raise ValueError("The complete packed file inventories differ")
    files = []
    for name in sorted(actual_files):
        path = folder / "packed" / name
        current = {"file": name, "bytes": path.stat().st_size, "sha256": digest(path)}
        entry = {"file": name, "current_file": current,
                 "matches_cloud_manifest": current == actual_files[name],
                 "matches_local_complete_file": current == expected_files[name]}
        if name.endswith(".npy"):
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            current_payload = {"shape": list(array.shape), "dtype": array.dtype.str,
                               "header_bytes": array.offset,
                               "payload_sha256": digest(path, array.offset),
                               "payload_bytes": int(array.nbytes)}
            entry["array"] = current_payload
            entry["matches_local_payload"] = all(current_payload[key] == payloads[name][key]
                for key in ("shape", "dtype", "payload_sha256", "payload_bytes"))
            with path.open("rb") as stream:
                entry["npy_header_hex"] = stream.read(array.offset).hex()
            del array
        files.append(entry)
    metadata_keys = ["status", "kind", "verification_scope", "fps", "clips", "frames",
        "duration_hours", "selection_rule", "author_membership_confirmed",
        "semantic_motion_quality_verified", "parent_sources", "member_ledger",
        "selected_array_comparisons", "builder_sha256"]
    metadata_differences = [key for key in metadata_keys if actual[key] != expected[key]]
    intact = all(entry["matches_cloud_manifest"] for entry in files)
    payload_identity = all(entry.get("matches_local_payload", True) for entry in files)
    file_identity = all(entry["matches_local_complete_file"] for entry in files)
    # JSON file identity is required separately; payload comparison covers NPY only.
    json_identity = all(entry["matches_local_complete_file"] for entry in files
                        if entry["file"].endswith(".json"))
    result = {"status": "cross_machine_library_diagnostic_completed",
        "cloud_manifest_sha256": digest(folder / "library-manifest.json"),
        "expected_manifest_sha256": digest(expected_manifest),
        "expected_payloads_sha256": digest(expected_payloads),
        "cloud_files_intact": intact, "all_numpy_payloads_equal": payload_identity,
        "all_json_files_identical": json_identity, "all_complete_files_identical": file_identity,
        "metadata_differences": metadata_differences, "files": files,
        "numpy_version": np.__version__, "dataset_modified": False,
        "author_membership_confirmed": False, "paper_results_reproduced": False}
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("VELOCITY_LIBRARY_DIAGNOSTIC " + json.dumps({key: value for key, value in result.items()
          if key != "files"}), flush=True)
    if not intact or not payload_identity or not json_identity or metadata_differences:
        raise RuntimeError("Cross-machine data identity is unresolved; inspect the complete audit")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--expected-manifest", type=Path, required=True)
    parser.add_argument("--expected-payloads", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.folder, args.expected_manifest, args.expected_payloads, args.output)


if __name__ == "__main__":
    main()
