#!/usr/bin/env bash
# Run from a pinned checkout in a finite CPU ACP job. No credentials are needed.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
root=/mnt/afs/260010076/SuperMimic/official-20261112
prefix="$root/conda/supermimic-official"
mkdir -p "$root/logs" "$root/downloads" "$root/cache" "$root/data"
exec > >(tee -a "$root/logs/bootstrap-linux.log") 2>&1
printf 'OFFICIAL_BOOTSTRAP_START %s\n' "$(date -u +%FT%TZ)"
git -C "$repo" rev-parse HEAD
conda_bin=$(command -v conda || true)
if [[ -z "$conda_bin" && -x /opt/conda/bin/conda ]]; then
  conda_bin=/opt/conda/bin/conda
fi
# Reuse only the Conda executable; create a separate environment below our run root.
existing_conda=/mnt/afs/260010076/sensecore-envs/miniforge3-26.7.2-0/bin/conda
if [[ -z "$conda_bin" && -x "$existing_conda" ]]; then
  conda_bin="$existing_conda"
fi
if [[ -z "$conda_bin" ]]; then
  conda_bin="$root/miniforge/bin/conda"
  if [[ ! -x "$conda_bin" ]]; then
    url=https://github.com/conda-forge/miniforge/releases/download/26.9.4-2
    installer=Miniforge3-26.9.4-2-Linux-x86_64.sh
    cd "$root/downloads"
    curl --fail --location --retry 3 --connect-timeout 20 --max-time 600 "$url/$installer" -o "$installer"
    curl --fail --location --retry 3 --connect-timeout 20 --max-time 90 "$url/$installer.sha256" -o "$installer.sha256"
    sha256sum --check "$installer.sha256"
    bash "$installer" -b -p "$root/miniforge"
  fi
fi
export CONDA_PKGS_DIRS="$root/cache/conda"
export UV_CACHE_DIR="$root/cache/uv"
export PIP_DISABLE_PIP_VERSION_CHECK=1
printf 'CONDA_EXECUTABLE=%s\n' "$conda_bin"
if [[ ! -x "$prefix/bin/python" ]]; then
  "$conda_bin" create --prefix "$prefix" --override-channels --channel conda-forge python=3.13 pip -y
fi
"$prefix/bin/python" -m pip --isolated install --index-url https://pypi.org/simple uv==0.13.7 pytest==9.2.0
"$prefix/bin/uv" pip install --python "$prefix/bin/python" --index-url https://download.pytorch.org/whl/cu126 torch==2.14.0
cd "$repo"
"$prefix/bin/uv" pip install --python "$prefix/bin/python" --index-url https://pypi.org/simple \
  --constraint official-reproduction-20261112/runtime-constraints.txt \
  -e upstream/mjlab_research -e upstream/supermimic
"$prefix/bin/python" -m pip check
"$prefix/bin/python" tools/verify_official_import.py
"$prefix/bin/python" -m pip freeze > "$root/logs/linux-environment.freeze.txt"
"$prefix/bin/python" official-reproduction-20261112/runtime_probe.py "$root/logs/linux-runtime.json"
"$prefix/bin/python" -m pytest upstream/supermimic/tests -q --tb=short \
  --junitxml="$root/logs/official-tests-linux.xml"
printf 'OFFICIAL_BOOTSTRAP_PASS %s\n' "$(date -u +%FT%TZ)"
