#!/usr/bin/env bash
# Download the RMBench demo_clean demonstrations from the HuggingFace dataset
# TianxingChen/RMBench into this repo's canonical location:
#
#   data/bridgevla_data/RMBench/data/<task>/demo_clean/...
#   (RMBENCH_DATA_PATH overrides; eval/collect scripts resolve the same var)
#
# Requires huggingface_hub in the active env.
# Safe to re-run: snapshot_download resumes/skips completed files.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RMBENCH_DIR="$(dirname "${SCRIPT_DIR}")"                      # finetune/RMBench
BRIDGEVLA_ROOT="$(dirname "$(dirname "${RMBENCH_DIR}")")"     # repo root

export RMBENCH_DATA_PATH="${RMBENCH_DATA_PATH:-${BRIDGEVLA_ROOT}/data/bridgevla_data/RMBench/data}"
# Repo files are prefixed data/<task>/demo_clean/**. When the destination is
# itself named .../data (the canonical layout) download into its PARENT so the
# repo prefix lines up with the final layout and resume stays cheap; for a
# custom destination, download nested and lift data/* up afterwards.
mkdir -p "${RMBENCH_DATA_PATH}"
if [ "$(basename "${RMBENCH_DATA_PATH}")" = "data" ]; then
    _DL_ROOT="$(dirname "${RMBENCH_DATA_PATH}")"
    _LIFT=0
else
    _DL_ROOT="${RMBENCH_DATA_PATH}"
    _LIFT=1
fi

echo "[rmbench-data] downloading data/*/demo_clean -> ${RMBENCH_DATA_PATH}"
python - "$_DL_ROOT" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="TianxingChen/RMBench",
    repo_type="dataset",
    allow_patterns=["data/*/demo_clean/**"],
    local_dir=sys.argv[1],
)
PY
if [ "${_LIFT}" = "1" ] && [ -d "${_DL_ROOT}/data" ]; then
    echo "[rmbench-data] moving ${_DL_ROOT}/data/* up into ${RMBENCH_DATA_PATH}"
    mv -n "${_DL_ROOT}/data/"* "${RMBENCH_DATA_PATH}/" 2>/dev/null || true
    rmdir "${_DL_ROOT}/data" 2>/dev/null || true
fi
echo "[rmbench-data] done."
