#!/usr/bin/env bash
# Download RMBench simulation assets (embodiments + objects) from the
# HuggingFace dataset TianxingChen/RMBench into this repo's canonical
# location and rewrite the embodiment configs' absolute paths.
#
#   destination: data/bridgevla_data/RMBench/assets   (RMBENCH_ASSETS_PATH
#                overrides; eval/collect scripts resolve the same variable)
#
# Requires huggingface_hub in the active env (any of the repo envs has it).
# Safe to re-run: snapshot_download resumes/skips completed files.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RMBENCH_DIR="$(dirname "${SCRIPT_DIR}")"                      # finetune/RMBench
BRIDGEVLA_ROOT="$(dirname "$(dirname "${RMBENCH_DIR}")")"     # repo root

export RMBENCH_ASSETS_PATH="${RMBENCH_ASSETS_PATH:-${BRIDGEVLA_ROOT}/data/bridgevla_data/RMBench/assets}"
mkdir -p "${RMBENCH_ASSETS_PATH}"

echo "[rmbench-assets] downloading embodiments/ + objects/ -> ${RMBENCH_ASSETS_PATH}"
python - "$RMBENCH_ASSETS_PATH" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="TianxingChen/RMBench",
    repo_type="dataset",
    allow_patterns=["embodiments/**", "objects/**"],
    local_dir=sys.argv[1],
)
PY

# Rewrite absolute paths inside assets/embodiments/*/config.yml. The helper
# expects a cwd that CONTAINS assets/embodiments, i.e. the parent of the
# assets dir.
echo "[rmbench-assets] configuring embodiment config paths ..."
cd "$(dirname "${RMBENCH_ASSETS_PATH}")"
python "${SCRIPT_DIR}/update_embodiment_config_path.py"
echo "[rmbench-assets] done."
