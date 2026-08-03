#!/usr/bin/env bash
# Unzip the released MemoryBench data tarballs into the expected layout.
# The release ships `<task>.zip` for both train and test splits; this script
# unzips them in-place so client.py / dataset can read them directly.
set -euo pipefail

# BRIDGEVLA_ROOT is derived from this script's location (three levels above <repo>/finetune/memoryBench/scripts/).
BRIDGEVLA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${BRIDGEVLA_ROOT}/data/bridgevla_data/memorybench/data}"

for split in train test; do
    src_dir="${DATA_ROOT}/${split}"
    [ -d "${src_dir}" ] || { echo "[unzip] missing split dir: ${src_dir}"; continue; }
    for zip_path in "${src_dir}"/*.zip; do
        [ -f "${zip_path}" ] || continue
        task="$(basename "${zip_path}" .zip)"
        out="${src_dir}/${task}"
        if [ -d "${out}/all_variations/episodes" ] && \
           [ "$(ls -A "${out}/all_variations/episodes" 2>/dev/null | wc -l)" -ge 2 ]; then
            echo "[unzip] ${split}/${task}: already extracted, skipping"
            continue
        fi
        echo "[unzip] extracting ${zip_path} -> ${src_dir}"
        unzip -q -n "${zip_path}" -d "${src_dir}"
    done
done
echo "[unzip] DONE."
