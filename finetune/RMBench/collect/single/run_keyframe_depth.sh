#!/bin/bash
# Collect depth ONLY at keyframe frames (~30-40x faster than full collection)
# Usage: bash collect/single/run_keyframe_depth.sh <task_name> <gpu_id>
# Available tasks (10 with env modules):
#   battery_try        blocks_ranking_try  cover_blocks      observe_and_pickup
#   place_block_mat    press_button        put_back_block    rearrange_blocks
#   swap_blocks        swap_T

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Paths are derived from the repository root (RMBENCH_DATA_PATH overrides them); conda is auto-detected (CONDA_BASE overrides it).
BRIDGEVLA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
export RMBENCH_DATA_PATH="${RMBENCH_DATA_PATH:-${BRIDGEVLA_ROOT}/data/bridgevla_data/RMBench/data}"
export RMBENCH_ASSETS_PATH="${RMBENCH_ASSETS_PATH:-${BRIDGEVLA_ROOT}/data/bridgevla_data/RMBench/assets}"
if [ -z "${CONDA_BASE:-}" ]; then
    CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    [ -z "${CONDA_BASE}" ] && [ -d "${HOME}/miniconda3" ] && CONDA_BASE="${HOME}/miniconda3"
    [ -z "${CONDA_BASE}" ] && [ -d "${HOME}/anaconda3" ] && CONDA_BASE="${HOME}/anaconda3"
fi
source "${CONDA_BASE}/bin/activate" "${RMBENCH_CONDA_ENV:-bridgevla_plus_rmbench}"

task_name=${1:-battery_try}
gpu_id=${2:-0}

./script/.update_path.sh > /dev/null 2>&1

export CUDA_VISIBLE_DEVICES=${gpu_id}

PYTHONWARNINGS=ignore::UserWarning \
python script/collect_keyframe_depth.py $task_name

rm -rf "${RMBENCH_DATA_PATH}/keyframe_data/${task_name}/keyframe_depth/.cache"
