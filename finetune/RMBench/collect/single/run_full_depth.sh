#!/bin/bash
# Collect full-frame depth for one task (demo_clean_depth config)
# Usage: bash collect/single/run_full_depth.sh <task_name> <gpu_id>
# Available tasks (10 with env modules + seed.txt + _traj_data):
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
task_config=demo_clean_depth
gpu_id=${2:-0}

./script/.update_path.sh > /dev/null 2>&1

src_dir="${RMBENCH_DATA_PATH}/data/${task_name}/demo_clean"
dst_dir="${RMBENCH_DATA_PATH}/data_self/${task_name}/${task_config}"
mkdir -p "$dst_dir"

# use_seed=true in config: skip motion planning, replay saved joint paths directly
# _traj_data stores per-episode joint waypoints computed during seed collection;
# without it the data-collection phase has no path to replay and will crash
if [ ! -d "$dst_dir/_traj_data" ]; then
    cp -r "$src_dir/_traj_data" "$dst_dir/_traj_data"
fi

# seed.txt: fixed episode seeds so scene randomization matches original data
if [ ! -f "$dst_dir/seed.txt" ]; then
    cp "$src_dir/seed.txt" "$dst_dir/seed.txt"
fi

export CUDA_VISIBLE_DEVICES=${gpu_id}

PYTHONWARNINGS=ignore::UserWarning \
python script/collect_data.py $task_name $task_config

rm -rf "${RMBENCH_DATA_PATH}/data_self/${task_name}/${task_config}/.cache"
