#!/bin/bash
# Generic data collection wrapper (any task + config)
# Usage: bash collect/single/run_generic.sh <task_name> <task_config> <gpu_id>
# Example: bash collect/single/run_generic.sh cover_blocks demo_clean 0

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# The data/ symlink is gone; the real data location is derived from the repository root (RMBENCH_DATA_PATH overrides it).
BRIDGEVLA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
export RMBENCH_DATA_PATH="${RMBENCH_DATA_PATH:-${BRIDGEVLA_ROOT}/data/bridgevla_data/RMBench/data}"
export RMBENCH_ASSETS_PATH="${RMBENCH_ASSETS_PATH:-${BRIDGEVLA_ROOT}/data/bridgevla_data/RMBench/assets}"

task_name=${1}
task_config=${2}
gpu_id=${3}

./script/.update_path.sh > /dev/null 2>&1

export CUDA_VISIBLE_DEVICES=${gpu_id}

PYTHONWARNINGS=ignore::UserWarning \
python script/collect_data.py $task_name $task_config
rm -rf "${RMBENCH_DATA_PATH}/${task_name}/${task_config}/.cache"
