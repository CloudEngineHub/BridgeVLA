#!/bin/bash
# Batch: collect full-frame depth for all tasks across 8 GPUs
# (serial within each GPU, parallel across GPUs)
# Usage: bash collect/batch/run_all_full_depth.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SINGLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../single" && pwd)"
cd "$ROOT_DIR"

TASKS=(
    battery_try
    blocks_ranking_try
    cover_blocks
    observe_and_pickup
    place_block_mat
    press_button
    put_back_block
    rearrange_blocks
    swap_blocks
    swap_T
)

NUM_GPUS=8

for i in "${!TASKS[@]}"; do
    gpu_id=$((i % NUM_GPUS))
    task="${TASKS[$i]}"
    eval "GPU_${gpu_id}+=(\"$task\")"
done

for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    var="GPU_${gpu_id}[@]"
    task_list=("${!var}")
    [ ${#task_list[@]} -eq 0 ] && continue

    (
        for task in "${task_list[@]}"; do
            echo "[GPU $gpu_id] Starting: $task"
            bash "$SINGLE_DIR/run_full_depth.sh" "$task" "$gpu_id"
            echo "[GPU $gpu_id] Finished: $task"
        done
    ) &
done

wait
echo "All full-depth tasks completed."
