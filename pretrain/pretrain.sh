#!/usr/bin/env bash

# Paths and environment: all derived from the repository root, no machine-specific config;
_PT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRIDGEVLA_ROOT="$(dirname "${_PT_DIR}")"                           # <repo>
FINETUNE_DIR="${BRIDGEVLA_ROOT}/finetune"
export BRIDGEVLA_DATA_ROOT="${BRIDGEVLA_DATA_ROOT:-${BRIDGEVLA_ROOT}/data/bridgevla_data}"
export BRIDGEVLA_LOG_DIR="${BRIDGEVLA_LOG_DIR:-${BRIDGEVLA_DATA_ROOT}/logs}"
export BRIDGEVLA_CKPT_ROOT="${BRIDGEVLA_CKPT_ROOT:-${BRIDGEVLA_ROOT}/data/bridgevla_ckpt}"

# conda (CONDA_BASE overridable, auto-detected by default; the env name can be overridden with PRETRAIN_CONDA_ENV).
# The default env is built by pretrain/install_pretrain.sh. Its packages are a strict subset of the
# GemBench env with identical pins, so PRETRAIN_CONDA_ENV=bridgevla_plus_gembench also works.
if [ -z "${CONDA_BASE:-}" ]; then
    CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    [ -z "${CONDA_BASE}" ] && [ -d "${HOME}/miniconda3" ] && CONDA_BASE="${HOME}/miniconda3"
    [ -z "${CONDA_BASE}" ] && [ -d "${HOME}/anaconda3" ] && CONDA_BASE="${HOME}/anaconda3"
fi
source "${CONDA_BASE}/bin/activate" "${PRETRAIN_CONDA_ENV:-bridgevla_plus_pretrain}"
PRETRAIN_DIR="${BRIDGEVLA_ROOT}/pretrain"

# The pretrain script imports from bridgevla.*, so point PYTHONPATH at the
# finetune directory (which contains the `bridgevla` package).
export PYTHONPATH="${FINETUNE_DIR}:${FINETUNE_DIR}/bridgevla/libs/point-renderer:${FINETUNE_DIR}/bridgevla/libs/peract_colab:${FINETUNE_DIR}/bridgevla/libs/YARR:${FINETUNE_DIR}/GemBench:${PYTHONPATH:-}"
export PALIGEMMA_PATH="${PALIGEMMA_PATH:-${BRIDGEVLA_CKPT_ROOT}/paligemma-3b-pt-224}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
# ---- SwanLab one-line switch ----
# Set to "cloud" to upload, or "offline" to only save locally (offline is the default).
# Note: swanlab's "local" mode connects to a self-hosted swanboard server and needs
# `pip install 'swanlab[dashboard]'` — it is not "local files only". For purely offline
# files use "offline".
SWANLAB_UPLOAD="${SWANLAB_UPLOAD:-offline}"
# Backwards compatibility: SWANLAB_UPLOAD=local in older scripts actually meant "save locally only".
if [ "${SWANLAB_UPLOAD}" = "local" ]; then
    SWANLAB_UPLOAD="offline"
fi
export SWANLAB_MODE="${SWANLAB_UPLOAD}"
if [ "${SWANLAB_UPLOAD}" = "cloud" ]; then
    export SWANLAB_API_KEY="${SWANLAB_API_KEY:?SWANLAB_UPLOAD=cloud requires exporting SWANLAB_API_KEY=<your key> first}"
else
    unset SWANLAB_API_KEY
fi

# The data paths live in pretrain_config.yaml (image_folder / json_detection_path);
# to swap the data temporarily, just pass --image_folder / --json_detection_path through.

# Distributed: single node, 8 GPUs by default; override with WORLD_SIZE / RESOURCE_GPU for multi-node or fewer GPUs.
export MLP_WORKER_NUM=${WORLD_SIZE:-1}
export MLP_WORKER_GPU=${RESOURCE_GPU:-8}
export MLP_ROLE_INDEX=${RANK:-0}
export MLP_WORKER_0_HOST=${MASTER_ADDR:-localhost}
export MLP_WORKER_0_PORT=${MASTER_PORT:-29504}

# --- Terminal log capture -------------------------------------------------
# Pre-compute the run dir in shell so we can tee torchrun's stdout/stderr
# into the SAME folder pretrain.py writes checkpoints to. We share the stamp
# with Python via PRETRAIN_RUN_STAMP so the two sides agree on the path.
# On multi-node runs every node tees to its own train_node{N}.log to avoid
# clobbering on a shared FS.
CONFIG_FILE="${PRETRAIN_DIR}/pretrain_config.yaml"
OUTPUT_DIR_CFG="$(python -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["output_dir"])' "${CONFIG_FILE}")"
EXP_NAME_CFG="$(python -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["exp_name"])' "${CONFIG_FILE}")"
export PRETRAIN_RUN_STAMP="${PRETRAIN_RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${BRIDGEVLA_ROOT}/${OUTPUT_DIR_CFG}/${EXP_NAME_CFG}/${PRETRAIN_RUN_STAMP}"
mkdir -p "${RUN_DIR}"
LOG_FILE="${RUN_DIR}/train_node${MLP_ROLE_INDEX}.log"
echo "[pretrain.sh] run dir: ${RUN_DIR}"
echo "[pretrain.sh] logging torchrun stdout/stderr to: ${LOG_FILE}"
# Force line-buffered Python so tee flushes per line, not per block.
export PYTHONUNBUFFERED=1

# pipefail: propagate torchrun's exit status through the tee pipeline so
# set -e still aborts the script on training failure.
set -e -x -o pipefail

cd "${BRIDGEVLA_ROOT}"

torchrun \
    --nnodes=$MLP_WORKER_NUM \
    --node_rank=$MLP_ROLE_INDEX \
    --nproc_per_node=$MLP_WORKER_GPU \
    --master_addr=$MLP_WORKER_0_HOST \
    --master_port=$MLP_WORKER_0_PORT \
    pretrain/pretrain.py \
    --branches 2 \
    --config_path pretrain/pretrain_config.yaml \
    "$@" 2>&1 | tee -a "${LOG_FILE}"
