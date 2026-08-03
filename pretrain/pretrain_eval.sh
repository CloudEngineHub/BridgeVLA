#!/usr/bin/env bash
#
# Stand-alone eval for a pretrain checkpoint (branches=3 in pretrain.py).
#
# What it does:
#   * Loads $CHECKPOINT_PATH into BridgeVLAPretrainModel.
#   * Builds the dataset from $CONFIG_PATH: the val holdout carved off the
#     RoboPoint corpus (`val.holdout_samples`, the same rows the val/*
#     training metrics report), falling back to the train split when no
#     holdout is configured.
#   * Output layout under <checkpoint_dir>/eval_<ckpt_name>_<stamp>/:
#       <source_tag>_NNN.png   (1x4: image+GT bbox / image+GT hm /
#                               raw pred logits / image+pred hm)
#     <source_tag> is "val" or "train". Each panel caption carries the
#     dataset, source, VLM prompt and a metric line (PointHit / center_L1 /
#     loss_hm).
#   * Dumps the aggregates to eval_metrics.json alongside the panels.
#
# Override via env vars, e.g.:
#   CHECKPOINT_PATH=... CONFIG_PATH=... N_SAMPLES_PER_TASK=30 ./pretrain_eval.sh

# Paths and environment: all derived from the repository root, no machine-specific config;
_PT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRIDGEVLA_ROOT="$(dirname "${_PT_DIR}")"                           # <repo>
FINETUNE_DIR="${BRIDGEVLA_ROOT}/finetune"
export BRIDGEVLA_DATA_ROOT="${BRIDGEVLA_DATA_ROOT:-${BRIDGEVLA_ROOT}/data/bridgevla_data}"
export BRIDGEVLA_CKPT_ROOT="${BRIDGEVLA_CKPT_ROOT:-${BRIDGEVLA_ROOT}/data/bridgevla_ckpt}"

# conda (CONDA_BASE overridable, auto-detected by default; the env name can be overridden with PRETRAIN_CONDA_ENV)
if [ -z "${CONDA_BASE:-}" ]; then
    CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    [ -z "${CONDA_BASE}" ] && [ -d "${HOME}/miniconda3" ] && CONDA_BASE="${HOME}/miniconda3"
    [ -z "${CONDA_BASE}" ] && [ -d "${HOME}/anaconda3" ] && CONDA_BASE="${HOME}/anaconda3"
fi
source "${CONDA_BASE}/bin/activate" "${PRETRAIN_CONDA_ENV:-bridgevla_plus_pretrain}"
PRETRAIN_DIR="${BRIDGEVLA_ROOT}/pretrain"

export PYTHONPATH="${FINETUNE_DIR}:${FINETUNE_DIR}/bridgevla/libs/point-renderer:${FINETUNE_DIR}/bridgevla/libs/peract_colab:${FINETUNE_DIR}/bridgevla/libs/YARR:${FINETUNE_DIR}/GemBench:${PYTHONPATH:-}"
export PALIGEMMA_PATH="${PALIGEMMA_PATH:-${BRIDGEVLA_CKPT_ROOT}/paligemma-3b-pt-224}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# --- Eval inputs (override via env) ---------------------------------------
# Single-GPU evaluation; CHECKPOINT_PATH is required (defaults to the most recent run's pretrain_last.pth).
CONFIG_PATH="${CONFIG_PATH:-${PRETRAIN_DIR}/pretrain_config.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:?set CHECKPOINT_PATH=<run_dir>/pretrain_epoch_<N>.pth}"
N_SAMPLES_PER_TASK="${N_SAMPLES_PER_TASK:-20}"
# --- Terminal log capture -------------------------------------------------
# test_inference writes its artifacts under
#   <ckpt_dir>/eval_<ckpt_name>_<stamp>/
# but uses Python's own datetime stamp. We cannot predict that exact path
# from shell, so we tee to <ckpt_dir>/eval_<ckpt_name>_<shell_stamp>.log
# right next to the run dirs. Good enough to diagnose crashes (especially
# signal deaths like SIGFPE that never hit Python's traceback path).
EVAL_SHELL_STAMP="$(date +%Y%m%d_%H%M%S)"
CKPT_DIR="$(dirname "${CHECKPOINT_PATH}")"
CKPT_NAME="$(basename "${CHECKPOINT_PATH}" .pth)"
LOG_FILE="${CKPT_DIR}/eval_${CKPT_NAME}_${EVAL_SHELL_STAMP}.log"
echo "[pretrain_eval.sh] log: ${LOG_FILE}"
export PYTHONUNBUFFERED=1
# Dump a native traceback on SIGSEGV / SIGFPE / SIGBUS so C/CUDA-level
# crashes surface their origin instead of just "Floating point exception".
export PYTHONFAULTHANDLER=1

set -e -x -o pipefail

cd "${BRIDGEVLA_ROOT}"

python "${PRETRAIN_DIR}/pretrain.py" \
    --branches 3 \
    --config_path "${CONFIG_PATH}" \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --n_samples_per_task "${N_SAMPLES_PER_TASK}" \
    "$@" 2>&1 | tee -a "${LOG_FILE}"
