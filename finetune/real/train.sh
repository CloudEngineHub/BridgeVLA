#!/usr/bin/env bash
# Real-robot offline trainer launcher. Mirrors pretrain/pretrain.sh's SwanLab switch + shell-level tee.
#
# Usage: bash finetune/real/train.sh [extra args forwarded to train.py]
#
# Flags handled here (everything else is forwarded verbatim to train.py, so --freeze_epochs / --epochs /
# --max_iter all work from the shell):
#   --temporal_memory <bool>  memory ablation: stage-1 temporal memory (the two neighbouring frames + anchor)
#   --spatial_memory <bool>   memory ablation: stage-2 local anchor
# Bools accept true/false/1/0/yes/no/on/off, and the CLI wins over YAML. Disabling either appends
# _no_temporal_mem / _no_spatial_mem / _no_mem to swanlab_run and the run directory, keeping the shell tee
# aligned with train.py's checkpoint directory. The eval side must declare the same switches (the flask
# server's MEMORY_TEMPORAL/MEMORY_SPATIAL) or loading the ckpt aborts on a mismatch with its mvt_cfg.yaml.
#
# Env-var toggles (all optional):
#   DEBUG=true              single-GPU debug path (requires WORLD_SIZE=1, RESOURCE_GPU=1)
#   VISUALIZE=0             turn off the start-of-epoch viz (on by default)
#   USE_PRETRAIN / PRETRAIN_PATH   enable + locate the pretrain warm start (opt-in)
#   USE_RESUME  / RESUME_PATH      enable + locate a checkpoint to resume from (opt-in)
#   SWANLAB_UPLOAD          "cloud" (needs SWANLAB_API_KEY) or "offline" (default; "local" is a legacy alias)
#   CLIP_CHECKPOINT_DIR     override the local CLIP RN50.pt directory
#   REAL_TRAIN_CONDA_ENV    conda env to run in (default bridgevla_plus_real_train,
#                           built by finetune/real/install_real_train.sh)
#
# Examples:
#   DEBUG=true RESOURCE_GPU=1 bash finetune/real/train.sh --debug --epochs 1 --max_iter 2
#   SWANLAB_UPLOAD=cloud bash finetune/real/train.sh
#   bash finetune/real/train.sh --temporal_memory false --spatial_memory false  # full memory ablation

# ---- Paths and environment: all derived from the repository root, no machine-specific config ----
FINETUNE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # <repo>/finetune
BRIDGEVLA_ROOT="$(dirname "${FINETUNE_DIR}")"                      # <repo>
export BRIDGEVLA_DATA_ROOT="${BRIDGEVLA_DATA_ROOT:-${BRIDGEVLA_ROOT}/data/bridgevla_data}"
export BRIDGEVLA_LOG_DIR="${BRIDGEVLA_LOG_DIR:-${BRIDGEVLA_DATA_ROOT}/logs}"
export BRIDGEVLA_CKPT_ROOT="${BRIDGEVLA_CKPT_ROOT:-${BRIDGEVLA_ROOT}/data/bridgevla_ckpt}"
export PALIGEMMA_PATH="${PALIGEMMA_PATH:-${BRIDGEVLA_CKPT_ROOT}/paligemma-3b-pt-224}"
export CLIP_CACHE_DIR="${CLIP_CACHE_DIR:-${BRIDGEVLA_CKPT_ROOT}/clip}"
export CLIP_CHECKPOINT_DIR="${CLIP_CHECKPOINT_DIR:-${BRIDGEVLA_CKPT_ROOT}/clip}"
export HF_HOME="${HF_HOME:-${BRIDGEVLA_ROOT}/.cache/hf}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
# Log root of the real pipeline (a separate tree from sim's BRIDGEVLA_LOG_DIR) and the robot data root.
export BRIDGEVLA_REAL_LOG_DIR="${BRIDGEVLA_REAL_LOG_DIR:-${BRIDGEVLA_DATA_ROOT}/logs_real}"
export REAL_DATA_ROOT="${REAL_DATA_ROOT:-${BRIDGEVLA_DATA_ROOT}/Real}"
REAL_DIR="${FINETUNE_DIR}/real"

# conda (CONDA_BASE overridable, auto-detected by default)
if [ -z "${CONDA_BASE:-}" ]; then
    CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    [ -z "${CONDA_BASE}" ] && [ -d "${HOME}/miniconda3" ] && CONDA_BASE="${HOME}/miniconda3"
    [ -z "${CONDA_BASE}" ] && [ -d "${HOME}/anaconda3" ] && CONDA_BASE="${HOME}/anaconda3"
fi
# The default env is built by finetune/real/install_real_train.sh. Its packages are a strict subset
# of the GemBench env with identical pins, so REAL_TRAIN_CONDA_ENV=bridgevla_plus_gembench also works.
source "${CONDA_BASE}/bin/activate" "${REAL_TRAIN_CONDA_ENV:-bridgevla_plus_real_train}"

export PYTHONPATH="${FINETUNE_DIR}:${FINETUNE_DIR}/bridgevla/libs/point-renderer:${FINETUNE_DIR}/bridgevla/libs/peract_colab:${FINETUNE_DIR}/bridgevla/libs/YARR:${FINETUNE_DIR}/GemBench:${PYTHONPATH:-}"

# GPU selection: respect an exported CUDA_VISIBLE_DEVICES, otherwise leave it unset (a hard-coded list
# goes out of range on single-GPU vGPU environments).

# ---- SwanLab one-line switch (aligned with RLBench/train.sh) ----
# "cloud" (default) uploads to swanlab.cn and needs SWANLAB_API_KEY; if init fails train.py falls back to
# offline with logs in <run dir>/swanlog. "offline" saves locally only. Note that swanlab's "local" mode
# means a self-hosted swanboard server, not local files — use "offline" for that.
SWANLAB_UPLOAD="${SWANLAB_UPLOAD:-offline}"
[ "${SWANLAB_UPLOAD}" = "local" ] && SWANLAB_UPLOAD="offline"
export SWANLAB_MODE="${SWANLAB_UPLOAD}"
# train.py creates the swanlog directory under the run directory (<log_dir>/swanlog); a globally shared
# SWANLAB_LOG_DIR is no longer exported, but you can still export one to override.
if [ "${SWANLAB_UPLOAD}" = "cloud" ]; then
    export SWANLAB_API_KEY="${SWANLAB_API_KEY:?SWANLAB_UPLOAD=cloud requires exporting SWANLAB_API_KEY=<your key> first}"
else
    unset SWANLAB_API_KEY
fi

# 7_20_real_updated = the first 6 collected tasks + 13 new ones (5 put_shelf, 8 put_drawer), i.e. 19 tasks
# / 5 categories / 479 episodes. It is the clean version of 7_19_real, with the four episodes missing
# zed_depth/5.png dropped at index time, so all 479 on disk are usable. For the data mixture see
# task_sampling in real_config.yaml (group_mode=category, alpha=0.3); rank0 prints the nested
# category->variant raw% / eff% table at startup — trust that table.
DATA_FOLDER="${DATA_FOLDER:-${REAL_DATA_ROOT}/7_20_real_updated}"

# ---- Pretrain warm-start (mirrors GemBench/train.sh) ----
# USE_PRETRAIN=true + PRETRAIN_PATH=..., or --pretrain_path on the CLI; USE_RESUME takes priority.
# Defaults to BridgeVLA's released grounding pretrain (an HF sharded-safetensors directory); point it at
# <run>/pretrain_epoch_<N>.pth for your own run — mvt_single accepts either layout (README §3).
USE_PRETRAIN="${USE_PRETRAIN:-true}"
PRETRAIN_PATH="${PRETRAIN_PATH:-${BRIDGEVLA_CKPT_ROOT}/pretrain}"

# ---- Resume from checkpoint (mirrors GemBench/train.sh) ----
# USE_RESUME=true + RESUME_PATH=..., or --resume_path on the CLI. Restores model + optimizer + LR-warmup
# step + epoch and continues in the checkpoint's OWN run folder. Takes PRIORITY over the pretrain warm
# start; --no-resume force-disables it for one run.
USE_RESUME="${USE_RESUME:-false}"
RESUME_PATH="${RESUME_PATH:-}"

# Normalise bools -> the Python literals True/False (yacs merge_from_list uses literal_eval); invalid values abort.
to_pybool() {
  case "${1,,}" in
    1|true|yes|on|t)  echo "True" ;;
    0|false|no|off|f) echo "False" ;;
    *) echo -e "\033[31m[real/train.sh] invalid bool value: '$1' (expected true/false)\033[0m" >&2; exit 2 ;;
  esac
}

RESUME_ARGS=()
PRETRAIN_ARGS=()
FORWARD_ARGS=()
# Memory ablation switches: non-empty only when given on the CLI. Turned into --exp_cfg_opts YAML
# overrides, which also drive the run-directory ablation suffix.
CLI_TEMPORAL_MEM=""
CLI_SPATIAL_MEM=""
_want_resume_val=false
_want_pretrain_val=false
_want_tmem_val=false
_want_smem_val=false
for a in "$@"; do
    if [[ "$_want_resume_val" == "true" ]]; then
        RESUME_PATH="$a"; _want_resume_val=false; continue
    fi
    if [[ "$_want_pretrain_val" == "true" ]]; then
        PRETRAIN_PATH="$a"; USE_PRETRAIN=true; _want_pretrain_val=false; continue
    fi
    if [[ "$_want_tmem_val" == "true" ]]; then
        CLI_TEMPORAL_MEM="$(to_pybool "$a")" || exit 2
        _want_tmem_val=false; continue
    fi
    if [[ "$_want_smem_val" == "true" ]]; then
        CLI_SPATIAL_MEM="$(to_pybool "$a")" || exit 2
        _want_smem_val=false; continue
    fi
    case "$a" in
        --no-pretrain)   USE_PRETRAIN=false ;;
        --no-resume)     USE_RESUME=false ;;
        --resume_path)   USE_RESUME=true; _want_resume_val=true ;;
        --resume_path=*) USE_RESUME=true; RESUME_PATH="${a#--resume_path=}" ;;
        --pretrain_path) USE_PRETRAIN=true; _want_pretrain_val=true ;;
        --pretrain_path=*) USE_PRETRAIN=true; PRETRAIN_PATH="${a#--pretrain_path=}" ;;
        --temporal_memory)    _want_tmem_val=true ;;
        --temporal_memory=*)  CLI_TEMPORAL_MEM="$(to_pybool "${a#--temporal_memory=}")" || exit 2 ;;
        --spatial_memory)     _want_smem_val=true ;;
        --spatial_memory=*)   CLI_SPATIAL_MEM="$(to_pybool "${a#--spatial_memory=}")" || exit 2 ;;
        *)               FORWARD_ARGS+=("$a") ;;
    esac
done

if [[ "$USE_RESUME" == "true" && -n "${RESUME_PATH}" ]]; then
    RESUME_ARGS+=(--resume_path "${RESUME_PATH}")
fi
# Pretrain warm-start only when USE_PRETRAIN=true and not resuming (--pretrain_path / PRETRAIN_PATH).
if [[ "$USE_RESUME" != "true" || -z "${RESUME_PATH}" ]]; then
    if [[ "$USE_PRETRAIN" == "true" ]]; then
        # Fail early rather than letting train.py initialise and then raise FileNotFoundError from
        # torch.load. -e not -f: the checkpoint may be a .pth or an HF sharded-safetensors directory.
        if [ ! -e "${PRETRAIN_PATH}" ]; then
            echo "[real/train.sh] ERROR: pretrain ckpt not found:" >&2
            echo "  ${PRETRAIN_PATH}" >&2
            echo "  Fix: export PRETRAIN_PATH=/abs/path, or pass --no-pretrain to cold-start." >&2
            exit 1
        fi
        PRETRAIN_ARGS+=(--load_pretrain --pretrain_path "${PRETRAIN_PATH}")
    fi
fi

# ---- Start-of-epoch visualization: VISUALIZE=1 -> --visualize (default), 0 -> --no-visualize.
# An explicit CLI --visualize / --no-visualize always wins. ----
VISUALIZE="${VISUALIZE:-1}"
VIZ_ARGS=()
if [[ " $* " != *" --visualize "* && " $* " != *" --no-visualize "* ]]; then
    if [[ "${VISUALIZE}" == "0" ]]; then
        VIZ_ARGS+=(--no-visualize)
    else
        VIZ_ARGS+=(--visualize)
    fi
fi
# ---- GPU visibility: an exported CUDA_VISIBLE_DEVICES always wins; otherwise it is left unset for the
# container/scheduler and the torch probe below (a hard-coded list breaks single-GPU vGPU environments). ----

export MLP_WORKER_NUM=${WORLD_SIZE:-1}
# ---- nproc_per_node ----
# Must be an integer and <= the GPUs PyTorch actually sees. RESOURCE_GPU cannot be used directly (some
# schedulers set it to a float string like "1.00", which torchrun rejects), and counting commas in
# CUDA_VISIBLE_DEVICES is not enough (the container / Orion vGPU may narrow the set further). So
# torch.cuda.device_count() is the authoritative bound, and RESOURCE_GPU is only an explicit override under it.
_CUDA_PROBE_OUTPUT=$(python - <<'PY' 2>&1
import time
import torch

count = 0
for _ in range(5):
    count = torch.cuda.device_count()
    if count > 0:
        break
    time.sleep(2)
print(f"__TORCH_CUDA_DEVICE_COUNT__={count}")
PY
)
_TORCH_CUDA_DEVICE_COUNT=$(printf '%s\n' "${_CUDA_PROBE_OUTPUT}" | sed -n 's/^__TORCH_CUDA_DEVICE_COUNT__=//p' | tail -n 1)
if ! [[ "${_TORCH_CUDA_DEVICE_COUNT}" =~ ^[0-9]+$ ]] || [ "${_TORCH_CUDA_DEVICE_COUNT}" -lt 1 ]; then
    echo "[real/train.sh] ERROR: GPU probe failed — either PyTorch sees no CUDA device," >&2
    echo "  or 'python' has no torch (conda env not activated?)." >&2
    echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}  python=$(command -v python)" >&2
    echo "  Raw probe output:" >&2
    printf '%s\n' "${_CUDA_PROBE_OUTPUT}" >&2
    exit 1
fi
# "1.00" -> 1; a non-zero fractional part / non-numeric / <1 -> empty string (falls back to the probed value).
_RESOURCE_GPU_INT=""
if [ -n "${RESOURCE_GPU:-}" ]; then
    _RESOURCE_GPU_INT=$(awk -v v="${RESOURCE_GPU}" 'BEGIN{ if (v+0 == int(v+0) && v+0 >= 1) printf "%d", v+0 }')
    if [ -z "${_RESOURCE_GPU_INT}" ]; then
        echo "[real/train.sh] WARNING: ignoring unusable RESOURCE_GPU='${RESOURCE_GPU}'; using PyTorch-visible count ${_TORCH_CUDA_DEVICE_COUNT}." >&2
    elif [ "${_RESOURCE_GPU_INT}" -gt "${_TORCH_CUDA_DEVICE_COUNT}" ]; then
        echo "[real/train.sh] WARNING: RESOURCE_GPU=${RESOURCE_GPU} exceeds PyTorch-visible ${_TORCH_CUDA_DEVICE_COUNT} device(s); clamping." >&2
        _RESOURCE_GPU_INT=""
    fi
fi
MLP_WORKER_GPU="${_RESOURCE_GPU_INT:-${_TORCH_CUDA_DEVICE_COUNT}}"
export MLP_WORKER_GPU
echo "[real/train.sh] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}; torch_cuda_device_count=${_TORCH_CUDA_DEVICE_COUNT}; nproc_per_node=${MLP_WORKER_GPU}"
export MLP_ROLE_INDEX=${RANK:-0}
export MLP_WORKER_0_HOST=${MASTER_ADDR:-localhost}
export MLP_WORKER_0_PORT=${MASTER_PORT:-29503}

# --- Terminal log capture (mirrors pretrain.sh) ---
# Pre-compute the run dir in shell so torchrun's stdout/stderr tees into the SAME folder train.py writes
# checkpoints to; the stamp is shared via REAL_RUN_STAMP. On multi-node runs each node tees to its own
# train_node{N}.log to avoid clobbering on a shared FS.
CONFIG_FILE="${REAL_DIR}/configs/real_config.yaml"
# The log root comes from BRIDGEVLA_REAL_LOG_DIR (derived at the top) and is passed to train.py via
# --log_dir below, keeping both sides consistent. The YAML value is only a fallback.
LOG_DIR_CFG="${BRIDGEVLA_REAL_LOG_DIR}"
SWANLAB_RUN_CFG="$(python -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["swanlab_run"])' "${CONFIG_FILE}")"
# If the caller overrides freeze_epochs on the CLI, mirror train.py's "_freeze{N}" suffix so the tee log
# lands in the same run dir.
FREEZE_EPOCHS_OVERRIDE=""
prev=""
for a in "$@"; do
    if [[ "$prev" == "--freeze_epochs" ]]; then
        FREEZE_EPOCHS_OVERRIDE="$a"
    elif [[ "$a" == --freeze_epochs=* ]]; then
        FREEZE_EPOCHS_OVERRIDE="${a#--freeze_epochs=}"
    fi
    prev="$a"
done
if [[ -n "${FREEZE_EPOCHS_OVERRIDE}" ]]; then
    SWANLAB_RUN_CFG="${SWANLAB_RUN_CFG}_freeze${FREEZE_EPOCHS_OVERRIDE}"
fi
# Memory-ablation suffix, mirroring train.py:memory_ablation_suffix so the shell tee lands in the run dir
# Python creates. swanlab_run is deliberately NOT overridden here: real's get_logdir appends the suffix
# itself after the _freeze{N} one, and overriding it via --exp_cfg_opts would fight that ordering.
export REAL_CLI_TEMPORAL_MEM="${CLI_TEMPORAL_MEM}"
export REAL_CLI_SPATIAL_MEM="${CLI_SPATIAL_MEM}"
MEM_SUFFIX="$(python - "${CONFIG_FILE}" <<'PY'
import os, sys, yaml
mem = (yaml.safe_load(open(sys.argv[1])) or {}).get("memory", {}) or {}
def _ov(envname, cur):
    # CLI override (True/False) > YAML; empty env -> keep YAML value.
    v = os.environ.get(envname, "")
    return cur if v == "" else (v == "True")
suffix = ""
if bool(mem.get("enabled", False)):
    t = _ov("REAL_CLI_TEMPORAL_MEM", bool(mem.get("temporal_memory", True)))
    s = _ov("REAL_CLI_SPATIAL_MEM", bool(mem.get("spatial_memory", True)))
    if not t and not s:
        suffix = "_no_mem"
    elif not t:
        suffix = "_no_temporal_mem"
    elif not s:
        suffix = "_no_spatial_mem"
print(suffix)
PY
)"
SWANLAB_RUN_CFG="${SWANLAB_RUN_CFG}${MEM_SUFFIX}"

# Assemble --exp_cfg_opts (a space-separated key value string for yacs merge_from_list); the YAML is only
# overridden when the memory switches were given on the CLI.
EXP_CFG_OPTS_ARGS=()
EXP_CFG_OPTS_STR=""
if [[ -n "${CLI_TEMPORAL_MEM}" ]]; then
    EXP_CFG_OPTS_STR="memory.temporal_memory ${CLI_TEMPORAL_MEM}"
fi
if [[ -n "${CLI_SPATIAL_MEM}" ]]; then
    EXP_CFG_OPTS_STR="${EXP_CFG_OPTS_STR:+${EXP_CFG_OPTS_STR} }memory.spatial_memory ${CLI_SPATIAL_MEM}"
fi
if [[ -n "${EXP_CFG_OPTS_STR}" ]]; then
    EXP_CFG_OPTS_ARGS=(--exp_cfg_opts "${EXP_CFG_OPTS_STR}")
fi
export REAL_RUN_STAMP="${REAL_RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
if [[ "$USE_RESUME" == "true" && -n "${RESUME_PATH}" ]]; then
    # Resume: the shell tee follows the checkpoint's own run folder, matching train.py pinning log_dir to dirname(resume_path).
    RUN_DIR="$(cd "$(dirname "${RESUME_PATH}")" && pwd)"
    echo "[real/train.sh] resume: reusing original run dir ${RUN_DIR}"
else
    RUN_DIR="${LOG_DIR_CFG}/train/${SWANLAB_RUN_CFG}_${REAL_RUN_STAMP}"
fi
mkdir -p "${RUN_DIR}"
LOG_FILE="${RUN_DIR}/train_node${MLP_ROLE_INDEX}.log"
echo "[real/train.sh] run dir: ${RUN_DIR}"
echo "[real/train.sh] logging torchrun stdout/stderr to: ${LOG_FILE}"
echo "[real/train.sh] memory: temporal=${CLI_TEMPORAL_MEM:-<yaml>}  spatial=${CLI_SPATIAL_MEM:-<yaml>}"
# Force line-buffered Python so tee flushes per line, not per block.
export PYTHONUNBUFFERED=1

# pipefail: propagate torchrun's exit status through the tee pipeline so set -e still aborts on failure.
set -e -x -o pipefail

cd "${FINETUNE_DIR}"

if [[ "${DEBUG:-false}" == "true" && "${MLP_WORKER_NUM}" == "1" && "${MLP_WORKER_GPU}" == "1" ]]; then
    python real/train.py \
        --exp_cfg_path real/configs/real_config.yaml \
        --mvt_cfg_path real/configs/mvt_cfg.yaml \
        --log_dir "${BRIDGEVLA_REAL_LOG_DIR}" \
        --data_folder "${DATA_FOLDER}" \
        "${EXP_CFG_OPTS_ARGS[@]}" \
        "${PRETRAIN_ARGS[@]}" \
        "${RESUME_ARGS[@]}" \
        "${VIZ_ARGS[@]}" \
        "${FORWARD_ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"
else
    torchrun \
        --nnodes=${MLP_WORKER_NUM} \
        --node_rank=${MLP_ROLE_INDEX} \
        --nproc_per_node=${MLP_WORKER_GPU} \
        --master_addr=${MLP_WORKER_0_HOST} \
        --master_port=${MLP_WORKER_0_PORT} \
        real/train.py \
        --exp_cfg_path real/configs/real_config.yaml \
        --mvt_cfg_path real/configs/mvt_cfg.yaml \
        --log_dir "${BRIDGEVLA_REAL_LOG_DIR}" \
        --data_folder "${DATA_FOLDER}" \
        "${EXP_CFG_OPTS_ARGS[@]}" \
        "${PRETRAIN_ARGS[@]}" \
        "${RESUME_ARGS[@]}" \
        "${VIZ_ARGS[@]}" \
        "${FORWARD_ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"
fi
