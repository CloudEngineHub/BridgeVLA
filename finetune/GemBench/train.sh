#!/usr/bin/env bash

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

# conda (CONDA_BASE overridable, auto-detected by default)
if [ -z "${CONDA_BASE:-}" ]; then
    CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    [ -z "${CONDA_BASE}" ] && [ -d "${HOME}/miniconda3" ] && CONDA_BASE="${HOME}/miniconda3"
    [ -z "${CONDA_BASE}" ] && [ -d "${HOME}/anaconda3" ] && CONDA_BASE="${HOME}/anaconda3"
fi
source "${CONDA_BASE}/bin/activate" "${GEMBENCH_CONDA_ENV:-bridgevla_plus_gembench}"

# GPU selection: respect an exported CUDA_VISIBLE_DEVICES, otherwise use every detected GPU. Hard-coding a
# list breaks single-GPU containers (rank>=1 fails torch.cuda.set_device with "invalid device ordinal").
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    _NGPU_DETECTED=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
    [ "${_NGPU_DETECTED}" -lt 1 ] && _NGPU_DETECTED=1
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((_NGPU_DETECTED - 1)))
fi
_CUDA_VISIBLE_DEVICES_COUNT=$(awk -F, '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")
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
    echo "[train.sh] ERROR: PyTorch sees no CUDA device. CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
    printf '%s\n' "${_CUDA_PROBE_OUTPUT}" >&2
    exit 1
fi
if [ "${_CUDA_VISIBLE_DEVICES_COUNT}" -ne "${_TORCH_CUDA_DEVICE_COUNT}" ]; then
    echo "[train.sh] WARNING: CUDA_VISIBLE_DEVICES has ${_CUDA_VISIBLE_DEVICES_COUNT} entries (${CUDA_VISIBLE_DEVICES}), but PyTorch sees ${_TORCH_CUDA_DEVICE_COUNT} CUDA device(s). Using PyTorch-visible count for torchrun." >&2
fi

# The five prefixes below replace the `pip install -e .` previously done inside the conda env, so every
# import resolves against the current tree: bridgevla, point_renderer, peract_colab, yarr, genrobo3d.
export PYTHONPATH="${FINETUNE_DIR}:${FINETUNE_DIR}/bridgevla/libs/point-renderer:${FINETUNE_DIR}/bridgevla/libs/peract_colab:${FINETUNE_DIR}/bridgevla/libs/YARR:${FINETUNE_DIR}/GemBench:${FINETUNE_DIR}/bridgevla/libs/PyRep:${FINETUNE_DIR}/bridgevla/libs/RLBench:${PYTHONPATH:-}"
# CLIP_* / PALIGEMMA_PATH / HF_HOME / HF offline flags are already derived at the top of this script.
export COPPELIASIM_ROOT="${FINETUNE_DIR}/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM=offscreen
# ---- CUDA caching allocator: suppress the steady VRAM "creep" during training ----
# Allocation shapes change every step: move_pc_in_bound crops each sample to a variable-length point cloud,
# and processor(padding="longest") makes the PaliGemma sequence length follow the batch's longest prompt.
# The default allocator pins segments sized for the worst-case combination and cannot reuse the smaller
# holes, so reserved memory grows monotonically (during the frozen Stage 1 too). expandable_segments
# absorbs the jitter with resizable segments. Pure allocator policy — no change to numerics or model logic.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# ---- SwanLab one-line switch ----
# "cloud" uploads, "offline" (default) saves locally. Note that swanlab's "local" mode means a self-hosted
# swanboard server, not local files. View offline logs: `swanlab watch -l <log_dir>/swanlog`
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
export GEMBENCH_DATA_FOLDER="${GEMBENCH_DATA_FOLDER:-${BRIDGEVLA_DATA_ROOT}/GemBench/train_dataset}"
if [ ! -f "${GEMBENCH_DATA_FOLDER}/taskvars_instructions_new.json" ]; then
    echo "[train.sh] ERROR: missing ${GEMBENCH_DATA_FOLDER}/taskvars_instructions_new.json" >&2
    echo "[train.sh] Set GEMBENCH_DATA_FOLDER to the real train_dataset directory." >&2
    exit 1
fi

# Cluster env vars (same pattern as memoryBench/RMBench_vla train.sh). nproc_per_node must be an integer
# equal to the GPUs PyTorch actually sees: not RESOURCE_GPU (a float string like "1.00" is rejected by
# torchrun) and not a comma count of CUDA_VISIBLE_DEVICES (the container may narrow visibility further).
export MLP_WORKER_NUM=${WORLD_SIZE:-1}
export MLP_WORKER_GPU=${_TORCH_CUDA_DEVICE_COUNT}
export MLP_ROLE_INDEX=${RANK:-0}
export MLP_WORKER_0_HOST=${MASTER_ADDR:-localhost}
export MLP_WORKER_0_PORT=${MASTER_PORT:-29606}
echo "[train.sh] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}; torch_cuda_device_count=${_TORCH_CUDA_DEVICE_COUNT}; nproc_per_node=${MLP_WORKER_GPU}"

# ---- Pretrain warm-start ----
# USE_PRETRAIN=true warm-starts from PRETRAIN_PATH; false cold-starts with PaliGemma base weights only.
# Defaults to BridgeVLA's released grounding-pretrain checkpoint (an HF sharded-safetensors directory);
# point PRETRAIN_PATH at <run>/pretrain_epoch_<N>.pth for your own run — mvt_single accepts either layout
# (README §3). Append `--no-pretrain` to disable for a single run.
USE_PRETRAIN=true
PRETRAIN_PATH="${PRETRAIN_PATH:-${BRIDGEVLA_CKPT_ROOT}/pretrain}"

# ---- Resume from checkpoint ----
# USE_RESUME=true + RESUME_PATH restores model + optimizer + LR-warmup step + epoch and continues in the
# checkpoint's OWN run folder (original timestamp preserved); scheduling config still comes from YAML.
# Resume takes PRIORITY over the pretrain warm start. One-off overrides: `--no-resume`, or
# `--resume_path /path/to/model_last.pth`.
USE_RESUME=false
RESUME_PATH="${RESUME_PATH:-}"

# ---- CLI parsing ----
# Handled here: --no-pretrain / --no-resume / --resume_path, plus --temporal_memory <bool> /
# --spatial_memory <bool> (the = form works too). The memory switches become --exp_cfg_opts YAML overrides
# and drive the ablation suffix of swanlab_run / the run directory. Everything else goes to train.py verbatim.

# Normalise bools -> the Python literals True/False (yacs merge_from_list uses literal_eval); invalid values abort.
to_pybool() {
  case "${1,,}" in
    1|true|yes|on|t)  echo "True" ;;
    0|false|no|off|f) echo "False" ;;
    *) echo -e "\033[31m[train.sh] invalid bool value: '$1' (expected true/false)\033[0m" >&2; exit 2 ;;
  esac
}

PRETRAIN_ARGS=()
RESUME_ARGS=()
FORWARD_ARGS=()
CLI_TEMPORAL_MEM=""
CLI_SPATIAL_MEM=""
_want_resume_val=false
_want_tmem_val=false
_want_smem_val=false
for a in "$@"; do
    if [[ "$_want_resume_val" == "true" ]]; then
        RESUME_PATH="$a"
        _want_resume_val=false
        continue
    fi
    if [[ "$_want_tmem_val" == "true" ]]; then
        CLI_TEMPORAL_MEM="$(to_pybool "$a")" || exit 2
        _want_tmem_val=false
        continue
    fi
    if [[ "$_want_smem_val" == "true" ]]; then
        CLI_SPATIAL_MEM="$(to_pybool "$a")" || exit 2
        _want_smem_val=false
        continue
    fi
    case "$a" in
        --no-pretrain)   USE_PRETRAIN=false ;;
        --no-resume)     USE_RESUME=false ;;
        --resume_path)   USE_RESUME=true; _want_resume_val=true ;;
        --resume_path=*) USE_RESUME=true; RESUME_PATH="${a#--resume_path=}" ;;
        --temporal_memory)    _want_tmem_val=true ;;
        --temporal_memory=*)  CLI_TEMPORAL_MEM="$(to_pybool "${a#--temporal_memory=}")" || exit 2 ;;
        --spatial_memory)     _want_smem_val=true ;;
        --spatial_memory=*)   CLI_SPATIAL_MEM="$(to_pybool "${a#--spatial_memory=}")" || exit 2 ;;
        *)               FORWARD_ARGS+=("$a") ;;
    esac
done
if [[ "$USE_RESUME" == "true" ]] && [[ -n "${RESUME_PATH}" ]]; then
    USE_PRETRAIN=false   # resume wins over pretrain warm-start
    RESUME_ARGS+=(--resume_path "${RESUME_PATH}")
fi
if [[ "$USE_PRETRAIN" == "true" ]]; then
    # -e not -f: the pretrain checkpoint may be a .pth or an HF sharded-safetensors directory.
    if [ ! -e "${PRETRAIN_PATH}" ]; then
        echo "[train.sh] ERROR: pretrain checkpoint not found:" >&2
        echo "  ${PRETRAIN_PATH}" >&2
        echo "  Fix: put BridgeVLA's released pretrain there, export" >&2
        echo "       PRETRAIN_PATH=/abs/path, or pass --no-pretrain to cold-start." >&2
        exit 1
    fi
    PRETRAIN_ARGS+=(--load_pretrain --pretrain_path "${PRETRAIN_PATH}")
fi

# --- Terminal log capture ---
# Pre-compute the run dir in shell so torchrun's stdout/stderr tees into the SAME folder train.py writes
# checkpoints to; the stamp is shared via GEMBENCH_RUN_STAMP. On multi-node runs each node tees to its own
# train_node{N}.log to avoid clobbering on a shared FS.
CONFIG_FILE="${FINETUNE_DIR}/GemBench/configs/gembench_config.yaml"
# The log root is BRIDGEVLA_LOG_DIR (derived at the top), passed to train.py via --log_dir; the shell tee uses the same root.
LOG_DIR_CFG="${BRIDGEVLA_LOG_DIR}"
# swanlab_run + memory ablation suffix, kept in sync with train.py:memory_ablation_suffix so the shell tee
# and train.py land in the same run directory. CLI switches win over YAML; an empty env var keeps the YAML value.
export GEMBENCH_CLI_TEMPORAL_MEM="${CLI_TEMPORAL_MEM}"
export GEMBENCH_CLI_SPATIAL_MEM="${CLI_SPATIAL_MEM}"
read -r SWANLAB_RUN_CFG SWANLAB_RUN_BASE <<< "$(python - "${CONFIG_FILE}" <<'PY'
import os, sys, yaml

cfg = yaml.safe_load(open(sys.argv[1]))
base = str(cfg.get("swanlab_run", "run"))
run = base

mem = cfg.get("memory", {}) or {}
def _ov(envname, cur):
    # CLI override (True/False) > YAML; empty env -> keep YAML value.
    v = os.environ.get(envname, "")
    return cur if v == "" else (v == "True")
mem_suffix = ""
if bool(mem.get("enabled", False)):
    t = _ov("GEMBENCH_CLI_TEMPORAL_MEM", bool(mem.get("temporal_memory", True)))
    s = _ov("GEMBENCH_CLI_SPATIAL_MEM", bool(mem.get("spatial_memory", True)))
    if not t and not s:
        mem_suffix = "_no_mem"
    elif not t:
        mem_suffix = "_no_temporal_mem"
    elif not s:
        mem_suffix = "_no_spatial_mem"
if mem_suffix and not run.endswith(mem_suffix):
    run = f"{run}{mem_suffix}"

print(run, base)
PY
)"
# Assemble --exp_cfg_opts (a space-separated key value string for yacs merge_from_list): swanlab_run only
# when there is an ablation suffix, and the memory switches only when given explicitly on the CLI.
EXP_CFG_OPTS_ARGS=()
EXP_CFG_OPTS_STR=""
if [[ "${SWANLAB_RUN_CFG}" != "${SWANLAB_RUN_BASE}" ]]; then
    EXP_CFG_OPTS_STR="swanlab_run ${SWANLAB_RUN_CFG}"
fi
if [[ -n "${CLI_TEMPORAL_MEM}" ]]; then
    EXP_CFG_OPTS_STR="${EXP_CFG_OPTS_STR:+${EXP_CFG_OPTS_STR} }memory.temporal_memory ${CLI_TEMPORAL_MEM}"
fi
if [[ -n "${CLI_SPATIAL_MEM}" ]]; then
    EXP_CFG_OPTS_STR="${EXP_CFG_OPTS_STR:+${EXP_CFG_OPTS_STR} }memory.spatial_memory ${CLI_SPATIAL_MEM}"
fi
if [[ -n "${EXP_CFG_OPTS_STR}" ]]; then
    EXP_CFG_OPTS_ARGS=(--exp_cfg_opts "${EXP_CFG_OPTS_STR}")
fi
echo "[train.sh] memory: temporal=${CLI_TEMPORAL_MEM:-<yaml>}  spatial=${CLI_SPATIAL_MEM:-<yaml>}"
export GEMBENCH_RUN_STAMP="${GEMBENCH_RUN_STAMP:-$(date +%m_%d_%H_%M)}"
if [[ "$USE_RESUME" == "true" ]] && [[ -n "${RESUME_PATH}" ]]; then
    # Resume: the shell tee follows the checkpoint's own run folder, matching train.py pinning log_dir to dirname(resume_path).
    RUN_DIR="$(cd "$(dirname "${RESUME_PATH}")" && pwd)"
    echo "[train.sh] resume: reusing original run dir ${RUN_DIR}"
else
    RUN_DIR="${LOG_DIR_CFG}/train_gembench/${SWANLAB_RUN_CFG}_${GEMBENCH_RUN_STAMP}"
fi
mkdir -p "${RUN_DIR}"
LOG_FILE="${RUN_DIR}/train_node${MLP_ROLE_INDEX}.log"
echo "[train.sh] run dir: ${RUN_DIR}"
echo "[train.sh] logging torchrun stdout/stderr to: ${LOG_FILE}"
# Force line-buffered Python so tee flushes per line, not per block.
export PYTHONUNBUFFERED=1

# pipefail: propagate torchrun's exit status through the tee pipeline so set -e still aborts on failure.
set -e -x -o pipefail

cd "${FINETUNE_DIR}/GemBench"

torchrun \
    --nnodes=$MLP_WORKER_NUM \
    --node_rank=$MLP_ROLE_INDEX \
    --nproc_per_node=$MLP_WORKER_GPU \
    --master_addr=$MLP_WORKER_0_HOST \
    --master_port=$MLP_WORKER_0_PORT \
    train.py \
    --exp_cfg_path configs/gembench_config.yaml \
    --log_dir "${BRIDGEVLA_LOG_DIR}" \
    --data_folder "${GEMBENCH_DATA_FOLDER}" \
    --mvt_cfg_opts "img_aug_2 0.0" \
    "${EXP_CFG_OPTS_ARGS[@]}" \
    "${PRETRAIN_ARGS[@]}" \
    "${RESUME_ARGS[@]}" \
    "${FORWARD_ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"
