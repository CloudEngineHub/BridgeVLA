#!/usr/bin/env bash
#
# Usage examples:
#   bash train.sh --tasks press_button
#   bash train.sh                          # use the tasks from configs/rmbench_config.yaml
#   CUDA_VISIBLE_DEVICES=0 bash train.sh --tasks press_button
#   # memory ablation: switches given on the command line (higher priority than YAML)
#   bash train.sh --tasks press_button --temporal_memory false   # disable temporal memory (memory 1)
#   bash train.sh --tasks press_button --spatial_memory false    # disable spatial memory (memory 2)
#   bash train.sh --temporal_memory false --spatial_memory false # disable both
#   bash train.sh --tasks observe_and_pickup --epochs 1500     # override the YAML epochs
#   bash train.sh --tasks observe_and_pickup --save_every_n_epochs 10  # override the checkpoint interval
#   (bools accept true/false/1/0/yes/no/on/off; --temporal_memory=false also works)
#
# For a single task (from the CLI or the YAML) the run directory / SwanLab name is {swanlab_run}_{task}_{stamp},
# e.g. swanlab_run=6_24 with task=press_button -> 6_24_press_button_06_24_19_47.
# Disabling a memory appends the ablation suffix, e.g. ..._press_button_no_temporal_mem_{stamp}.
# CLI --tasks wins over the YAML tasks; --temporal_memory/--spatial_memory win over YAML;
# --epochs / --save_every_n_epochs win over the YAML fields of the same name.

# Paths and environment: all derived from the repository root, no machine-specific config;
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
# Real location of the RMBench data/assets (envs/_GLOBAL_CONFIGS.py reads these two variables too).
export RMBENCH_DATA_PATH="${RMBENCH_DATA_PATH:-${BRIDGEVLA_DATA_ROOT}/RMBench/data}"
export RMBENCH_ASSETS_PATH="${RMBENCH_ASSETS_PATH:-${BRIDGEVLA_DATA_ROOT}/RMBench/assets}"

# conda (CONDA_BASE overridable, auto-detected by default). Training runs in the bridgevla (gembench)
# env — it needs paligemma + point-renderer + torch, but not sapien/curobo (the data is already
# offline HDF5 keyframes).
if [ -z "${CONDA_BASE:-}" ]; then
    CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    [ -z "${CONDA_BASE}" ] && [ -d "${HOME}/miniconda3" ] && CONDA_BASE="${HOME}/miniconda3"
    [ -z "${CONDA_BASE}" ] && [ -d "${HOME}/anaconda3" ] && CONDA_BASE="${HOME}/anaconda3"
fi
source "${CONDA_BASE}/bin/activate" "${GEMBENCH_CONDA_ENV:-bridgevla_plus_gembench}"

# GPU selection: respect an exported CUDA_VISIBLE_DEVICES (e.g. `CUDA_VISIBLE_DEVICES=0,3
# bash train.sh`); otherwise default to every GPU detected on the machine. Hard-coding 0..7 would go
# out of range on single-GPU machines (some vGPU environments expose only one GPU).
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    _NGPU_DETECTED=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
    [ "${_NGPU_DETECTED}" -lt 1 ] && _NGPU_DETECTED=1
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((_NGPU_DETECTED - 1)))
fi
# Reduce CUDA caching-allocator fragmentation: bimanual training does several mixed-batch PaliGemma
# forwards per step (mvt1 trunk + one mvt2 trunk per arm + memory anchor/history), and expandable
# segments let physical pages grow and shrink on demand, cutting reserved memory noticeably. PyTorch >= 2.1.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# RMBench_vla comes first on PYTHONPATH so that `from utils.peract_utils_rmbench import ...` resolves
# to RMBench_vla/utils rather than the same-named directory in GemBench/memoryBench.
export PYTHONPATH="${FINETUNE_DIR}/RMBench_vla:${FINETUNE_DIR}:${FINETUNE_DIR}/bridgevla/libs/point-renderer:${FINETUNE_DIR}/bridgevla/libs/peract_colab:${FINETUNE_DIR}/bridgevla/libs/YARR:${FINETUNE_DIR}/bridgevla/libs/peract:${FINETUNE_DIR}/GemBench:${PYTHONPATH:-}"
# CLIP_* / PALIGEMMA_PATH / HF_HOME / HF offline flags are already derived at the top of this script.

# SwanLab toggle (same convention as memoryBench/train.sh).
SWANLAB_UPLOAD="${SWANLAB_UPLOAD:-offline}"
[ "${SWANLAB_UPLOAD}" = "local" ] && SWANLAB_UPLOAD="offline"
export SWANLAB_MODE="${SWANLAB_UPLOAD}"
if [ "${SWANLAB_UPLOAD}" = "cloud" ]; then
    export SWANLAB_API_KEY="${SWANLAB_API_KEY:?SWANLAB_UPLOAD=cloud requires exporting SWANLAB_API_KEY=<your key> first}"
else
    unset SWANLAB_API_KEY
fi

# RMBench keyframe data root (keyframe_data: colour-corrected keyframe-only HDF5, with depth +
# intrinsics/extrinsics + both-arm endpose; released as datasets/rmbench/keyframe_data — fetch with
# `bash scripts/download_checkpoints_hf.sh rmbench_data`). The older data_self stored R<->B swapped
# colours and is deprecated; the pre-rename name of this tree was data_self_convertRGB.
export RMBENCH_VLA_DATA_ROOT="${RMBENCH_VLA_DATA_ROOT:-${RMBENCH_DATA_PATH}/keyframe_data}"

# Fail fast on missing training data. keyframes/ must sit NEXT TO the data
# root (the dataset derives it as a sibling — see rmbench_dataset.py).
if [ ! -d "${RMBENCH_VLA_DATA_ROOT}" ] || [ ! -d "$(dirname "${RMBENCH_VLA_DATA_ROOT}")/keyframes" ]; then
    echo "[train.sh] ERROR: RMBench keyframe data missing:" >&2
    echo "[train.sh]   need ${RMBENCH_VLA_DATA_ROOT} AND $(dirname "${RMBENCH_VLA_DATA_ROOT}")/keyframes" >&2
    echo "[train.sh]   fetch both with: bash scripts/download_checkpoints_hf.sh rmbench_data" >&2
    echo "[train.sh]   (or regenerate: finetune/RMBench/script/extract_key_frames/extract_keyframes.py" >&2
    echo "[train.sh]    + finetune/RMBench/collect/batch/run_all_keyframe_depth.sh — needs raw demo_clean)" >&2
    exit 1
fi

# Cluster-launch shims (same as memoryBench/train.sh).
# nproc_per_node must be an integer and == the number of visible GPUs: count the commas in
# CUDA_VISIBLE_DEVICES rather than using RESOURCE_GPU (a float string like "1.00" here, which torchrun rejects).
export MLP_WORKER_NUM=${WORLD_SIZE:-1}
_NGPU=$(awk -F, '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")
export MLP_WORKER_GPU=${_NGPU}
export MLP_ROLE_INDEX=${RANK:-0}
export MLP_WORKER_0_HOST=${MASTER_ADDR:-localhost}
export MLP_WORKER_0_PORT=${MASTER_PORT:-29629}

# Pretrain warm start (optional; warm-starts from a single-arm memoryBench/GemBench ckpt).
# Defaults to the pre-training weights released with BridgeVLA (an HF directory); if you ran
# pretrain/pretrain.sh yourself, point PRETRAIN_PATH at <run>/pretrain_epoch_<N>.pth — both layouts work (see README §3).
USE_PRETRAIN=true
PRETRAIN_PATH="${PRETRAIN_PATH:-${BRIDGEVLA_CKPT_ROOT}/pretrain}"

# Resume from a checkpoint (optional; resuming takes priority over the pretrain warm start)
USE_RESUME=false
RESUME_PATH="${RESUME_PATH:-}"

# ---- CLI parsing ----
# This script handles: --no-pretrain / --no-resume / --resume_path
#   plus the memory ablation switches: --temporal_memory <bool> / --spatial_memory <bool>
#   plus --epochs <positive int> / --save_every_n_epochs <positive int>, which override
#     epochs / save_every_n_epochs in configs/rmbench_config.yaml
#   (the --temporal_memory=false / --epochs=1500 forms work too; bools accept true/false/...)
#   These switches are not passed through to train.py; they are turned into --exp_cfg_opts
#   overrides of the YAML and also drive the ablation suffix of swanlab_run / the run directory.
# Everything else (including --tasks) is forwarded to train.py verbatim; CLI --tasks wins over YAML.

# Normalise bools -> the Python literals True/False (yacs merge_from_list uses literal_eval, so
# they must be capitalised); an invalid value aborts with an error.
to_pybool() {
  case "${1,,}" in
    1|true|yes|on|t)  echo "True" ;;
    0|false|no|off|f) echo "False" ;;
    *) echo -e "\033[31m[train.sh] invalid bool value: '$1' (expected true/false)\033[0m" >&2; exit 2 ;;
  esac
}

# Positive-integer validation (epochs / save_every_n_epochs etc.); $2 is the field name used in the error message.
to_posint() {
  if [[ ! "$1" =~ ^[1-9][0-9]*$ ]]; then
    echo -e "\033[31m[train.sh] invalid ${2:-value}: '$1' (expected a positive integer)\033[0m" >&2
    exit 2
  fi
  echo "$1"
}

PRETRAIN_ARGS=()
RESUME_ARGS=()
FORWARD_ARGS=()
CLI_TASKS=()
CLI_TEMPORAL_MEM=""
CLI_SPATIAL_MEM=""
CLI_EPOCHS=""
CLI_SAVE_EVERY=""
_want_resume_val=false
_want_tmem_val=false
_want_smem_val=false
_want_epochs_val=false
_want_save_every_val=false
_collect_tasks=false
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
    if [[ "$_want_epochs_val" == "true" ]]; then
        CLI_EPOCHS="$(to_posint "$a" epochs)" || exit 2
        _want_epochs_val=false
        continue
    fi
    if [[ "$_want_save_every_val" == "true" ]]; then
        CLI_SAVE_EVERY="$(to_posint "$a" save_every_n_epochs)" || exit 2
        _want_save_every_val=false
        continue
    fi
    if [[ "$_collect_tasks" == "true" ]]; then
        if [[ "$a" == --* ]]; then
            _collect_tasks=false
        else
            CLI_TASKS+=("$a")
            FORWARD_ARGS+=("$a")
            continue
        fi
    fi
    case "$a" in
        --no-pretrain)   USE_PRETRAIN=false; continue ;;
        --no-resume)     USE_RESUME=false; continue ;;
        --resume_path)   USE_RESUME=true; _want_resume_val=true; continue ;;
        --resume_path=*) USE_RESUME=true; RESUME_PATH="${a#--resume_path=}"; continue ;;
        --temporal_memory)    _want_tmem_val=true; continue ;;
        --temporal_memory=*)  CLI_TEMPORAL_MEM="$(to_pybool "${a#--temporal_memory=}")" || exit 2; continue ;;
        --spatial_memory)     _want_smem_val=true; continue ;;
        --spatial_memory=*)   CLI_SPATIAL_MEM="$(to_pybool "${a#--spatial_memory=}")" || exit 2; continue ;;
        --epochs)             _want_epochs_val=true; continue ;;
        --epochs=*)           CLI_EPOCHS="$(to_posint "${a#--epochs=}" epochs)" || exit 2; continue ;;
        --save_every_n_epochs)   _want_save_every_val=true; continue ;;
        --save_every_n_epochs=*) CLI_SAVE_EVERY="$(to_posint "${a#--save_every_n_epochs=}" save_every_n_epochs)" || exit 2; continue ;;
        --tasks)
            _collect_tasks=true
            FORWARD_ARGS+=("$a")
            continue
            ;;
        --tasks=*)
            _raw="${a#--tasks=}"
            IFS=',' read -ra _parts <<< "${_raw// /,}"
            for _t in "${_parts[@]}"; do
                [[ -n "$_t" ]] && CLI_TASKS+=("$_t")
            done
            FORWARD_ARGS+=("$a")
            continue
            ;;
    esac
    FORWARD_ARGS+=("$a")
done
if [[ "$USE_RESUME" == "true" ]] && [[ -n "${RESUME_PATH}" ]]; then
    USE_PRETRAIN=false   # resuming takes priority over the pretrain warm start
    RESUME_ARGS+=(--resume_path "${RESUME_PATH}")
fi
if [[ "$USE_PRETRAIN" == "true" ]]; then
    # -e rather than -f: the pretrain weights may be a single .pth or a sharded HF directory.
    if [ ! -e "${PRETRAIN_PATH}" ]; then
        echo "[train.sh] ERROR: pretrain weights not found: ${PRETRAIN_PATH}" >&2
        echo "[train.sh]   put the BridgeVLA pre-training weights in place, or export PRETRAIN_PATH=<your own pretrain>," >&2
        echo "[train.sh]   or add --no-pretrain to cold start (expect a noticeable accuracy drop)." >&2
        exit 1
    fi
    PRETRAIN_ARGS+=(--load_pretrain --pretrain_path "${PRETRAIN_PATH}")
fi

# Run-dir tee logging (same as memoryBench/train.sh).
# For a single task the run name = swanlab_run + "_" + task (e.g. 6_24_press_button);
# the task comes from CLI --tasks > YAML tasks; multi-task / all adds no suffix.
CONFIG_FILE="${FINETUNE_DIR}/RMBench_vla/configs/rmbench_config.yaml"
LOG_DIR_CFG="${BRIDGEVLA_LOG_DIR}"
export RMBENCH_CLI_TASKS="${CLI_TASKS[*]}"
export RMBENCH_CLI_TEMPORAL_MEM="${CLI_TEMPORAL_MEM}"
export RMBENCH_CLI_SPATIAL_MEM="${CLI_SPATIAL_MEM}"
read -r SWANLAB_RUN SWANLAB_RUN_BASE EFFECTIVE_TASKS <<< "$(python - "${CONFIG_FILE}" <<'PY'
import os, sys, yaml

cfg = yaml.safe_load(open(sys.argv[1]))
base = str(cfg.get("swanlab_run", "run"))
cli = [t for t in os.environ.get("RMBENCH_CLI_TASKS", "").split() if t]

if cli:
    tasks = cli
else:
    raw = str(cfg.get("tasks", "all") or "all")
    parsed = [t for t in raw.replace(",", " ").split() if t]
    tasks = parsed if parsed and parsed[0] != "all" else ["all"]

if len(tasks) == 1 and tasks[0] != "all":
    task = tasks[0]
    suffix = f"_{task}"
    run = base if base.endswith(suffix) else f"{base}{suffix}"
else:
    run = base

# Memory-ablation suffix (must mirror train.py:memory_ablation_suffix so the
# run dir == SwanLab run name). Only the two memory master switches matter:
#   temporal_memory off -> _no_temporal_mem ; spatial_memory off -> _no_spatial_mem
mem = cfg.get("memory", {}) or {}
def _ov(envname, cur):
    # CLI override (True/False) > YAML; empty env -> keep YAML value.
    v = os.environ.get(envname, "")
    return cur if v == "" else (v == "True")
mem_suffix = ""
if bool(mem.get("enabled", False)):
    t = _ov("RMBENCH_CLI_TEMPORAL_MEM", bool(mem.get("temporal_memory", True)))
    s = _ov("RMBENCH_CLI_SPATIAL_MEM", bool(mem.get("spatial_memory", True)))
    if not t and not s:
        mem_suffix = "_no_mem"
    elif not t:
        mem_suffix = "_no_temporal_mem"
    elif not s:
        mem_suffix = "_no_spatial_mem"
if mem_suffix and not run.endswith(mem_suffix):
    run = f"{run}{mem_suffix}"

print(run, base, ",".join(tasks))
PY
)"
# Assemble --exp_cfg_opts (yacs merge_from_list takes a space-separated key value string):
#   - swanlab_run is only overridden when there is a suffix (task/ablation);
#   - memory.temporal_memory / memory.spatial_memory are only overridden when given explicitly on the CLI;
#   - epochs / save_every_n_epochs are only overridden when given explicitly on the CLI.
EXP_CFG_OPTS_ARGS=()
EXP_CFG_OPTS_STR=""
if [[ "${SWANLAB_RUN}" != "${SWANLAB_RUN_BASE}" ]]; then
    EXP_CFG_OPTS_STR="swanlab_run ${SWANLAB_RUN}"
fi
if [[ -n "${CLI_TEMPORAL_MEM}" ]]; then
    EXP_CFG_OPTS_STR="${EXP_CFG_OPTS_STR:+${EXP_CFG_OPTS_STR} }memory.temporal_memory ${CLI_TEMPORAL_MEM}"
fi
if [[ -n "${CLI_SPATIAL_MEM}" ]]; then
    EXP_CFG_OPTS_STR="${EXP_CFG_OPTS_STR:+${EXP_CFG_OPTS_STR} }memory.spatial_memory ${CLI_SPATIAL_MEM}"
fi
if [[ -n "${CLI_EPOCHS}" ]]; then
    EXP_CFG_OPTS_STR="${EXP_CFG_OPTS_STR:+${EXP_CFG_OPTS_STR} }epochs ${CLI_EPOCHS}"
fi
if [[ -n "${CLI_SAVE_EVERY}" ]]; then
    EXP_CFG_OPTS_STR="${EXP_CFG_OPTS_STR:+${EXP_CFG_OPTS_STR} }save_every_n_epochs ${CLI_SAVE_EVERY}"
fi
if [[ -n "${EXP_CFG_OPTS_STR}" ]]; then
    EXP_CFG_OPTS_ARGS=(--exp_cfg_opts "${EXP_CFG_OPTS_STR}")
fi
export RMBENCH_RUN_STAMP="${RMBENCH_RUN_STAMP:-$(date +%m_%d_%H_%M)}"
if [[ "$USE_RESUME" == "true" ]] && [[ -n "${RESUME_PATH}" ]]; then
    # Resuming: the shell-side tee follows the checkpoint's own run directory (its parent), matching
    # train.py pinning log_dir to dirname(resume_path).
    RUN_DIR="$(cd "$(dirname "${RESUME_PATH}")" && pwd)"
    echo "[train.sh] resume: reusing original run dir ${RUN_DIR}"
else
    RUN_DIR="${LOG_DIR_CFG}/train_rmbench/${SWANLAB_RUN}_${RMBENCH_RUN_STAMP}"
fi
mkdir -p "${RUN_DIR}"
LOG_FILE="${RUN_DIR}/train_node${MLP_ROLE_INDEX}.log"
echo "[train.sh] tasks:      ${EFFECTIVE_TASKS}"
echo "[train.sh] swanlab_run: ${SWANLAB_RUN}"
echo "[train.sh] memory:     temporal=${CLI_TEMPORAL_MEM:-<yaml>}  spatial=${CLI_SPATIAL_MEM:-<yaml>}"
echo "[train.sh] epochs:     ${CLI_EPOCHS:-<yaml>}  save_every_n_epochs=${CLI_SAVE_EVERY:-<yaml>}"
echo "[train.sh] run dir:    ${RUN_DIR}"
echo "[train.sh] log:        ${LOG_FILE}"
export PYTHONUNBUFFERED=1

set -e -x -o pipefail
cd "${FINETUNE_DIR}/RMBench_vla"

torchrun \
    --nnodes=$MLP_WORKER_NUM \
    --node_rank=$MLP_ROLE_INDEX \
    --nproc_per_node=$MLP_WORKER_GPU \
    --master_addr=$MLP_WORKER_0_HOST \
    --master_port=$MLP_WORKER_0_PORT \
    train.py \
    --exp_cfg_path configs/rmbench_config.yaml \
    --log_dir "${BRIDGEVLA_LOG_DIR}" \
    --data_root "${RMBENCH_VLA_DATA_ROOT}" \
    "${EXP_CFG_OPTS_ARGS[@]}" \
    "${PRETRAIN_ARGS[@]}" \
    "${RESUME_ARGS[@]}" \
    "${FORWARD_ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"
