#!/usr/bin/env bash

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

# conda (CONDA_BASE overridable, auto-detected by default)
if [ -z "${CONDA_BASE:-}" ]; then
    CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    [ -z "${CONDA_BASE}" ] && [ -d "${HOME}/miniconda3" ] && CONDA_BASE="${HOME}/miniconda3"
    [ -z "${CONDA_BASE}" ] && [ -d "${HOME}/anaconda3" ] && CONDA_BASE="${HOME}/anaconda3"
fi
source "${CONDA_BASE}/bin/activate" "${GEMBENCH_CONDA_ENV:-bridgevla_plus_gembench}"

# GPU selection: respect an exported CUDA_VISIBLE_DEVICES; otherwise default to every detected GPU.
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    _NGPU_DETECTED=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
    [ "${_NGPU_DETECTED}" -lt 1 ] && _NGPU_DETECTED=1
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((_NGPU_DETECTED - 1)))
fi
# Reduce CUDA caching-allocator fragmentation. The training step issues
# 5 PaliGemma forwards with mixed batch sizes (bs=2 for current/anchor,
# bs*K=8 for history), which makes the default best-fit allocator hold
# large reserved-but-unused blocks; expandable_segments lets it grow/
# shrink physical pages on demand and typically cuts reserved memory by
# 20-30% at zero numerical cost. PyTorch >= 2.1 (we're on 2.5.1).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Same PYTHONPATH stack as GemBench/train.sh + memoryBench dir at the front so
# `from utils.peract_utils_memorybench import ...` resolves to memoryBench/utils,
# not GemBench/utils. peract import path is included so demo_loading_utils
# resolves at dataset-build time.
export PYTHONPATH="${FINETUNE_DIR}/memoryBench:${FINETUNE_DIR}:${FINETUNE_DIR}/bridgevla/libs/point-renderer:${FINETUNE_DIR}/bridgevla/libs/peract_colab:${FINETUNE_DIR}/bridgevla/libs/YARR:${FINETUNE_DIR}/bridgevla/libs/peract:${FINETUNE_DIR}/GemBench:${FINETUNE_DIR}/bridgevla/libs/PyRep:${FINETUNE_DIR}/bridgevla/libs/RLBench:${PYTHONPATH:-}"
# CLIP_* / PALIGEMMA_PATH / HF_HOME / HF offline flags are already derived at the top of this script.
export COPPELIASIM_ROOT="${FINETUNE_DIR}/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM=offscreen

# SwanLab toggle. Same convention as GemBench/train.sh.
SWANLAB_UPLOAD="${SWANLAB_UPLOAD:-offline}"
[ "${SWANLAB_UPLOAD}" = "local" ] && SWANLAB_UPLOAD="offline"
export SWANLAB_MODE="${SWANLAB_UPLOAD}"
if [ "${SWANLAB_UPLOAD}" = "cloud" ]; then
    export SWANLAB_API_KEY="${SWANLAB_API_KEY:?SWANLAB_UPLOAD=cloud requires exporting SWANLAB_API_KEY=<your key> first}"
else
    unset SWANLAB_API_KEY
fi

# MemoryBench data root + dataset cache (npz + index json).
export MEMORYBENCH_DATA_FOLDER="${BRIDGEVLA_ROOT}/data/bridgevla_data/memorybench/data/train"
# IMAGE_SIZE = 128 (matches the raw RLBench save resolution; see
# utils/peract_utils_memorybench.py). The cache subdir MUST match: an older
# size256/ directory is a stale bilinear-upsampled cache that bakes phantom
# intermediate-depth points across object boundaries -- if reused, the right-
# view render shows them as floating streaks. Always pin to the current size.
#
# `_v3` suffix: prepend frame 0 to the saved keyframe list (mirrors GemBench's
# `key_frames.insert(0, 0)`), so step_idx=0 in __getitem__ is the env-reset
# state -> first detected keyframe action, matching client.py's step_id=0
# rollout. Older `size128/` caches store the FIRST detected keyframe at slot
# 0 instead, shifting every step by 1 and dropping the (frame_0 -> kf0)
# transition; older `size128_v2/` caches also store ignore_collisions at the
# target keyframe instead of PerAct's target-previous-frame action label.
export MEMORYBENCH_CACHE_DIR="${BRIDGEVLA_ROOT}/data/bridgevla_data/memorybench/data/_keyframe_cache/size128_v3"

# Cluster-launch shims (same as GemBench/train.sh). nproc_per_node MUST equal
# the number of visible GPUs; derive it from CUDA_VISIBLE_DEVICES (set above)
# instead of a hardcoded 6 or RESOURCE_GPU (which on some machines is a float
# like "1.00" that torchrun rejects). Matches RLBench/RMBench_vla train.sh.
export MLP_WORKER_NUM=${WORLD_SIZE:-1}
_NGPU=$(awk -F, '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")
export MLP_WORKER_GPU=${_NGPU}
export MLP_ROLE_INDEX=${RANK:-0}
export MLP_WORKER_0_HOST=${MASTER_ADDR:-localhost}
export MLP_WORKER_0_PORT=${MASTER_PORT:-29622}

# Pretrain warm start (optional; warm-starts from a BridgeVLA grounding pretrain ckpt).
# Defaults to the pre-training weights released with BridgeVLA (an HF directory); if you ran
# pretrain/pretrain.sh yourself, point PRETRAIN_PATH at <run>/pretrain_epoch_<N>.pth — both layouts work (see README §3).
# Quick one-off override (no script edit): append `--no-pretrain` to disable.
USE_PRETRAIN=true
PRETRAIN_PATH="${PRETRAIN_PATH:-${BRIDGEVLA_CKPT_ROOT}/pretrain}"

# Resume from checkpoint — edit this line, or pass
# `--resume_path /path/to/model_last.pth` on the command line.
# Resume takes PRIORITY over the pretrain warm-start above: when set,
# USE_PRETRAIN is forced off, full training state (model + optimizer +
# LR-warmup step + epoch) is restored, and the run continues in the
# checkpoint's OWN run folder (original timestamp preserved). Scheduling
# config (freeze_epochs / save_every_n_epochs / epochs / ...) still comes
# from the YAML.
RESUME_PATH="${RESUME_PATH:-}"

# ---- CLI parsing: this script handles --no-pretrain / --resume_path plus the memory ablation switches
#   --temporal_memory <bool> / --spatial_memory <bool> (the = form works too).
#   The memory switches are not passed through to train.py; they are turned into --exp_cfg_opts
#   overrides of the YAML and also drive the ablation suffix of swanlab_run / the run directory
#   (same convention as RLBench/GemBench/RMBench_vla):
#     bash train.sh --temporal_memory false   # disable temporal memory (memory 1)
#     bash train.sh --spatial_memory false    # disable spatial memory (memory 2)
#   Everything else is forwarded to train.py verbatim. ----

# Normalise bools -> the Python literals True/False (yacs merge_from_list uses literal_eval, so
# they must be capitalised); an invalid value aborts with an error.
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
        --resume_path)   _want_resume_val=true ;;
        --resume_path=*) RESUME_PATH="${a#--resume_path=}" ;;
        --temporal_memory)    _want_tmem_val=true ;;
        --temporal_memory=*)  CLI_TEMPORAL_MEM="$(to_pybool "${a#--temporal_memory=}")" || exit 2 ;;
        --spatial_memory)     _want_smem_val=true ;;
        --spatial_memory=*)   CLI_SPATIAL_MEM="$(to_pybool "${a#--spatial_memory=}")" || exit 2 ;;
        *)               FORWARD_ARGS+=("$a") ;;
    esac
done
if [[ -n "${RESUME_PATH}" ]]; then
    USE_PRETRAIN=false   # resume wins over pretrain warm-start
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

# Make sure the 3 memorybench tasks are visible to RLBench (only matters if
# the dataset-build path tries to instantiate the task class for any reason).
bash "${FINETUNE_DIR}/memoryBench/scripts/install_memorybench_tasks.sh" || true

# Run-dir tee logging (same trick as GemBench/train.sh).
CONFIG_FILE="${FINETUNE_DIR}/memoryBench/configs/memorybench_config.yaml"
# The log root is the BRIDGEVLA_LOG_DIR derived at the top of this script (passed to train.py at
# runtime via --log_dir, overriding the fallback log_dir in the YAML); the shell-side tee uses the same root.
LOG_DIR_CFG="${BRIDGEVLA_LOG_DIR}"
# swanlab_run + memory ablation suffix (must stay in sync with train.py:memory_ablation_suffix so
# that the run directory the shell tees into == train.py's run directory):
#   temporal_memory off -> _no_temporal_mem ; spatial_memory off -> _no_spatial_mem
#   both off -> _no_mem. CLI switches (True/False) win over YAML; an empty env var = use the YAML value.
export MEMORYBENCH_CLI_TEMPORAL_MEM="${CLI_TEMPORAL_MEM}"
export MEMORYBENCH_CLI_SPATIAL_MEM="${CLI_SPATIAL_MEM}"
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
    t = _ov("MEMORYBENCH_CLI_TEMPORAL_MEM", bool(mem.get("temporal_memory", True)))
    s = _ov("MEMORYBENCH_CLI_SPATIAL_MEM", bool(mem.get("spatial_memory", True)))
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
# Assemble --exp_cfg_opts (yacs merge_from_list takes a space-separated key value string):
#   - swanlab_run is only overridden when there is an ablation suffix;
#   - memory.temporal_memory / memory.spatial_memory are only overridden when given explicitly on the CLI.
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
export MEMORYBENCH_RUN_STAMP="${MEMORYBENCH_RUN_STAMP:-$(date +%m_%d_%H_%M)}"
if [[ -n "${RESUME_PATH}" ]]; then
    # Resume: shell-side tee follows the checkpoint's own run folder (its parent
    # dir), matching train.py which pins log_dir to dirname(resume_path).
    RUN_DIR="$(cd "$(dirname "${RESUME_PATH}")" && pwd)"
    echo "[train.sh] resume: reusing original run dir ${RUN_DIR}"
else
    RUN_DIR="${LOG_DIR_CFG}/train_memorybench/${SWANLAB_RUN_CFG}_${MEMORYBENCH_RUN_STAMP}"
fi
mkdir -p "${RUN_DIR}"
LOG_FILE="${RUN_DIR}/train_node${MLP_ROLE_INDEX}.log"
echo "[train.sh] swanlab_run: ${SWANLAB_RUN_CFG}"
echo "[train.sh] memory:      temporal=${CLI_TEMPORAL_MEM:-<yaml>}  spatial=${CLI_SPATIAL_MEM:-<yaml>}"
echo "[train.sh] run dir: ${RUN_DIR}"
echo "[train.sh] log:     ${LOG_FILE}"
export PYTHONUNBUFFERED=1

set -e -x -o pipefail
cd "${FINETUNE_DIR}/memoryBench"

torchrun \
    --nnodes=$MLP_WORKER_NUM \
    --node_rank=$MLP_ROLE_INDEX \
    --nproc_per_node=$MLP_WORKER_GPU \
    --master_addr=$MLP_WORKER_0_HOST \
    --master_port=$MLP_WORKER_0_PORT \
    train.py \
    --exp_cfg_path configs/memorybench_config.yaml \
    --log_dir "${BRIDGEVLA_LOG_DIR}" \
    --data_folder "${MEMORYBENCH_DATA_FOLDER}" \
    --cache_dir "${MEMORYBENCH_CACHE_DIR}" \
    --mvt_cfg_opts "img_aug_2 0.0" \
    "${EXP_CFG_OPTS_ARGS[@]}" \
    "${PRETRAIN_ARGS[@]}" \
    "${RESUME_ARGS[@]}" \
    "${FORWARD_ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"
