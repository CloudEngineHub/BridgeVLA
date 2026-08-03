#!/usr/bin/env bash
#
# Colosseum trainer launcher (ported from RLBench/train.sh).
#
# Usage examples:
#   bash train.sh                      # use the tasks from configs/colosseum_config.yaml
#   CUDA_VISIBLE_DEVICES=0 bash train.sh
#   bash train.sh --no-pretrain        # cold start (no pretrain warm start)
#   bash train.sh --resume_path /path/to/model_xx.pth
#   # memory ablation: switches given on the command line (higher priority than YAML; same convention as RMBench_vla)
#   bash train.sh --temporal_memory false   # disable temporal memory (memory 1)
#   bash train.sh --spatial_memory false    # disable spatial memory (memory 2)
#   bash train.sh --temporal_memory false --spatial_memory false  # disable both
#   (bools accept true/false/1/0/yes/no/on/off; --temporal_memory=false also works)
#   Disabling a memory automatically appends an ablation suffix to swanlab_run / the run directory
#   (_no_temporal_mem / _no_spatial_mem / _no_mem).
#   All other arguments are passed through to train.py verbatim (e.g. --epochs 200).

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
# Colosseum training run root.
export COLOSSEUM_LOG_DIR="${COLOSSEUM_LOG_DIR:-${BRIDGEVLA_LOG_DIR}/train_colosseum}"

# conda (CONDA_BASE overridable, auto-detected by default). Colosseum reuses RLBench's python
# environment (same sim stack; robot-colosseum itself is vendored and comes in via PYTHONPATH).
if [ -z "${CONDA_BASE:-}" ]; then
    CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    [ -z "${CONDA_BASE}" ] && [ -d "${HOME}/miniconda3" ] && CONDA_BASE="${HOME}/miniconda3"
    [ -z "${CONDA_BASE}" ] && [ -d "${HOME}/anaconda3" ] && CONDA_BASE="${HOME}/anaconda3"
fi
source "${CONDA_BASE}/bin/activate" "${COLOSSEUM_CONDA_ENV:-bridgevla_plus_rlbench}"

# GPU selection: respect an exported CUDA_VISIBLE_DEVICES; otherwise default to every detected GPU.
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    _NGPU_DETECTED=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
    [ "${_NGPU_DETECTED}" -lt 1 ] && _NGPU_DETECTED=1
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((_NGPU_DETECTED - 1)))
fi
# Reduce CUDA caching-allocator fragmentation (memory.grad_through_tokens does several PaliGemma forwards per step).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Colosseum comes first on PYTHONPATH so that `from utils.peract_utils_colosseum import ...`
# and `from visualize import ...` resolve to Colosseum's own directory. robot-colosseum
# (vendored) provides the `colosseum` package (used by eval; train never imports it, but having
# it on the path is harmless). Other pure-Python packages are handled by PYTHONPATH (same convention as GemBench/RLBench).
export PYTHONPATH="${FINETUNE_DIR}/Colosseum:${FINETUNE_DIR}:${FINETUNE_DIR}/Colosseum/robot-colosseum:${FINETUNE_DIR}/bridgevla/libs/point-renderer:${FINETUNE_DIR}/bridgevla/libs/peract_colab:${FINETUNE_DIR}/bridgevla/libs/YARR:${FINETUNE_DIR}/bridgevla/libs/peract:${FINETUNE_DIR}/GemBench:${FINETUNE_DIR}/bridgevla/libs/PyRep:${FINETUNE_DIR}/bridgevla/libs/RLBench:${PYTHONPATH:-}"

# CoppeliaSim: on the training side get_stored_demo imports PyRep (so libcoppeliaSim.so must be on
# LD_LIBRARY_PATH) but never launches the simulator — offscreen is enough, no Xvfb needed (that is eval's business).
export COPPELIASIM_ROOT="${FINETUNE_DIR}/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM=offscreen
# PALIGEMMA_PATH / CLIP_* / HF offline flags are already derived at the top of this script.

# Colosseum data root (contains [train/]<task>/all_variations/episodes).
export COLOSSEUM_DATA_FOLDER="${COLOSSEUM_DATA_FOLDER:-${BRIDGEVLA_DATA_ROOT}/Colosseum}"

# Fail fast on a missing keyframe cache BEFORE torchrun launches — one clean
# message instead of a traceback per DDP rank (the dataset still verifies every
# episode strictly). Matches the default cache dir derived in
# colosseum_keyframe_dataset.py: <data_root>/_keyframe_cache/size128_v2.
_KF_CACHE="${COLOSSEUM_DATA_FOLDER}/_keyframe_cache/size128_v2"
if ! ls "${_KF_CACHE}"/*/episode*.npz.meta >/dev/null 2>&1; then
    echo "[train.sh] ERROR: Colosseum keyframe cache missing or empty: ${_KF_CACHE}" >&2
    echo "[train.sh]   Training loads it strictly and never builds it. Build it with the" >&2
    echo "[train.sh]   data prep script (extracts the raw data AND builds the cache):" >&2
    echo "[train.sh]     bash finetune/Colosseum/prepare_colosseum_data.sh" >&2
    exit 1
fi

# SwanLab toggle (same convention as GemBench/train.sh: cloud upload by default; if cloud init
# fails train.py falls back to offline with logs in <run dir>/swanlog).
SWANLAB_UPLOAD="${SWANLAB_UPLOAD:-offline}"
[ "${SWANLAB_UPLOAD}" = "local" ] && SWANLAB_UPLOAD="offline"
export SWANLAB_MODE="${SWANLAB_UPLOAD}"
if [ "${SWANLAB_UPLOAD}" = "cloud" ]; then
    export SWANLAB_API_KEY="${SWANLAB_API_KEY:?SWANLAB_UPLOAD=cloud requires exporting SWANLAB_API_KEY=<your key> first}"
else
    unset SWANLAB_API_KEY
fi

# Cluster-launch shims (same as RMBench_vla/train.sh). nproc_per_node == number of visible GPUs.
export MLP_WORKER_NUM=${WORLD_SIZE:-1}
_NGPU=$(awk -F, '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")
export MLP_WORKER_GPU=${_NGPU}
export MLP_ROLE_INDEX=${RANK:-0}
export MLP_WORKER_0_HOST=${MASTER_ADDR:-localhost}
export MLP_WORKER_0_PORT=${MASTER_PORT:-29532}

# ---- Pretrain warm start (optional; a memory + 6D compatible warm-start ckpt) ----
# Defaults to the grounding pre-training weights released with BridgeVLA (an HF directory); if you ran
# pretrain/pretrain.sh yourself, point it at <run>/pretrain_epoch_<N>.pth — mvt_single accepts both layouts (see README §3).
USE_PRETRAIN=true
PRETRAIN_PATH="${PRETRAIN_PATH:-${BRIDGEVLA_CKPT_ROOT}/pretrain}"

# Resume from a checkpoint (optional; resuming takes priority over the pretrain warm start).
USE_RESUME=false
RESUME_PATH="${RESUME_PATH:-}"

# ---- CLI parsing: this script handles --no-pretrain / --resume_path plus the memory ablation switches
#   --temporal_memory <bool> / --spatial_memory <bool> (the = form works too).
#   The memory switches are not passed through to train.py; they are turned into --exp_cfg_opts
#   overrides of the YAML and also drive the ablation suffix of swanlab_run / the run directory.
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
        RESUME_PATH="$a"; _want_resume_val=false; continue
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
        --no-pretrain)   USE_PRETRAIN=false; continue ;;
        --resume_path)   USE_RESUME=true; _want_resume_val=true; continue ;;
        --resume_path=*) USE_RESUME=true; RESUME_PATH="${a#--resume_path=}"; continue ;;
        --temporal_memory)    _want_tmem_val=true; continue ;;
        --temporal_memory=*)  CLI_TEMPORAL_MEM="$(to_pybool "${a#--temporal_memory=}")" || exit 2; continue ;;
        --spatial_memory)     _want_smem_val=true; continue ;;
        --spatial_memory=*)   CLI_SPATIAL_MEM="$(to_pybool "${a#--spatial_memory=}")" || exit 2; continue ;;
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

# Run-dir tee logging (matches train.py's get_logdir:
# ${COLOSSEUM_LOG_DIR}/{swanlab_run}_{stamp}, with COLOSSEUM_LOG_DIR derived at the top of this script).
CONFIG_FILE="${FINETUNE_DIR}/Colosseum/configs/colosseum_config.yaml"
# swanlab_run + memory ablation suffix (must stay in sync with train.py:memory_ablation_suffix so
# that the run directory the shell tees into == train.py's run directory):
#   temporal_memory off -> _no_temporal_mem ; spatial_memory off -> _no_spatial_mem
#   both off -> _no_mem. CLI switches (True/False) win over YAML; an empty env var = use the YAML value.
export COLOSSEUM_CLI_TEMPORAL_MEM="${CLI_TEMPORAL_MEM}"
export COLOSSEUM_CLI_SPATIAL_MEM="${CLI_SPATIAL_MEM}"
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
    t = _ov("COLOSSEUM_CLI_TEMPORAL_MEM", bool(mem.get("temporal_memory", True)))
    s = _ov("COLOSSEUM_CLI_SPATIAL_MEM", bool(mem.get("spatial_memory", True)))
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
export COLOSSEUM_RUN_STAMP="${COLOSSEUM_RUN_STAMP:-$(date +%m_%d_%H_%M)}"
if [[ "$USE_RESUME" == "true" ]] && [[ -n "${RESUME_PATH}" ]]; then
    RUN_DIR="$(cd "$(dirname "${RESUME_PATH}")" && pwd)"
    echo "[train.sh] resume: reusing original run dir ${RUN_DIR}"
else
    RUN_DIR="${COLOSSEUM_LOG_DIR}/${SWANLAB_RUN_CFG}_${COLOSSEUM_RUN_STAMP}"
fi
mkdir -p "${RUN_DIR}"
LOG_FILE="${RUN_DIR}/train_node${MLP_ROLE_INDEX}.log"
echo "[train.sh] swanlab_run: ${SWANLAB_RUN_CFG}"
echo "[train.sh] memory:      temporal=${CLI_TEMPORAL_MEM:-<yaml>}  spatial=${CLI_SPATIAL_MEM:-<yaml>}"
echo "[train.sh] log root:    ${COLOSSEUM_LOG_DIR}"
echo "[train.sh] run dir:     ${RUN_DIR}"
echo "[train.sh] log:         ${LOG_FILE}"
export PYTHONUNBUFFERED=1

set -e -x -o pipefail
cd "${FINETUNE_DIR}/Colosseum"

torchrun \
    --nnodes=$MLP_WORKER_NUM \
    --node_rank=$MLP_ROLE_INDEX \
    --nproc_per_node=$MLP_WORKER_GPU \
    --master_addr=$MLP_WORKER_0_HOST \
    --master_port=$MLP_WORKER_0_PORT \
    train.py \
    --exp_cfg_path configs/colosseum_config.yaml \
    --log_dir "${BRIDGEVLA_LOG_DIR}" \
    --data_folder "${COLOSSEUM_DATA_FOLDER}" \
    "${EXP_CFG_OPTS_ARGS[@]}" \
    "${PRETRAIN_ARGS[@]}" \
    "${RESUME_ARGS[@]}" \
    "${FORWARD_ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"
