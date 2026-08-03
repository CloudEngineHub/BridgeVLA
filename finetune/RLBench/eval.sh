#!/usr/bin/env bash
#
# RLBench eval launcher.
#
# Usage examples (override model/tasks/devices via the environment, no need to edit the script):
#   MODEL_FOLDER=/path/to/run MODEL_NAME=model_40.pth bash eval.sh
#   EVAL_TASKS=close_jar EVAL_DEVICE=0 bash eval.sh
#   SAVE_VIDEO=true bash eval.sh
#   SAVE_POINTCLOUD=true bash eval.sh    # save the merged multi-camera point cloud of every step (.ply, see eval.py)
#   EVAL_LOG_NAME=20260708_132137 bash eval.sh   # name the eval log subdirectory (defaults to the current timestamp)
set -e

# Paths and environment: all derived from the repository root, no machine-specific config;
FINETUNE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # <repo>/finetune
BRIDGEVLA_ROOT="$(dirname "${FINETUNE_DIR}")"                      # <repo>
export BRIDGEVLA_DATA_ROOT="${BRIDGEVLA_DATA_ROOT:-${BRIDGEVLA_ROOT}/data/bridgevla_data}"
export BRIDGEVLA_LOG_DIR="${BRIDGEVLA_LOG_DIR:-${BRIDGEVLA_DATA_ROOT}/logs}"
export BRIDGEVLA_CKPT_ROOT="${BRIDGEVLA_CKPT_ROOT:-${BRIDGEVLA_ROOT}/data/bridgevla_ckpt}"
# Released checkpoint root: one subdirectory per bench, holding model_<E>.pth + exp_cfg.yaml +
# mvt_cfg.yaml (all three are needed to rebuild the right model). Your own training runs follow
# the same convention, so just override MODEL_FOLDER=<run dir>.
export BRIDGEVLA_RELEASE_CKPT_DIR="${BRIDGEVLA_RELEASE_CKPT_DIR:-${BRIDGEVLA_CKPT_ROOT}/bridgevla_plus}"
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
source "${CONDA_BASE}/bin/activate" "${RLBENCH_CONDA_ENV:-bridgevla_plus_rlbench}"

# Model & data — override via the environment, no need to edit the script.
MODEL_FOLDER="${MODEL_FOLDER:-${BRIDGEVLA_RELEASE_CKPT_DIR}/rlbench}"
# Weight filename inside MODEL_FOLDER. To evaluate a specific epoch of your own run, use
# `MODEL_FOLDER=/path/to/run MODEL_NAME=model_40.pth bash eval.sh`.
MODEL_NAME="${MODEL_NAME:-model_130.pth}"
# A released directory usually holds a single weight: when MODEL_NAME does not match, the only
# model_*.pth in the directory is picked automatically, so re-releasing a different epoch needs no
# script change. With several candidates the value is kept and the sanity check below errors out.
if [ ! -f "${MODEL_FOLDER}/${MODEL_NAME}" ]; then
    _CKPT_CANDS=( "${MODEL_FOLDER}"/model_*.pth )
    if [ "${#_CKPT_CANDS[@]}" -eq 1 ] && [ -f "${_CKPT_CANDS[0]}" ]; then
        MODEL_NAME="$(basename "${_CKPT_CANDS[0]}")"
        echo "[Info] MODEL_NAME auto-resolved to ${MODEL_NAME} (the only weight in ${MODEL_FOLDER})"
    fi
fi
# Eval results land in {MODEL_FOLDER}/eval/{model_N}/{EVAL_LOG_NAME} (see eval.py).
# Defaults to the current timestamp; eval_multi_gpu.sh passes it explicitly so the stdout logs land in the same directory.
EVAL_LOG_NAME="${EVAL_LOG_NAME:-$(date +%Y%m%d_%H%M%S)}"
EVAL_DATAFOLDER="${RLBENCH_DATA_FOLDER:-${BRIDGEVLA_DATA_ROOT}/RLBench}"
DEFAULT_EVAL_TASKS="place_shape_in_shape_sorter place_cups stack_blocks place_wine_at_rack_location reach_and_drag insert_onto_square_peg push_buttons put_groceries_in_cupboard put_item_in_drawer put_money_in_safe light_bulb_in slide_block_to_color_target stack_cups sweep_to_dustpan_of_size turn_tap close_jar open_drawer meat_off_grill"
EVAL_TASKS="${EVAL_TASKS:-${DEFAULT_EVAL_TASKS}}"
EVAL_DEVICE="${EVAL_DEVICE:-0}"
EVAL_EPISODES="${EVAL_EPISODES:-25}"
EPISODE_LENGTH="${EPISODE_LENGTH:-25}"           # global fallback step_limit (used when a task is absent from the table)
# Per-task step_limit table (RMBench style), configs/eval_step_limit.yml by default.
# Non-null values in the table override EPISODE_LENGTH; set it to the empty string to always use EPISODE_LENGTH.
STEP_LIMIT_CONFIG="${STEP_LIMIT_CONFIG:-configs/eval_step_limit.yml}"
# Per-step heatmap visualisation (on by default): lands in {visualize_root_dir}/{task}/episode_{ep}/.
VISUALIZE="${VISUALIZE:-true}"
# Save a rollout mp4 (needs ffmpeg, off by default); independent of VISUALIZE (which is the heatmap).
SAVE_VIDEO="${SAVE_VIDEO:-false}"
# Save the point cloud of every step (merged multi-camera xyz+rgb, .ply, needs open3d); independent of VISUALIZE/SAVE_VIDEO.
# Outside visualize mode it lands in {eval_log_dir}/pointclouds/{task}/episode_{ep}/step_xxx.ply,
# in visualize mode in {visualize_root_dir}/{task}/episode_{ep}/pointclouds/step_xxx.ply.
SAVE_POINTCLOUD="${SAVE_POINTCLOUD:-false}"
# --- Memory ablation switches (train/eval must agree) ---
# Must match the memory.temporal_memory / memory.spatial_memory used when the model was trained
# (i.e. the values recorded in MODEL_FOLDER's mvt_cfg.yaml); a mismatch makes eval.py fail on load.
#   temporal_memory: memory 1 = stage-1 (mvt1) temporal memory;
#   spatial_memory:  memory 2 = stage-2 (mvt2) spatial anchor.
# Usage: TEMPORAL_MEMORY=false MODEL_FOLDER=/path/to/no_temporal_mem_run bash eval.sh
TEMPORAL_MEMORY="${TEMPORAL_MEMORY:-true}"
SPATIAL_MEMORY="${SPATIAL_MEMORY:-true}"

# PALIGEMMA_PATH / HF offline flags are already derived at the top of this script.

# PYTHONPATH (RLBench first, so that utils / visualize resolve to this directory).
export PYTHONPATH="${FINETUNE_DIR}/RLBench:${FINETUNE_DIR}:${FINETUNE_DIR}/bridgevla/libs/point-renderer:${FINETUNE_DIR}/bridgevla/libs/peract_colab:${FINETUNE_DIR}/bridgevla/libs/YARR:${FINETUNE_DIR}/bridgevla/libs/peract:${FINETUNE_DIR}/GemBench:${FINETUNE_DIR}/bridgevla/libs/PyRep:${PYTHONPATH:-}"

# ---- RLBench benchmark-specific sim stack (version sensitive, must match training!) --------
# The RLBench ckpts were trained and evaluated on buttomnutstoast/RLBench@587a6a0 plus the
# rlbench_patch_bundle patches, so that stack is put in front of PYTHONPATH, ahead of the
# shared libs/RLBench installed in the conda env.
# The directory is built by install_rlbench.sh; RLBENCH_SIM_STACK overrides it, and an empty value falls back to the shared stack (not recommended).
RLBENCH_SIM_STACK="${RLBENCH_SIM_STACK-${FINETUNE_DIR}/bridgevla/libs/RLBench_peract587}"
if [ -n "${RLBENCH_SIM_STACK}" ]; then
    if [ -d "${RLBENCH_SIM_STACK}/rlbench" ]; then
        export PYTHONPATH="${RLBENCH_SIM_STACK}:${PYTHONPATH}"
    else
        echo "[eval.sh] ERROR: RLBENCH_SIM_STACK does not exist: ${RLBENCH_SIM_STACK}" >&2
        echo "[eval.sh]        run install_rlbench.sh first (or build it by hand per rlbench_patch_bundle/README.md);" >&2
        echo "[eval.sh]        to deliberately use the shared libs/RLBench, set RLBENCH_SIM_STACK=\"\" to skip (results will not match the paper)." >&2
        exit 1
    fi
fi

# ---- Likewise: RLBench benchmark-specific PyRep (training env = stepjam/PyRep@231a1ac) -----
# The shared libs/PyRep is the cshizhe fork (GemBench lineage, with arm.py/sim.py modifications);
# for strict parity with the training environment the RLBench benchmark uses stepjam@231a1ac (no local patches).
# The directory is built and its cffi extension compiled in place by install_rlbench.sh; PYREP_SIM_STACK overrides it.
PYREP_SIM_STACK="${PYREP_SIM_STACK-${FINETUNE_DIR}/bridgevla/libs/PyRep_stepjam231}"
if [ -n "${PYREP_SIM_STACK}" ]; then
    if ls "${PYREP_SIM_STACK}"/pyrep/backend/_sim_cffi*.so >/dev/null 2>&1; then
        export PYTHONPATH="${PYREP_SIM_STACK}:${PYTHONPATH}"
    else
        echo "[SIM_STACK] ERROR: PYREP_SIM_STACK missing or not compiled: ${PYREP_SIM_STACK}" >&2
        echo "[SIM_STACK]        build it with install_rlbench.sh or (cd <that dir> && python setup.py build_ext --inplace)" >&2
        echo "[SIM_STACK]        to deliberately use the shared libs/PyRep, set PYREP_SIM_STACK=\"\" to skip (differs from the training environment)." >&2
        exit 1
    fi
fi

# CoppeliaSim / Qt / display (eval launches the simulator, which needs a display).
export COPPELIASIM_ROOT="${FINETUNE_DIR}/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}"
# Qt xcb needs the system Qt5 GLX plugin (apt-get install -y libqt5gui5).
export QT_PLUGIN_PATH="${COPPELIASIM_ROOT}:/usr/lib/x86_64-linux-gnu/qt5/plugins"

# Xvfb: the virtual X display used for headless rendering (reuse :99 or create a new one).
XVFB_DISPLAY="${XVFB_DISPLAY:-:99}"
if [ -z "${DISPLAY:-}" ] || ! xdpyinfo -display "${DISPLAY}" >/dev/null 2>&1; then
    if xdpyinfo -display "${XVFB_DISPLAY}" >/dev/null 2>&1; then
        echo "[Info] Reusing existing Xvfb on ${XVFB_DISPLAY}"
    else
        echo "[Info] Starting Xvfb on ${XVFB_DISPLAY} ..."
        Xvfb ${XVFB_DISPLAY} -screen 0 1024x768x24 -ac +extension GLX +render -noreset &
        XVFB_PID=$!
        sleep 2
        if ! kill -0 ${XVFB_PID} 2>/dev/null; then
            echo "[Error] Xvfb failed to start"; exit 1
        fi
        echo "[Info] Xvfb started (PID=${XVFB_PID})"
    fi
    export DISPLAY="${XVFB_DISPLAY}"
else
    echo "[Info] Using existing DISPLAY=${DISPLAY}"
fi

cd "${FINETUNE_DIR}/RLBench"

# Sanity checks
[ -d "${MODEL_FOLDER}" ]               || { echo "[Error] MODEL_FOLDER missing: ${MODEL_FOLDER}"; exit 1; }
[ -f "${MODEL_FOLDER}/${MODEL_NAME}" ] || { echo "[Error] checkpoint missing: ${MODEL_FOLDER}/${MODEL_NAME}"; exit 1; }
for _cfg in exp_cfg.yaml mvt_cfg.yaml; do
    [ -f "${MODEL_FOLDER}/${_cfg}" ] || {
        echo "[Error] ${_cfg} missing in ${MODEL_FOLDER}"
        echo "        the weights must sit next to exp_cfg.yaml + mvt_cfg.yaml (they define the model structure)."
        echo "        your own run directories already have them; to keep a standalone copy, put all three files in one directory:"
        echo "          model_<E>.pth  exp_cfg.yaml  mvt_cfg.yaml"
        exit 1
    }
done
[ -d "${EVAL_DATAFOLDER}" ]            || { echo "[Error] EVAL_DATAFOLDER missing: ${EVAL_DATAFOLDER}"; exit 1; }
[ -d "${PALIGEMMA_PATH}" ]             || { echo "[Error] PALIGEMMA_PATH missing: ${PALIGEMMA_PATH}"; exit 1; }

echo "[Info] MODEL_FOLDER     = ${MODEL_FOLDER}"
echo "[Info] MODEL_NAME       = ${MODEL_NAME}"
echo "[Info] EVAL_LOG_NAME    = ${EVAL_LOG_NAME}"
echo "[Info] EVAL_DATAFOLDER  = ${EVAL_DATAFOLDER}"
echo "[Info] EVAL_TASKS       = ${EVAL_TASKS}"
echo "[Info] EVAL_EPISODES    = ${EVAL_EPISODES}"
echo "[Info] EPISODE_LENGTH   = ${EPISODE_LENGTH}"
echo "[Info] STEP_LIMIT_CONFIG= ${STEP_LIMIT_CONFIG}"
echo "[Info] VISUALIZE        = ${VISUALIZE}"
echo "[Info] SAVE_VIDEO       = ${SAVE_VIDEO}"
echo "[Info] SAVE_POINTCLOUD  = ${SAVE_POINTCLOUD}"
echo "[Info] TEMPORAL_MEMORY  = ${TEMPORAL_MEMORY}"
echo "[Info] SPATIAL_MEMORY   = ${SPATIAL_MEMORY}"
echo "[Info] COPPELIASIM_ROOT = ${COPPELIASIM_ROOT}"
echo "[Info] DISPLAY          = ${DISPLAY}"

# Run evaluation. The agent loads memory + 6D rotation settings from the
# checkpoint's mvt_cfg.yaml; the YARR rollout drives the per-episode
# MemoryBank via agent.reset(). Extra flags can still be passed via "$@".
EVAL_EXTRA_ARGS=()
case "${VISUALIZE,,}" in
    true|1|yes|y|on)
        EVAL_EXTRA_ARGS+=(--visualize)
        ;;
    false|0|no|n|off)
        ;;
    *)
        echo "[Error] VISUALIZE must be true/false, got: ${VISUALIZE}"
        exit 1
        ;;
esac
case "${SAVE_VIDEO,,}" in
    true|1|yes|y|on)
        EVAL_EXTRA_ARGS+=(--save-video)
        ;;
    false|0|no|n|off)
        ;;
    *)
        echo "[Error] SAVE_VIDEO must be true/false, got: ${SAVE_VIDEO}"
        exit 1
        ;;
esac
case "${SAVE_POINTCLOUD,,}" in
    true|1|yes|y|on)
        EVAL_EXTRA_ARGS+=(--save-pointcloud)
        ;;
    false|0|no|n|off)
        ;;
    *)
        echo "[Error] SAVE_POINTCLOUD must be true/false, got: ${SAVE_POINTCLOUD}"
        exit 1
        ;;
esac

python3 eval.py \
    --model-folder    "${MODEL_FOLDER}" \
    --eval-datafolder "${EVAL_DATAFOLDER}" \
    --model-name      "${MODEL_NAME}" \
    --tasks           ${EVAL_TASKS} \
    --eval-episodes   "${EVAL_EPISODES}" \
    --episode-length  "${EPISODE_LENGTH}" \
    --step-limit-config "${STEP_LIMIT_CONFIG}" \
    --log-name        "${EVAL_LOG_NAME}" \
    --device          "${EVAL_DEVICE}" \
    --temporal_memory "${TEMPORAL_MEMORY}" \
    --spatial_memory  "${SPATIAL_MEMORY}" \
    --headless \
    "${EVAL_EXTRA_ARGS[@]}" \
    "$@"
