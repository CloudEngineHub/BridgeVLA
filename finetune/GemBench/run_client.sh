#!/usr/bin/env bash
set -euo pipefail

# Paths and environment: all derived from the repository root, no machine-specific config;
FINETUNE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # <repo>/finetune
BRIDGEVLA_ROOT="$(dirname "${FINETUNE_DIR}")"                      # <repo>
export BRIDGEVLA_DATA_ROOT="${BRIDGEVLA_DATA_ROOT:-${BRIDGEVLA_ROOT}/data/bridgevla_data}"
export BRIDGEVLA_LOG_DIR="${BRIDGEVLA_LOG_DIR:-${BRIDGEVLA_DATA_ROOT}/logs}"
export BRIDGEVLA_CKPT_ROOT="${BRIDGEVLA_CKPT_ROOT:-${BRIDGEVLA_ROOT}/data/bridgevla_ckpt}"
# Released checkpoint root: one subdirectory per bench, holding model_<E>.pth + exp_cfg.yaml +
# mvt_cfg.yaml (all three are needed to rebuild the right model). Your own training runs follow the
# same convention, so just override MODEL_FOLDER=<run dir>.
export BRIDGEVLA_RELEASE_CKPT_DIR="${BRIDGEVLA_RELEASE_CKPT_DIR:-${BRIDGEVLA_CKPT_ROOT}/bridgevla_plus}"
export PALIGEMMA_PATH="${PALIGEMMA_PATH:-${BRIDGEVLA_CKPT_ROOT}/paligemma-3b-pt-224}"
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

MODEL_FOLDER="${MODEL_FOLDER:-${BRIDGEVLA_RELEASE_CKPT_DIR}/gembench}"
GEMBENCH_ROOT="${GEMBENCH_ROOT:-${BRIDGEVLA_DATA_ROOT}/GemBench}"
SEED="${1:-300}"
MODEL_EPOCH="${2:-200}"
# A released directory usually holds a single weight: when model_${MODEL_EPOCH}.pth does not match, the
# only model_*.pth in the directory is used (no script change needed to release a different epoch); with several candidates the value is kept and the sanity check errors out.
# It must resolve to the same epoch as run_server.sh, or the server's weights and the client's labels disagree.
if [ ! -f "${MODEL_FOLDER}/model_${MODEL_EPOCH}.pth" ]; then
    _CKPT_CANDS=( "${MODEL_FOLDER}"/model_*.pth )
    if [ "${#_CKPT_CANDS[@]}" -eq 1 ] && [ -f "${_CKPT_CANDS[0]}" ]; then
        _CKPT_EPOCH="$(basename "${_CKPT_CANDS[0]}" .pth)"; _CKPT_EPOCH="${_CKPT_EPOCH#model_}"
        if [[ "${_CKPT_EPOCH}" =~ ^[0-9]+$ ]]; then
            MODEL_EPOCH="${_CKPT_EPOCH}"
            echo "[Info] MODEL_EPOCH auto-resolved to ${MODEL_EPOCH} (the only weight in ${MODEL_FOLDER})"
        fi
    fi
fi
MICROSTEP_DATA_DIR="${3:-${GEMBENCH_ROOT}/test_dataset/microsteps/seed${SEED}}"
OUTPUT_ROOT="${4:-${MODEL_FOLDER}/eval/gembench}"
PORT="${PORT:-13130}"
IP="${IP:-localhost}"
# --- Memory ablation switches (client/server must agree) ---
# Must match run_server.sh's TEMPORAL_MEMORY / SPATIAL_MEMORY (which the server has already validated
# against the loaded model). At startup the client validates against the server's /memory_config and
# errors out on a mismatch, so an ablation group is never silently evaluated wrong.
# Usage: TEMPORAL_MEMORY=false bash run_client.sh 300 80
TEMPORAL_MEMORY="${TEMPORAL_MEMORY:-true}"
SPATIAL_MEMORY="${SPATIAL_MEMORY:-true}"
export PYTHONPATH="${FINETUNE_DIR}:${FINETUNE_DIR}/bridgevla/libs/point-renderer:${FINETUNE_DIR}/bridgevla/libs/peract_colab:${FINETUNE_DIR}/bridgevla/libs/YARR:${FINETUNE_DIR}/GemBench:${FINETUNE_DIR}/bridgevla/libs/PyRep:${FINETUNE_DIR}/bridgevla/libs/RLBench:${PYTHONPATH:-}"
# PALIGEMMA_PATH / HF_HOME / HF offline flags are already derived at the top of this script.
export COPPELIASIM_ROOT="${FINETUNE_DIR}/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"

[ -d "${MICROSTEP_DATA_DIR}" ] || { echo "[Error] MICROSTEP_DATA_DIR missing: ${MICROSTEP_DATA_DIR}"; echo "[Hint] Extract ${GEMBENCH_ROOT}/test_dataset/microsteps.tar.gz under ${GEMBENCH_ROOT}/test_dataset first."; exit 1; }

# Video recording is off by default; per-step heatmap visualisation is on by default (lands in .../visualize/).
RECORD_VIDEO="${RECORD_VIDEO:-0}"
VIDEO_ROTATE_CAM="${VIDEO_ROTATE_CAM:-1}"
VIDEO_RES_W="${VIDEO_RES_W:-320}"
VIDEO_RES_H="${VIDEO_RES_H:-180}"
VISUALIZE="${VISUALIZE:-1}"
VISUALIZE_ROOT_DIR="${VISUALIZE_ROOT_DIR:-${OUTPUT_ROOT}/model_${MODEL_EPOCH}/seed${SEED}/visualize}"
MAX_EPISODES="${MAX_EPISODES:-0}"

EXTRA_ARGS=()
[ "${RECORD_VIDEO}" = "1" ] && EXTRA_ARGS+=(--record_video --video_resolution_width "${VIDEO_RES_W}" --video_resolution_height "${VIDEO_RES_H}")
[ "${VIDEO_ROTATE_CAM}" = "1" ] && EXTRA_ARGS+=(--video_rotate_cam)
[ "${VISUALIZE}" = "1" ] && EXTRA_ARGS+=(--visualize --visualize_root_dir "${VISUALIZE_ROOT_DIR}")
[ "${MAX_EPISODES}" != "0" ] && EXTRA_ARGS+=(--max_episodes "${MAX_EPISODES}")

echo "[Info] SERVER       = http://${IP}:${PORT}"
echo "[Info] MODEL_EPOCH  = ${MODEL_EPOCH}"
echo "[Info] SEED         = ${SEED}"
echo "[Info] TEMPORAL_MEMORY = ${TEMPORAL_MEMORY}"
echo "[Info] SPATIAL_MEMORY  = ${SPATIAL_MEMORY}"
echo "[Info] MICROSTEPS   = ${MICROSTEP_DATA_DIR}"
echo "[Info] OUTPUT_ROOT  = ${OUTPUT_ROOT}"
echo "[Info] RECORD_VIDEO = ${RECORD_VIDEO} (${VIDEO_RES_W}x${VIDEO_RES_H}, rotate=${VIDEO_ROTATE_CAM})"
echo "[Info] VISUALIZE    = ${VISUALIZE} (${VISUALIZE_ROOT_DIR})"
echo "[Info] MAX_EPISODES = ${MAX_EPISODES} (0 = all demos)"

cd "${FINETUNE_DIR}/GemBench"

TASKS_DIR="${FINETUNE_DIR}/GemBench/assets"
declare -A TASK_JSONS
TASK_JSONS[train]="${TASKS_DIR}/taskvars_train.json"
TASK_JSONS[test_l2]="${TASKS_DIR}/taskvars_test_l2.json"
TASK_JSONS[test_l3]="${TASKS_DIR}/taskvars_test_l3.json"
TASK_JSONS[test_l4]="${TASKS_DIR}/taskvars_test_l4.json"

SPLITS="${SPLITS:-train test_l2 test_l3 test_l4}"
for split in ${SPLITS}; do
    mapfile -t taskvars < <(python3 -c "import json,sys; [print(t) for t in json.load(open('${TASK_JSONS[$split]}'))]")
    for taskvar in "${taskvars[@]}"; do
        xvfb-run -a python3 client.py \
            --ip "${IP}" \
            --port "${PORT}" \
            --output_file "${OUTPUT_ROOT}/model_${MODEL_EPOCH}/seed${SEED}/${split}/result.json" \
            --microstep_data_dir "${MICROSTEP_DATA_DIR}" \
            --taskvar "${taskvar}" \
            --temporal_memory "${TEMPORAL_MEMORY}" \
            --spatial_memory "${SPATIAL_MEMORY}" \
            "${EXTRA_ARGS[@]}"
    done
done

