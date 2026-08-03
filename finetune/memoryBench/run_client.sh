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
# same convention — just point the first positional argument at one.
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

MODEL_FOLDER="${1:-${BRIDGEVLA_RELEASE_CKPT_DIR}/memorybench}"
MODEL_EPOCH="${2:-180}"
# A released directory usually holds a single weight: when model_${MODEL_EPOCH}.pth does not match, the
# only model_*.pth in the directory is used; it must resolve to the same epoch as run_server.sh (here it only names the result directory).
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
# SEED does two things: (1) passed to client.py --seed, it seeds the RNGs
# (python random + np.random) that drive RLBench's task.reset() episode
# randomization; (2) it names the output dir seed${SEED}/.
SEED="${SEED:-608}"
# Optional: per-variation symlink tree built by scripts/reorg_data_for_eval.py.
# Empty -> client falls back to fresh task.reset().
DEMO_DIR="${DEMO_DIR:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${MODEL_FOLDER}/eval/memorybench}"
PORT="${PORT:-13168}"
IP="${IP:-localhost}"
NUM_EPISODES="${NUM_EPISODES:-25}"
MAX_STEPS="${MAX_STEPS:-25}"
IMAGE_SIZE="${IMAGE_SIZE:-128}"

# ---- Memory ablation switches (eval side) ----
# Must match run_server.sh's TEMPORAL_MEMORY / SPATIAL_MEMORY (which the server has already checked
# against the loaded model's mvt_cfg.yaml); the client re-checks against the server at startup and aborts on a mismatch.
# Usage: TEMPORAL_MEMORY=false bash run_client.sh /path/to/no_temporal_mem_run 180
TEMPORAL_MEMORY="${TEMPORAL_MEMORY:-true}"
SPATIAL_MEMORY="${SPATIAL_MEMORY:-true}"

export PYTHONPATH="${FINETUNE_DIR}/memoryBench:${FINETUNE_DIR}:${FINETUNE_DIR}/bridgevla/libs/point-renderer:${FINETUNE_DIR}/bridgevla/libs/peract_colab:${FINETUNE_DIR}/bridgevla/libs/YARR:${FINETUNE_DIR}/bridgevla/libs/peract:${FINETUNE_DIR}/GemBench:${FINETUNE_DIR}/bridgevla/libs/PyRep:${FINETUNE_DIR}/bridgevla/libs/RLBench:${PYTHONPATH:-}"
# PALIGEMMA_PATH / HF_HOME / HF offline flags are already derived at the top of this script.
export COPPELIASIM_ROOT="${FINETUNE_DIR}/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"

bash "${FINETUNE_DIR}/memoryBench/scripts/install_memorybench_tasks.sh"

# Video recording is off by default; per-step heatmap visualisation is on by default (lands in .../visualize/).
RECORD_VIDEO="${RECORD_VIDEO:-0}"
VIDEO_ROTATE_CAM="${VIDEO_ROTATE_CAM:-0}"
VIDEO_RES_W="${VIDEO_RES_W:-320}"
VIDEO_RES_H="${VIDEO_RES_H:-180}"
VISUALIZE="${VISUALIZE:-1}"
MODEL_TAG="model_${MODEL_EPOCH}"
VISUALIZE_ROOT_DIR="${VISUALIZE_ROOT_DIR:-${OUTPUT_ROOT}/${MODEL_TAG}/seed${SEED}/visualize}"
MAX_EPISODES="${MAX_EPISODES:-25}"

EXTRA_ARGS=()
[ "${RECORD_VIDEO}" = "1" ] && EXTRA_ARGS+=(--record_video --video_resolution_width "${VIDEO_RES_W}" --video_resolution_height "${VIDEO_RES_H}")
[ "${VIDEO_ROTATE_CAM}" = "1" ] && EXTRA_ARGS+=(--video_rotate_cam)
[ "${VISUALIZE}" = "1" ] && EXTRA_ARGS+=(--visualize --visualize_root_dir "${VISUALIZE_ROOT_DIR}")
[ "${MAX_EPISODES}" != "0" ] && EXTRA_ARGS+=(--max_episodes "${MAX_EPISODES}")
[ -n "${DEMO_DIR}" ] && EXTRA_ARGS+=(--microstep_data_dir "${DEMO_DIR}")

echo "[Info] SERVER       = http://${IP}:${PORT}"
echo "[Info] MODEL_EPOCH  = ${MODEL_EPOCH}"
echo "[Info] SEED         = ${SEED}"
echo "[Info] DEMO_DIR     = ${DEMO_DIR:-(none, using task.reset())}"
echo "[Info] OUTPUT_ROOT  = ${OUTPUT_ROOT}"
echo "[Info] NUM_EPISODES = ${NUM_EPISODES}"
echo "[Info] MAX_STEPS    = ${MAX_STEPS}"
echo "[Info] TEMPORAL_MEMORY = ${TEMPORAL_MEMORY}"
echo "[Info] SPATIAL_MEMORY  = ${SPATIAL_MEMORY}"

cd "${FINETUNE_DIR}/memoryBench"

TASKS_JSON="${FINETUNE_DIR}/memoryBench/assets/taskvars.json"

# Filter which taskvars to roll out. Precedence:
#   1) TASKS env (comma-separated task names, or "all")
#   2) Else read `tasks:` from memorybench_config.yaml (same convention as training)
#   3) "all" -> no filtering (all 9 taskvars from taskvars.json)
# Filtering matches by the task-name prefix before '+'; so TASKS=reopen_drawer
# keeps reopen_drawer+0, reopen_drawer+1, reopen_drawer+2.
CONFIG_FILE="${FINETUNE_DIR}/memoryBench/configs/memorybench_config.yaml"
TASKS="${TASKS:-$(python3 -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1])).get("tasks","all") or "all")' "${CONFIG_FILE}")}"
echo "[Info] TASKS filter = ${TASKS}"

mapfile -t taskvars < <(python3 -c "
import json, sys
taskvars = json.load(open(sys.argv[1]))
filt = sys.argv[2].strip()
if filt and filt != 'all':
    keep = {t.strip() for t in filt.split(',') if t.strip()}
    taskvars = [tv for tv in taskvars if tv.split('+', 1)[0] in keep]
for t in taskvars:
    print(t)
" "${TASKS_JSON}" "${TASKS}")

if [ "${#taskvars[@]}" -eq 0 ]; then
    echo "[Error] No taskvars matched filter '${TASKS}'. Check spelling against ${TASKS_JSON}." >&2
    exit 1
fi
echo "[Info] Will roll out ${#taskvars[@]} taskvars: ${taskvars[*]}"

for taskvar in "${taskvars[@]}"; do
    xvfb-run -a python3 client.py \
        --ip "${IP}" \
        --port "${PORT}" \
        --output_file "${OUTPUT_ROOT}/${MODEL_TAG}/seed${SEED}/result.jsonl" \
        --taskvar "${taskvar}" \
        --num_episodes "${NUM_EPISODES}" \
        --max_steps "${MAX_STEPS}" \
        --image_size "${IMAGE_SIZE}" \
        --seed "${SEED}" \
        --temporal_memory "${TEMPORAL_MEMORY}" \
        --spatial_memory "${SPATIAL_MEMORY}" \
        "${EXTRA_ARGS[@]}"
done
