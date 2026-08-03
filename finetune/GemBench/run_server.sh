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
# same convention — just point the second positional argument at one.
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

MODEL_EPOCH="${1:-200}"
# Loads the released ckpt by default; to evaluate your own run, pass its directory as the 2nd argument,
# e.g. `bash run_server.sh 40 /path/to/run_folder`.
MODEL_FOLDER="${2:-${BRIDGEVLA_RELEASE_CKPT_DIR}/gembench}"
# A released directory usually holds a single weight: when model_${MODEL_EPOCH}.pth does not match, the
# only model_*.pth in the directory is used (no script change needed to release a different epoch); with several candidates the value is kept and the sanity check errors out.
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
PORT="${PORT:-13130}"
# --- Memory ablation switches (train/eval must agree) ---
# Must match the memory.temporal_memory / memory.spatial_memory used when the model was trained
# (i.e. the values recorded in its run directory's mvt_cfg.yaml); a mismatch makes the server abort at startup.
#   temporal_memory: memory 1 = stage-1 (mvt1) temporal memory;
#   spatial_memory:  memory 2 = stage-2 (mvt2) spatial anchor.
# Usage: TEMPORAL_MEMORY=false bash run_server.sh 80 /path/to/no_temporal_mem_run
# run_client.sh must be given the same switches (the client validates against the server at startup and errors out on a mismatch).
TEMPORAL_MEMORY="${TEMPORAL_MEMORY:-true}"
SPATIAL_MEMORY="${SPATIAL_MEMORY:-true}"
export PYTHONPATH="${FINETUNE_DIR}:${FINETUNE_DIR}/bridgevla/libs/point-renderer:${FINETUNE_DIR}/bridgevla/libs/peract_colab:${FINETUNE_DIR}/bridgevla/libs/YARR:${FINETUNE_DIR}/GemBench:${FINETUNE_DIR}/bridgevla/libs/PyRep:${FINETUNE_DIR}/bridgevla/libs/RLBench:${PYTHONPATH:-}"
# PALIGEMMA_PATH / HF_HOME / HF offline flags are already derived at the top of this script.
export COPPELIASIM_ROOT="${FINETUNE_DIR}/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}"
export QT_QPA_PLATFORM=offscreen
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

[ -d "${MODEL_FOLDER}" ] || { echo "[Error] MODEL_FOLDER missing: ${MODEL_FOLDER}"; exit 1; }
[ -f "${MODEL_FOLDER}/model_${MODEL_EPOCH}.pth" ] || { echo "[Error] checkpoint missing: ${MODEL_FOLDER}/model_${MODEL_EPOCH}.pth"; exit 1; }
for _cfg in exp_cfg.yaml mvt_cfg.yaml; do
    [ -f "${MODEL_FOLDER}/${_cfg}" ] || {
        echo "[Error] ${_cfg} missing in ${MODEL_FOLDER}"
        echo "        the weights must sit next to exp_cfg.yaml + mvt_cfg.yaml (they define the model structure)."
        echo "        your own run directories already have them; to keep a standalone copy, put all three files in one directory:"
        echo "          model_<E>.pth  exp_cfg.yaml  mvt_cfg.yaml"
        exit 1
    }
done
[ -d "${PALIGEMMA_PATH}" ] || { echo "[Error] PALIGEMMA_PATH missing: ${PALIGEMMA_PATH}"; exit 1; }

echo "[Info] MODEL_FOLDER    = ${MODEL_FOLDER}"
echo "[Info] MODEL_EPOCH     = ${MODEL_EPOCH}"
echo "[Info] PORT            = ${PORT}"
echo "[Info] TEMPORAL_MEMORY = ${TEMPORAL_MEMORY}"
echo "[Info] SPATIAL_MEMORY  = ${SPATIAL_MEMORY}"

cd "${FINETUNE_DIR}/GemBench"
xvfb-run -a python3 server.py --port "${PORT}" --model_epoch "${MODEL_EPOCH}" --base_path "${MODEL_FOLDER}" \
    --temporal_memory "${TEMPORAL_MEMORY}" --spatial_memory "${SPATIAL_MEMORY}"
