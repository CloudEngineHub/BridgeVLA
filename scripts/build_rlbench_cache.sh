#!/usr/bin/env bash
# Build (or upgrade) the RLBench canonical keyframe cache OUTSIDE of training.
#
# Training loads the cache strictly (allow_build=False) and never builds it.
# The released cache reproduces our runs exactly — prefer:
#   bash scripts/download_checkpoints_hf.sh rlbench_cache
# Use this script only to (re)build from raw RLBench demos (hours), e.g. for a
# custom episode set. NOTE: canonical keyframes are majority-voted over the
# episodes present locally — a partial episode set can yield DIFFERENT
# keyframes than the released cache.
#
# Run inside the RLBench training env (conda activate bridgevla_plus_rlbench).
# All args are forwarded to rlbench_keyframe_dataset.py:
#   bash scripts/build_rlbench_cache.sh                      # all 18 tasks, 100 eps
#   bash scripts/build_rlbench_cache.sh --tasks close_jar    # one task
#   bash scripts/build_rlbench_cache.sh --data_root data/bridgevla_data/RLBench

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FINETUNE_DIR="${REPO_ROOT}/finetune"

# Same simulator stack the trainer uses (mirrors finetune/RLBench/train.sh):
# get_stored_demo imports PyRep -> needs libcoppeliaSim.so on LD_LIBRARY_PATH,
# but never launches the simulator — offscreen is enough, no Xvfb.
PYREP_SIM_STACK="${PYREP_SIM_STACK-${FINETUNE_DIR}/bridgevla/libs/PyRep_stepjam231}"
if [ -n "${PYREP_SIM_STACK}" ]; then
    if ! ls "${PYREP_SIM_STACK}"/pyrep/backend/_sim_cffi*.so >/dev/null 2>&1; then
        echo "[build_rlbench_cache] ERROR: PYREP_SIM_STACK missing or not compiled: ${PYREP_SIM_STACK}" >&2
        echo "[build_rlbench_cache]        run finetune/RLBench/install_rlbench.sh first (or set PYREP_SIM_STACK)." >&2
        exit 1
    fi
    export PYTHONPATH="${PYREP_SIM_STACK}:${PYTHONPATH:-}"
fi
export COPPELIASIM_ROOT="${FINETUNE_DIR}/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM=offscreen
export PYTHONPATH="${FINETUNE_DIR}:${PYTHONPATH:-}"

cd "${REPO_ROOT}"
exec python finetune/RLBench/utils/rlbench_keyframe_dataset.py "$@"
