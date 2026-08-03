#!/usr/bin/env bash
#
# Colosseum environment preparation (BridgeVLA edition).
#
# Key differences from the older BridgeVLA install_colosseum.sh:
#   1. buttomnutstoast/RLBench + stepjam/PyRep are no longer cloned separately — Colosseum reuses
#      the RLBench pipeline's shared environment (conda env bridgevla_plus_rlbench), where
#      rlbench/pyrep come from bridgevla/libs/{RLBench,PyRep} (the rjgpinel/cshizhe forks, used via
#      PYTHONPATH; see the stack matrix in bridgevla/libs/rlbench_patches/README.md).
#      robot-colosseum ships a compatibility shim for the PerAct-fork/upstream rlbench differences
#      (extensions/environment.py) and works on that fork; 9-DoF action routing is handled by
#      MoveArmThenGripper2 in utils/rlbench_planning.py (the same as RLBench eval).
#   2. utils/colosseum_utils.py no longer overwrites the shared rlbench/utils.py (that would break
#      the fork behaviour shared by RLBench/GemBench). The missing overhead_* image directories in
#      the colosseum data are instead handled inside vendored robot-colosseum's
#      colosseum/rlbench/extensions/task_environment.py by passing load_images=False to
#      get_stored_demos (eval's reset_to_demo only needs the low-dimensional state).
#   3. robot-colosseum needs no pip install: train.sh / eval.sh already put
#      finetune/Colosseum/robot-colosseum on PYTHONPATH (assets/tasks live inside the package
#      directory, so vendoring is enough). Only its runtime dependency hydra-core has to be added.
#
# Prerequisite: install the RLBench environment first via finetune/RLBench/install_rlbench.sh
# (with CoppeliaSim V4_1_0 under finetune/, where train.sh/eval.sh point).
set -e

# conda (CONDA_BASE overridable, auto-detected by default)
if [ -z "${CONDA_BASE:-}" ]; then
    CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    [ -z "${CONDA_BASE}" ] && [ -d "${HOME}/miniconda3" ] && CONDA_BASE="${HOME}/miniconda3"
    [ -z "${CONDA_BASE}" ] && [ -d "${HOME}/anaconda3" ] && CONDA_BASE="${HOME}/anaconda3"
fi
source "${CONDA_BASE}/bin/activate" "${COLOSSEUM_CONDA_ENV:-bridgevla_plus_rlbench}"

echo "[install_colosseum] python = $(command -v python3)"

# robot-colosseum's runtime dependencies (requirements.txt: hydra-core>=1.3.2, numpy>=1.20.1; numpy
# is already in the env). omegaconf is a direct dependency of utils/multitask_rlbench_env.py and comes
# in with hydra-core, but is listed explicitly just in case.
python3 -m pip install "hydra-core>=1.3.2" omegaconf

# Quick self-check: both the vendored colosseum package and the patched rlbench stack import cleanly.
# colosseum.rlbench.utils -> pyrep's cffi backend dlopens libcoppeliaSim.so.1, so the self-check needs
# CoppeliaSim's library path too (the same exports as train.sh/eval.sh).
COLOSSEUM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FINETUNE_DIR="$(dirname "${COLOSSEUM_DIR}")"
export COPPELIASIM_ROOT="${COPPELIASIM_ROOT:-${FINETUNE_DIR}/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04}"
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM=offscreen
PYTHONPATH="${COLOSSEUM_DIR}:${COLOSSEUM_DIR}/robot-colosseum:${FINETUNE_DIR}/bridgevla/libs/PyRep:${FINETUNE_DIR}/bridgevla/libs/RLBench:${PYTHONPATH:-}" \
python3 - <<'PY'
import colosseum
from colosseum import TASKS_PY_FOLDER, TASKS_TTM_FOLDER, ASSETS_CONFIGS_FOLDER
from colosseum.rlbench.utils import name_to_class
import os
assert os.path.isdir(TASKS_PY_FOLDER), TASKS_PY_FOLDER
assert os.path.isdir(ASSETS_CONFIGS_FOLDER), ASSETS_CONFIGS_FOLDER
print("[install_colosseum] colosseum package OK:", os.path.dirname(colosseum.__file__))
PY

echo "[install_colosseum] done."
