#!/usr/bin/env bash
# BridgeVLA++ — real-robot TRAINING environment installer
#
# Target machine: the GPU training server. Creates (or reuses) the conda env
# `bridgevla_plus_real_train` and installs everything finetune/real/train.sh
# needs to train on self-collected real-robot data. Idempotent: finished
# steps are detected and skipped; safe to re-run.
#
# Usage:
#   bash finetune/real/install_real_train.sh
#
# Not installed here (not needed for offline training):
#   - simulator bits (CoppeliaSim / PyRep / RLBench)
#   - robot/camera SDKs and the inference server — DEPLOYMENT on the robot
#     workstation is a different machine and env (python 3.10, torch
#     2.6.0+cu124): finetune/real/install_real_deploy.sh
#
# The packages here are a strict SUBSET of the GemBench env with identical
# pins (python 3.9, torch 2.5.1+cu121, transformers 4.51.3) — if you already
# built `bridgevla_plus_gembench`, you may skip this installer and train there:
#   REAL_TRAIN_CONDA_ENV=bridgevla_plus_gembench bash finetune/real/train.sh
#
# Install philosophy: NO `pip install -e` anywhere. All in-repo code is
# resolved through each launch script's PYTHONPATH; compiled artifacts stay
# inside the source trees (build_ext --inplace). The conda env contains no
# reference to this checkout's absolute path.
#
# Everything installs from official upstreams (pypi.org, github.com,
# download.pytorch.org). If your network needs a mirror or proxy, configure it
# in your own environment (pip.conf, git config, https_proxy); this script
# does not modify any global configuration.
#
# Overrides:
#   REAL_TRAIN_CONDA_ENV  conda env name          (default bridgevla_plus_real_train)
#   CONDA_BASE            conda install prefix     (default: auto-detect)
#   TORCH_INDEX_URL       torch wheel index        (default: official cu121)
#   SKIP_VERIFY=1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FINETUNE_DIR="$(dirname "${SCRIPT_DIR}")"
BRIDGEVLA_ROOT="$(dirname "${FINETUNE_DIR}")"
LIBS_DIR="${FINETUNE_DIR}/bridgevla/libs"

log()  { printf '\n\033[1;34m[install]\033[0m %s\n' "$*"; }
warn() { printf '\n\033[1;33m[warn]\033[0m %s\n' "$*"; }
die()  { printf '\n\033[1;31m[fatal]\033[0m %s\n' "$*" >&2; exit 1; }
skip() { printf '\033[0;32m[skip]\033[0m %s already installed\n' "$*"; }

# quieter, deterministic pip: no "new release available" self-check chatter
export PIP_DISABLE_PIP_VERSION_CHECK=1

has_pip_pkg() {
    python -c "import importlib, sys; importlib.import_module(sys.argv[1])" "$1" 2>/dev/null
}

TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
ENV_NAME="${REAL_TRAIN_CONDA_ENV:-bridgevla_plus_real_train}"
PY_VERSION="${PY_VERSION:-3.9}"

# STEP 1: conda env + pinned pip/setuptools
log "STEP 1: conda env [${ENV_NAME}] (python=${PY_VERSION})"
if [ -z "${CONDA_BASE:-}" ]; then
    CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    [ -z "${CONDA_BASE}" ] && [ -d "${HOME}/miniconda3" ] && CONDA_BASE="${HOME}/miniconda3"
    [ -z "${CONDA_BASE}" ] && [ -d "${HOME}/anaconda3" ] && CONDA_BASE="${HOME}/anaconda3"
fi
[ -n "${CONDA_BASE}" ] || die "conda not found — install miniconda or set CONDA_BASE"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    log "conda env ${ENV_NAME} already exists — reusing"
else
    conda create -n "${ENV_NAME}" "python=${PY_VERSION}" -y
fi
conda activate "${ENV_NAME}"
python -c "import sys; print('[install] python =', sys.version)"

# pip 25.0.1 + setuptools 76.1.0: known-good combo for the legacy
# `setup.py build_ext --inplace` build below.
pip install --upgrade pip==25.0.1
pip install setuptools==76.1.0 wheel ninja pyyaml

# STEP 2: PyTorch 2.5.1 + cu121 + xformers
log "STEP 2: PyTorch 2.5.1 + cu121 + xformers"
if has_pip_pkg torch; then
    skip "torch ($(python -c 'import torch; print(torch.__version__)'))"
else
    pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
        --index-url "${TORCH_INDEX_URL}"
fi
if has_pip_pkg xformers; then skip "xformers"; else
    # 0.0.28.post3 is the last xformers release compatible with torch 2.5.1
    pip install xformers==0.0.28.post3 --index-url "${TORCH_INDEX_URL}"
fi

# STEP 3: python dependencies
log "STEP 3: bridgevla + real-pipeline runtime deps"
pip install \
    numpy scipy einops matplotlib \
    transformers==4.51.3 safetensors \
    "accelerate>=0.26.0" \
    transforms3d pyrender trimesh tensorboard \
    yacs swanlab tqdm typing_extensions huggingface_hub

if has_pip_pkg pytorch3d; then skip "pytorch3d"; else
    log "building pytorch3d from source (5–15 min) ..."
    pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
fi

# headless OpenCV only (training runs on a display-less server; some deps pull
# in the GUI variant). --force-reinstall repairs the case where a GUI variant
# coexisted with the headless one: uninstalling the GUI package deletes the
# shared cv2/ files while the headless dist-info survives, so a plain install
# would no-op.
log "STEP 3: switching to opencv-python-headless"
pip uninstall -y opencv-python opencv-contrib-python 2>/dev/null || true
if has_pip_pkg cv2; then
    skip "opencv-python-headless"
else
    pip install --force-reinstall --no-deps opencv-python-headless
fi

# STEP 4: point-renderer CUDA extension (in-place, arch-specific — never copy
# the built .so across machines)
log "STEP 4: point-renderer CUDA extension"
PR_DIR="${LIBS_DIR}/point-renderer"
_PR_CHECK='import torch; from point_renderer import _C; print("[install] _C OK:", _C.__file__)'
if PYTHONPATH="${PR_DIR}" python -c "${_PR_CHECK}" 2>/dev/null; then
    skip "point_renderer._C"
else
    (cd "${PR_DIR}" && python setup.py build_ext --inplace)
    PYTHONPATH="${PR_DIR}" python -c "${_PR_CHECK}"
fi

# STEP 5: final import self-check (same PYTHONPATH as real/train.sh)
if [ "${SKIP_VERIFY:-0}" = "1" ]; then
    warn "STEP 5 skipped (SKIP_VERIFY=1)"
else
    log "STEP 5: import self-check"
    export PYTHONPATH="\
${FINETUNE_DIR}:\
${PR_DIR}:\
${LIBS_DIR}/peract_colab:\
${LIBS_DIR}/YARR:\
${FINETUNE_DIR}/GemBench:${PYTHONPATH:-}"
    python - <<'PY'
import importlib, sys
import torch  # first, so libc10.so enters the loader path
MODS = ["torch", "xformers", "transformers", "safetensors", "accelerate",
        "numpy", "scipy", "einops", "matplotlib", "PIL", "yaml", "cv2",
        "transforms3d", "pyrender", "trimesh", "pytorch3d",
        "yacs", "swanlab", "tqdm", "huggingface_hub",
        "bridgevla", "peract_colab", "yarr", "point_renderer",
        # the full agent chain train.py imports (pulls GemBench/RLBench utils)
        "bridgevla.models.bridgevla_agent",
        "real.real_dataset", "real.visualize"]
fail = []
for m in MODS:
    try:
        importlib.import_module(m); print(f"  OK   {m}")
    except Exception as e:
        print(f"  FAIL {m}: {e}"); fail.append(m)
from point_renderer import _C  # noqa: F401
from torch.utils.tensorboard import SummaryWriter  # noqa: F401
import numpy as _np
print(f"\ntorch={torch.__version__}  cuda={torch.cuda.is_available()}  numpy={_np.__version__}")
if fail:
    sys.exit(f"imports failed: {fail}")
print("ALL OK")
PY
    log "self-check passed"
fi

cat <<EOF

============================================================================
Done. Next steps:

  1) Real-robot data is self-collected (finetune/real/data_collection/, run on
     the robot workstation) and expected under
       data/bridgevla_data/Real/<collection>/
     Also fetch the training warm start + base VLM:
       bash scripts/download_checkpoints_hf.sh pretrain paligemma
  2) Train (the script activates the env and sets every path itself; override
     the env name with REAL_TRAIN_CONDA_ENV if you changed it):
       bash finetune/real/train.sh
  3) Deployment (inference server + robot client) runs on the robot
     workstation in its own env — finetune/real/install_real_deploy.sh.
  Note: this env is a strict subset of bridgevla_plus_gembench — if you built
  that one already, you can skip this env entirely and train with
       REAL_TRAIN_CONDA_ENV=bridgevla_plus_gembench bash finetune/real/train.sh
============================================================================
EOF
