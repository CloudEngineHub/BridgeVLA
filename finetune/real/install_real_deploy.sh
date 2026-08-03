#!/usr/bin/env bash
# BridgeVLA++ — real-robot DEPLOYMENT environment installer
#
# Target machine: the robot workstation (Dobot + ZED / RealSense) that runs
# the inference server and the robot client:
#     python finetune/real/rvt_our/eval_flask_app.py    # model server
#     python finetune/real/rvt_our/eval_client.py       # robot client
#
# It creates its OWN conda env `bridgevla_plus_real_deploy` (python 3.10,
# torch 2.6.0+cu124 — matching the workstation's newer driver stack).
#
# ⚠ Do NOT install this into a training env (`bridgevla_plus_real_train` /
#   `bridgevla_plus_gembench`): those are pinned to torch 2.5.1+cu121 and
#   would be broken by the 2.6.0+cu124 wheels here. TRAINING on real-robot
#   data happens on the GPU server in its own env — see
#   finetune/real/install_real_train.sh (finetune/real/train.sh, override
#   with REAL_TRAIN_CONDA_ENV).
#
# Not installed here (simulation-only): CoppeliaSim / PyRep / RLBench.
# Hardware SDKs are installed by their vendors' installers, not pip:
#   pyzed         — Stereolabs ZED SDK   (https://www.stereolabs.com/developers/release)
#   pyrealsense2  — Intel RealSense SDK  (apt librealsense2-* + pip pyrealsense2)
#
# Everything installs from official upstreams (pypi.org, github.com,
# download.pytorch.org). If your network needs a mirror or proxy, configure it
# in your own environment; this script does not modify any global config.
#
# Overrides:
#   REAL_DEPLOY_CONDA_ENV  env name          (default bridgevla_plus_real_deploy)
#   CONDA_BASE             conda prefix       (default: auto-detect)
#   TORCH_INDEX_URL        torch wheel index  (default: official cu124)
set -euo pipefail

FINETUNE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BRIDGEVLA_ROOT="$(dirname "${FINETUNE_DIR}")"
LIBS_DIR="${FINETUNE_DIR}/bridgevla/libs"

log()  { printf '\n\033[1;34m[install]\033[0m %s\n' "$*"; }
die()  { printf '\n\033[1;31m[fatal]\033[0m %s\n' "$*" >&2; exit 1; }
skip() { printf '\033[0;32m[skip]\033[0m %s already installed\n' "$*"; }

# quieter, deterministic pip: no "new release available" self-check chatter
export PIP_DISABLE_PIP_VERSION_CHECK=1
has_pip_pkg() {
    python -c "import importlib, sys; importlib.import_module(sys.argv[1])" "$1" 2>/dev/null
}

TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
ENV_NAME="${REAL_DEPLOY_CONDA_ENV:-bridgevla_plus_real_deploy}"
PY_VERSION="${PY_VERSION:-3.10}"

echo "============================================================"
echo "  BridgeVLA++  real-robot deployment installer"
echo "  repo   : ${BRIDGEVLA_ROOT}"
echo "  env    : ${ENV_NAME} (python ${PY_VERSION}, torch 2.6.0+cu124)"
echo "============================================================"

log "STEP 1: conda env [${ENV_NAME}]"
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
pip install --upgrade pip setuptools wheel ninja pyyaml

# STEP 2: PyTorch 2.6.0 + cu124 + xformers
log "STEP 2: PyTorch + xformers"
if has_pip_pkg torch; then
    skip "torch ($(python -c 'import torch; print(torch.__version__)'))"
else
    pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
        --index-url "${TORCH_INDEX_URL}"
fi
if has_pip_pkg xformers; then skip "xformers"; else
    # 0.0.29.post3 matches torch 2.6.0
    pip install xformers==0.0.29.post3 --index-url "${TORCH_INDEX_URL}"
fi

# STEP 3: bridgevla runtime deps (code itself is used via PYTHONPATH)
log "STEP 3: bridgevla runtime deps"
pip install \
    numpy scipy einops pyrender \
    transformers==4.51.3 \
    omegaconf natsort cffi pandas pyquaternion matplotlib \
    bitsandbytes transforms3d \
    "accelerate>=0.26.0"

# YARR / peract_colab runtime deps
pip install tensorboard moviepy psutil timeout-decorator hydra-core \
    imageio trimesh tqdm typing_extensions huggingface_hub
pip install yacs swanlab h5py

log "STEP 4: CLIP + pytorch3d"
if has_pip_pkg clip; then skip "CLIP"; else
    pip install "git+https://github.com/openai/CLIP.git"
fi
if has_pip_pkg pytorch3d; then skip "pytorch3d"; else
    # --no-build-isolation: its setup.py imports torch at build time
    log "building pytorch3d from source (5–15 min) ..."
    pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"
fi

# STEP 5: point-renderer CUDA extension (in-place, arch-specific)
log "STEP 5: point-renderer"
PR_DIR="${LIBS_DIR}/point-renderer"
_PR_CHECK='import torch; from point_renderer import _C; print("[install] _C OK:", _C.__file__)'
if [ -d "${PR_DIR}" ]; then
    if PYTHONPATH="${PR_DIR}" python -c "${_PR_CHECK}" 2>/dev/null; then
        skip "point_renderer._C"
    else
        (cd "${PR_DIR}" && python setup.py build_ext --inplace)
        PYTHONPATH="${PR_DIR}" python -c "${_PR_CHECK}"
    fi
else
    die "point-renderer not found at ${PR_DIR} (incomplete checkout?)"
fi

# STEP 6: real-robot specific packages
log "STEP 6: real-robot packages"
pip install flask open3d pymodbus
# real-robot GUI display wants the non-headless OpenCV
pip install opencv-python
echo "[info] pyzed / pyrealsense2 are installed by their vendor SDKs — see header"

# Done
cat <<EOF

============================================================================
Done. On this workstation:

  conda activate ${ENV_NAME}
  python finetune/real/rvt_our/eval_flask_app.py     # model server
  ARM_IP=<robot-ip> LOCAL_IP=<this-host-ip> \\
  python finetune/real/rvt_our/eval_client.py        # robot client

The eval scripts derive PYTHONPATH and checkpoint paths themselves; see
README ("Real robot") for the env-var overrides (REAL_EVAL_RUN_DIR, ...).
ZED SDK (pyzed) / RealSense SDK (pyrealsense2) must be installed separately.
============================================================================
EOF
