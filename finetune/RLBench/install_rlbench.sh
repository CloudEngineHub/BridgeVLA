#!/usr/bin/env bash
# BridgeVLA++ — RLBench / Colosseum environment installer
#
# Creates (or reuses) the conda env `bridgevla_plus_rlbench` and installs
# everything needed to train and evaluate on the RLBench and Colosseum
# benchmarks. Idempotent: finished steps are detected and skipped, so it is
# always safe to re-run after a network failure.
#
# Usage:
#   bash finetune/RLBench/install_rlbench.sh
#
# What it does:
#   1) system packages (apt; skipped when already present, SKIP_APT=1 to skip)
#   2) conda env (python 3.9) + pinned pip/setuptools
#   3) simulation source stacks via scripts/fetch_sim_stacks.sh --all
#      (clone pinned upstream commits + apply the bundled patches; RLBench's
#      license forbids redistributing its sources, so they are rebuilt here.
#      TWO stacks coexist and are NOT interchangeable — see
#      finetune/bridgevla/libs/rlbench_patches/README.md):
#        libs/RLBench + libs/PyRep                     shared stack (also used
#                                                      by GemBench/memoryBench)
#        libs/RLBench_peract587 + libs/PyRep_stepjam231  RLBench-benchmark stack
#      train.sh / eval.sh pick the right stack automatically via
#      RLBENCH_SIM_STACK / PYREP_SIM_STACK.
#   4) CoppeliaSim 4.1 download + extraction under finetune/
#   5) python dependencies (torch 2.5.1+cu121, xformers, transformers, ...)
#   6) in-place C/CUDA extension builds (PyRep cffi ×2, point-renderer)
#   7) final import self-check with the same PYTHONPATH train.sh uses
#
# Install philosophy: NO `pip install -e` anywhere. All in-repo / cloned code
# is resolved through each launch script's PYTHONPATH; compiled artifacts stay
# inside the source trees (build_ext --inplace). The conda env contains no
# reference to this checkout's absolute path — renaming or duplicating the
# repo is safe.
#
# Everything installs from official upstreams (pypi.org, github.com,
# download.pytorch.org). If your network needs a mirror or proxy, configure it
# in your own environment (pip.conf, git config, https_proxy); this script
# does not modify any global configuration.
#
# Overrides:
#   RLBENCH_CONDA_ENV   conda env name          (default bridgevla_plus_rlbench)
#   CONDA_BASE          conda install prefix     (default: auto-detect)
#   TORCH_INDEX_URL     torch wheel index        (default: official cu121)
#   RLBENCH_UPSTREAM / RLBENCH_PERACT_UPSTREAM / PYREP_UPSTREAM /
#   PYREP_STEPJAM_UPSTREAM   git upstreams — point at your own forks to guard
#                            against upstream deletion
#   SKIP_APT=1 / SKIP_APT_UPDATE=1 / SKIP_VERIFY=1
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

# has_pip_pkg <import_name> — true when the module imports in the active env
has_pip_pkg() {
    python -c "import importlib, sys; importlib.import_module(sys.argv[1])" "$1" 2>/dev/null
}

TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
ENV_NAME="${RLBENCH_CONDA_ENV:-bridgevla_plus_rlbench}"
PY_VERSION="${PY_VERSION:-3.9}"

# STEP 1: system packages (apt)
APT_PKGS=(
    libffi-dev xvfb libfontconfig1 ffmpeg
    libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0
    libxcb-cursor0 libxcb-xinerama0
    libxcb-xinput0 libx11-xcb1 libxcb1 libxcb-render0 libxcb-shm0
    libxcb-xfixes0 libxcb-shape0 libxcb-randr0 libxcb-sync1 libxcb-util1
    libxcb-glx0 libxcb-xkb1 libxkbcommon-x11-0
)
if [ "${SKIP_APT:-0}" = "1" ]; then
    warn "STEP 1 skipped (SKIP_APT=1)"
else
    log "STEP 1: system packages (dpkg check)"
    MISSING=()
    for pkg in "${APT_PKGS[@]}"; do
        dpkg -s "$pkg" >/dev/null 2>&1 || MISSING+=("$pkg")
    done
    if [ "${#MISSING[@]}" -eq 0 ]; then
        log "all ${#APT_PKGS[@]} system packages already present — skipping apt"
    else
        log "installing ${#MISSING[@]} missing packages: ${MISSING[*]}"
        SUDO=""
        if [ "$(id -u)" -ne 0 ]; then
            command -v sudo >/dev/null 2>&1 || die "missing apt packages but no sudo; ask an admin to install: ${MISSING[*]}"
            SUDO="sudo"
        fi
        APT_OPTS=(-o "Acquire::http::Timeout=5" -o "Acquire::https::Timeout=5")
        if [ "${SKIP_APT_UPDATE:-0}" != "1" ]; then
            $SUDO apt-get "${APT_OPTS[@]}" update || warn "apt-get update partially failed, continuing with install"
        fi
        $SUDO apt-get "${APT_OPTS[@]}" install -y "${MISSING[@]}"
    fi
fi

# STEP 2: conda env + pinned pip/setuptools
log "STEP 2: conda env [${ENV_NAME}] (python=${PY_VERSION})"
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
# `setup.py build_ext --inplace` builds below (newer setuptools drops APIs
# PyRep's cffi build relies on).
pip install --upgrade pip==25.0.1
pip install setuptools==76.1.0 wheel ninja pyyaml

# STEP 3: simulation source stacks (clone pinned commits + apply patches)
log "STEP 3: simulation source stacks (both RLBench/PyRep stacks)"
bash "${BRIDGEVLA_ROOT}/scripts/fetch_sim_stacks.sh" --all

# STEP 4: CoppeliaSim 4.1
COPP_TAR="CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz"
export COPPELIASIM_ROOT="${FINETUNE_DIR}/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
if [ ! -d "${COPPELIASIM_ROOT}" ]; then
    if [ ! -f "${FINETUNE_DIR}/${COPP_TAR}" ]; then
        log "STEP 4: downloading CoppeliaSim 4.1 ..."
        wget -P "${FINETUNE_DIR}" "https://downloads.coppeliarobotics.com/V4_1_0/${COPP_TAR}" \
            || wget -P "${FINETUNE_DIR}" "https://www.coppeliarobotics.com/files/V4_1_0/${COPP_TAR}"
    fi
    log "STEP 4: extracting ${COPP_TAR} ..."
    tar -xf "${FINETUNE_DIR}/${COPP_TAR}" -C "${FINETUNE_DIR}"
else
    skip "CoppeliaSim (already extracted)"
fi
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${COPPELIASIM_ROOT}"
export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}"
export QT_QPA_PLATFORM=offscreen

# STEP 5: python dependencies
log "STEP 5: PyTorch 2.5.1 + cu121 + xformers"
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

log "STEP 5: bridgevla runtime deps"
pip install \
    numpy scipy einops pyrender \
    transformers==4.51.3 \
    omegaconf natsort cffi pandas pyquaternion matplotlib \
    bitsandbytes transforms3d \
    "accelerate>=0.26.0"

log "STEP 5: simulator / logging / misc deps"
# --ignore-installed blinker: the distutils-installed system blinker 1.4
# cannot be uninstalled by pip
if has_pip_pkg open3d; then skip "open3d"; else pip install --ignore-installed blinker open3d; fi
pip install yacs swanlab pyqt6 ffmpeg-python
pip install tensorboard moviepy psutil timeout-decorator hydra-core \
    imageio trimesh tqdm typing_extensions huggingface_hub

if has_pip_pkg clip; then skip "CLIP"; else
    pip install "git+https://github.com/openai/CLIP.git"
fi
if has_pip_pkg pytorch3d; then skip "pytorch3d"; else
    log "building pytorch3d from source (5–15 min) ..."
    pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
fi

# PyRep / RLBench requirements (the code itself is used via PYTHONPATH)
pip install -r "${LIBS_DIR}/PyRep/requirements.txt"
pip install -r "${LIBS_DIR}/RLBench/requirements.txt"

# headless OpenCV only (GUI opencv breaks under xvfb; some deps pull it in).
# --force-reinstall repairs the case where a GUI variant coexisted with the
# headless one: uninstalling the GUI package deletes the shared cv2/ files
# while the headless dist-info survives, so a plain install would no-op.
log "STEP 5: switching to opencv-python-headless"
pip uninstall -y opencv-python opencv-contrib-python 2>/dev/null || true
if has_pip_pkg cv2; then
    skip "opencv-python-headless"
else
    pip install --force-reinstall --no-deps opencv-python-headless
fi

# STEP 6: in-place C/CUDA extension builds
log "STEP 6: PyRep cffi builds (both stacks)"
if [ -f "${LIBS_DIR}/PyRep/pyrep/backend/_sim_cffi.abi3.so" ]; then
    skip "PyRep cffi (.so exists)"
else
    (cd "${LIBS_DIR}/PyRep" && python setup.py build_ext --inplace)
fi
if ls "${LIBS_DIR}/PyRep_stepjam231/pyrep/backend/"_sim_cffi*.so >/dev/null 2>&1; then
    skip "PyRep_stepjam231 cffi (.so exists)"
else
    (cd "${LIBS_DIR}/PyRep_stepjam231" && python setup.py build_ext --inplace)
fi

log "STEP 6: point-renderer CUDA extension"
# arch-specific — never copy the built .so across machines
PR_DIR="${LIBS_DIR}/point-renderer"
_PR_CHECK='import torch; from point_renderer import _C; print("[install] _C OK:", _C.__file__)'
if PYTHONPATH="${PR_DIR}" python -c "${_PR_CHECK}" 2>/dev/null; then
    skip "point_renderer._C"
else
    (cd "${PR_DIR}" && python setup.py build_ext --inplace)
    PYTHONPATH="${PR_DIR}" python -c "${_PR_CHECK}"
fi

# STEP 7: final import self-check (same PYTHONPATH as train.sh / eval.sh)
if [ "${SKIP_VERIFY:-0}" = "1" ]; then
    warn "STEP 7 skipped (SKIP_VERIFY=1)"
else
    log "STEP 7: import self-check (shared stack, then RLBench-benchmark stack)"
    _COMMON_PP="${FINETUNE_DIR}:${PR_DIR}:${LIBS_DIR}/peract_colab:${LIBS_DIR}/YARR"
    # pyrep dlopens libcoppeliaSim.so.1 through cffi at import time
    PYTHONPATH="${_COMMON_PP}:${LIBS_DIR}/PyRep:${LIBS_DIR}/RLBench" python - <<'PY'
import importlib, sys
import torch  # first, so libc10.so enters the loader path
MODS = ["torch", "xformers", "transformers", "accelerate", "bitsandbytes",
        "numpy", "scipy", "einops", "pyrender", "omegaconf", "natsort",
        "cv2", "open3d", "swanlab", "yacs", "clip", "tqdm",
        "pyrep", "rlbench", "bridgevla", "peract_colab", "yarr", "point_renderer"]
fail = []
for m in MODS:
    try:
        importlib.import_module(m); print(f"  OK   {m}")
    except Exception as e:
        print(f"  FAIL {m}: {e}"); fail.append(m)
import pyrep, rlbench
print("\n  shared stack:", rlbench.__file__)
if fail:
    sys.exit(f"imports failed: {fail}")
PY
    PYTHONPATH="${_COMMON_PP}:${LIBS_DIR}/PyRep_stepjam231:${LIBS_DIR}/RLBench_peract587" \
        python -c 'import pyrep, rlbench; print("  peract587 stack:", rlbench.__file__)'
    log "self-check passed"
fi

cat <<EOF

============================================================================
Done. Next steps:

  1) Download checkpoints / prepare data (see README — Download):
       bash scripts/download_checkpoints_hf.sh pretrain rlbench
       bash scripts/download_datasets.sh rlbench
  2) Train / evaluate (the scripts activate the env and set every path
     themselves; override the env name with RLBENCH_CONDA_ENV if you changed it):
       bash finetune/RLBench/train.sh
       EVAL_TASKS=close_jar bash finetune/RLBench/eval.sh
  3) Colosseum shares this env — finish it with:
       bash finetune/Colosseum/install_colosseum.sh
============================================================================
EOF
