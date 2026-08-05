#!/usr/bin/env bash
# BridgeVLA++ — GemBench environment installer
#
# Creates (or reuses) the conda env `bridgevla_plus_gembench` and installs
# everything needed for:
#   - GemBench      (train + server/client eval)
#   - memoryBench   (same simulator stack; its 3 tasks are auto-installed by
#                    the memoryBench run scripts)
#   - RMBench       (the BridgeVLA++ policy side: RMBench_vla/train.sh and the
#                    eval policy server; the SAPIEN client lives in its own
#                    env — see finetune/RMBench/install_rmbench.sh, which runs
#                    THIS script in --policy-only mode automatically)
#
# Grounding pre-training and real-robot training have their own installers
# (pretrain/install_pretrain.sh, finetune/real/install_real_train.sh). Their
# packages are strict subsets of this env with identical pins, so to keep a
# single env you may instead point PRETRAIN_CONDA_ENV / REAL_TRAIN_CONDA_ENV
# at this one.
#
# Idempotent: finished steps are detected and skipped; safe to re-run.
#
# Usage:
#   bash finetune/GemBench/install_gembench.sh                 # full install
#   bash finetune/GemBench/install_gembench.sh --policy-only   # no simulator
#
# --policy-only installs only the BridgeVLA++ policy runtime (conda env + pip
# stack + point-renderer CUDA build) and skips everything CoppeliaSim-related:
# the apt xcb/xvfb packages, the RLBench/PyRep source stack, CoppeliaSim
# itself and the PyRep cffi build. That is all the RMBench policy side needs —
# its simulator is SAPIEN, in a separate env — so
# finetune/RMBench/install_rmbench.sh invokes this mode for you. Running the
# FULL installer later upgrades the same env in place (idempotence means only
# the skipped simulator steps are added).
#
# Note: this env and the RLBench/Colosseum env (install_rlbench.sh) share the
# exact same core pins (python 3.9, torch 2.5.1+cu121, transformers 4.51.3).
# The benchmarks differ only in which simulation *source stack* the launch
# scripts put on PYTHONPATH, not in installed packages — so you may install
# both benchmark families into ONE env if you prefer: run both installers
# with RLBENCH_CONDA_ENV / GEMBENCH_CONDA_ENV set to the same name.
#
# Install philosophy: NO `pip install -e` anywhere. All in-repo / cloned code
# is resolved through each launch script's PYTHONPATH; compiled artifacts stay
# inside the source trees (build_ext --inplace). The conda env contains no
# reference to this checkout's absolute path.
#
# Everything installs from official upstreams (pypi.org, github.com,
# download.pytorch.org). If your network needs a mirror or proxy, configure it
# in your own environment (pip.conf, git config, https_proxy); this script
# does not modify any global configuration.
#
# Overrides:
#   GEMBENCH_CONDA_ENV  conda env name          (default bridgevla_plus_gembench)
#   GEMBENCH_POLICY_ONLY=1                       same as --policy-only
#   CONDA_BASE          conda install prefix     (default: auto-detect)
#   TORCH_INDEX_URL     torch wheel index        (default: official cu121)
#   RLBENCH_UPSTREAM / PYREP_UPSTREAM   git upstreams for the shared sim stack
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

has_pip_pkg() {
    python -c "import importlib, sys; importlib.import_module(sys.argv[1])" "$1" 2>/dev/null
}

POLICY_ONLY="${GEMBENCH_POLICY_ONLY:-0}"
for arg in "$@"; do
    case "${arg}" in
        --policy-only) POLICY_ONLY=1 ;;
        *) die "unknown argument '${arg}' (supported: --policy-only)" ;;
    esac
done

TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
ENV_NAME="${GEMBENCH_CONDA_ENV:-bridgevla_plus_gembench}"
PY_VERSION="${PY_VERSION:-3.9}"
if [ "${POLICY_ONLY}" = "1" ]; then
    log "policy-only mode: simulator steps (apt xcb/xvfb, RLBench/PyRep sources, CoppeliaSim, PyRep build) will be skipped"
fi

# STEP 1: system packages (apt)
APT_PKGS=(
    libffi-dev xvfb libfontconfig1 ffmpeg
    libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0
    libxcb-cursor0 libxcb-xinerama0
    libxcb-xinput0 libx11-xcb1 libxcb1 libxcb-render0 libxcb-shm0
    libxcb-xfixes0 libxcb-shape0 libxcb-randr0 libxcb-sync1 libxcb-util1
    libxcb-glx0 libxcb-xkb1 libxkbcommon-x11-0
)
if [ "${POLICY_ONLY}" = "1" ]; then
    log "STEP 1 skipped (--policy-only): the apt packages only serve CoppeliaSim/Qt/xvfb"
elif [ "${SKIP_APT:-0}" = "1" ]; then
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
# `setup.py build_ext --inplace` builds below.
pip install --upgrade pip==25.0.1
pip install setuptools==76.1.0 wheel ninja pyyaml

# STEP 3: shared simulation source stack (clone pinned commits + patches)
if [ "${POLICY_ONLY}" = "1" ]; then
    log "STEP 3 skipped (--policy-only): no RLBench/PyRep sources needed"
else
    log "STEP 3: shared simulation stack (rjgpinel RLBench + cshizhe PyRep)"
    bash "${BRIDGEVLA_ROOT}/scripts/fetch_sim_stacks.sh" --shared
fi

# STEP 4: CoppeliaSim 4.1
if [ "${POLICY_ONLY}" = "1" ]; then
    log "STEP 4 skipped (--policy-only): CoppeliaSim not needed"
else
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
fi

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

log "STEP 5: GemBench / YARR / peract / renderer deps"
# --ignore-installed blinker: the distutils-installed system blinker 1.4
# cannot be uninstalled by pip
if has_pip_pkg open3d; then skip "open3d"; else pip install --ignore-installed blinker open3d; fi
# h5py: RMBench_vla/train.sh runs in this env and reads the keyframe HDF5 data
pip install yacs swanlab msgpack_numpy jsonlines lmdb h5py ffmpeg-python pyqt6
pip install tensorboard moviepy psutil timeout-decorator hydra-core
pip install imageio trimesh meshcat
pip install tqdm typing_extensions huggingface_hub

if has_pip_pkg clip; then skip "CLIP"; else
    pip install "git+https://github.com/openai/CLIP.git"
fi
if has_pip_pkg pytorch3d; then skip "pytorch3d"; else
    log "building pytorch3d from source (5–15 min) ..."
    pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
fi

# PyRep / RLBench requirements (the code itself is used via PYTHONPATH)
if [ "${POLICY_ONLY}" = "1" ]; then
    log "STEP 5: PyRep/RLBench requirements skipped (--policy-only)"
else
    pip install -r "${LIBS_DIR}/PyRep/requirements.txt"
    pip install -r "${LIBS_DIR}/RLBench/requirements.txt"
fi

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
if [ "${POLICY_ONLY}" = "1" ]; then
    log "STEP 6: PyRep cffi build skipped (--policy-only)"
else
    log "STEP 6: PyRep cffi build"
    if [ -f "${LIBS_DIR}/PyRep/pyrep/backend/_sim_cffi.abi3.so" ]; then
        skip "PyRep cffi (.so exists)"
    else
        (cd "${LIBS_DIR}/PyRep" && python setup.py build_ext --inplace)
    fi
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

# STEP 7: final import self-check (same PYTHONPATH as train.sh / run_*.sh)
if [ "${SKIP_VERIFY:-0}" = "1" ]; then
    warn "STEP 7 skipped (SKIP_VERIFY=1)"
else
    log "STEP 7: import self-check"
    # pyrep dlopens libcoppeliaSim.so.1 through cffi at import time, hence
    # the LD_LIBRARY_PATH / QT exports above (full mode only)
    _CHECK_PYTHONPATH="\
${FINETUNE_DIR}:\
${PR_DIR}:\
${LIBS_DIR}/peract_colab:\
${LIBS_DIR}/YARR:"
    if [ "${POLICY_ONLY}" != "1" ]; then
        _CHECK_PYTHONPATH="${_CHECK_PYTHONPATH}\
${LIBS_DIR}/PyRep:\
${LIBS_DIR}/RLBench:"
    fi
    export PYTHONPATH="${_CHECK_PYTHONPATH}${FINETUNE_DIR}/GemBench:${PYTHONPATH:-}"
    export BRIDGEVLA_CHECK_POLICY_ONLY="${POLICY_ONLY}"
    python - <<'PY'
import importlib, os, sys
import torch  # first, so libc10.so enters the loader path
MODS = ["torch", "xformers", "transformers", "accelerate", "bitsandbytes",
        "numpy", "scipy", "einops", "pyrender", "omegaconf", "natsort",
        "cv2", "open3d", "swanlab", "yacs", "msgpack_numpy", "jsonlines", "lmdb", "h5py",
        "clip", "tqdm", "huggingface_hub",
        "bridgevla", "peract_colab", "yarr", "point_renderer"]
if os.environ.get("BRIDGEVLA_CHECK_POLICY_ONLY") == "1":
    # RMBench policy side: no simulator, but its import chain must hold
    # (bridgevla.mvt.augmentation -> pytorch3d; RMBench_vla/visualize.py ->
    #  GemBench.utils.peract_utils_gembench -> peract_colab).
    MODS += ["pytorch3d", "GemBench.utils.peract_utils_gembench"]
else:
    MODS += ["pyrep", "rlbench", "genrobo3d"]
fail = []
for m in MODS:
    try:
        importlib.import_module(m); print(f"  OK   {m}")
    except Exception as e:
        print(f"  FAIL {m}: {e}"); fail.append(m)
from point_renderer import _C  # noqa: F401
import numpy as _np
print(f"\ntorch={torch.__version__}  cuda={torch.cuda.is_available()}  numpy={_np.__version__}")
if fail:
    sys.exit(f"imports failed: {fail}")
print("ALL OK")
PY
    log "self-check passed"
fi

if [ "${POLICY_ONLY}" = "1" ]; then
cat <<EOF

============================================================================
Done (policy-only). Env [${ENV_NAME}] now holds the BridgeVLA++ policy
runtime: pip stack + point-renderer build — no CoppeliaSim / RLBench / PyRep.

It serves the RMBench policy side out of the box:
    bash finetune/RMBench_vla/train.sh                         # training
    bash finetune/RMBench/policy/BridgeVLA_Plus/eval_double_env.sh  # eval (the
        launcher starts the policy server in this env and the SAPIEN client
        in the env from finetune/RMBench/install_rmbench.sh)

To use GemBench / memoryBench later, run the FULL installer — idempotent, it
only adds the missing simulator stack to this same env:
    bash finetune/GemBench/install_gembench.sh
============================================================================
EOF
else
cat <<EOF

============================================================================
Done. Next steps:

  1) Download checkpoints / prepare data (see README — Download):
       bash scripts/download_checkpoints_hf.sh pretrain gembench
       bash scripts/download_datasets.sh gembench
  2) Train / evaluate (the scripts activate the env and set every path
     themselves; override the env name with GEMBENCH_CONDA_ENV if you
     changed it):
       bash finetune/GemBench/train.sh
       bash finetune/GemBench/run_server.sh <epoch>   # + run_client.sh
  3) memoryBench and the RMBench policy side (RMBench_vla/train.sh + the
     eval policy server) reuse THIS env — no further installs needed for
     them (RMBench's SAPIEN client env is separate:
     finetune/RMBench/install_rmbench.sh). Pre-training and real-robot
     training have their own installers (pretrain/install_pretrain.sh,
     finetune/real/install_real_train.sh); their deps are subsets of this
     env, so PRETRAIN_CONDA_ENV / REAL_TRAIN_CONDA_ENV may point here too.
============================================================================
EOF
fi
