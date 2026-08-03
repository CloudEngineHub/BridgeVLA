#!/usr/bin/env bash
# BridgeVLA++ — grounding pre-training environment installer
#
# Creates (or reuses) the conda env `bridgevla_plus_pretrain` and installs
# everything needed by pretrain/pretrain.sh and pretrain/pretrain_eval.sh.
# Idempotent: finished steps are detected and skipped; safe to re-run.
#
# Usage:
#   bash pretrain/install_pretrain.sh
#
# Pre-training works on plain 2D images (the RoboPoint corpus), so this env is
# deliberately small: no simulator (CoppeliaSim / PyRep / RLBench), no CUDA
# extension builds, no pytorch3d. Its packages are a strict SUBSET of the
# GemBench env with identical pins (python 3.9, torch 2.5.1+cu121,
# transformers 4.51.3) — if you already built `bridgevla_plus_gembench`, you
# may skip this installer and run pre-training there instead:
#   PRETRAIN_CONDA_ENV=bridgevla_plus_gembench bash pretrain/pretrain.sh
#
# Install philosophy: NO `pip install -e` anywhere. All in-repo code is
# resolved through each launch script's PYTHONPATH; the conda env contains no
# reference to this checkout's absolute path.
#
# Everything installs from official upstreams (pypi.org, download.pytorch.org).
# If your network needs a mirror or proxy, configure it in your own
# environment (pip.conf, https_proxy); this script does not modify any global
# configuration.
#
# Overrides:
#   PRETRAIN_CONDA_ENV  conda env name          (default bridgevla_plus_pretrain)
#   CONDA_BASE          conda install prefix     (default: auto-detect)
#   TORCH_INDEX_URL     torch wheel index        (default: official cu121)
#   SKIP_VERIFY=1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRIDGEVLA_ROOT="$(dirname "${SCRIPT_DIR}")"
FINETUNE_DIR="${BRIDGEVLA_ROOT}/finetune"
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
ENV_NAME="${PRETRAIN_CONDA_ENV:-bridgevla_plus_pretrain}"
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

# same pip/setuptools pins as the other installers, for reproducibility
pip install --upgrade pip==25.0.1
pip install setuptools==76.1.0 wheel pyyaml

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
log "STEP 3: pre-training runtime deps"
pip install \
    numpy scipy einops matplotlib \
    transformers==4.51.3 safetensors \
    "accelerate>=0.26.0" \
    yacs swanlab tqdm typing_extensions huggingface_hub

# STEP 4: final import self-check (same PYTHONPATH as pretrain.sh)
if [ "${SKIP_VERIFY:-0}" = "1" ]; then
    warn "STEP 4 skipped (SKIP_VERIFY=1)"
else
    log "STEP 4: import self-check"
    export PYTHONPATH="\
${FINETUNE_DIR}:\
${LIBS_DIR}/point-renderer:\
${LIBS_DIR}/peract_colab:\
${LIBS_DIR}/YARR:\
${FINETUNE_DIR}/GemBench:${PYTHONPATH:-}"
    python - <<'PY'
import importlib, sys
import torch  # first, so libc10.so enters the loader path
MODS = ["torch", "xformers", "transformers", "safetensors", "accelerate",
        "numpy", "scipy", "einops", "matplotlib", "PIL", "yaml",
        "yacs", "swanlab", "tqdm", "huggingface_hub",
        # the exact model-side modules pretrain.py imports
        "bridgevla.mvt.raft_utils", "bridgevla.mvt.heads_focal",
        "bridgevla.mvt.memory", "bridgevla.mvt.utils"]
fail = []
for m in MODS:
    try:
        importlib.import_module(m); print(f"  OK   {m}")
    except Exception as e:
        print(f"  FAIL {m}: {e}"); fail.append(m)
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

  1) Download the base VLM and the RoboPoint corpus (see README — Download):
       bash scripts/download_checkpoints_hf.sh paligemma pretrain_data
       tar -xzf data/bridgevla_data/pretrain_data/coco.tar.gz \\
           -C data/bridgevla_data/pretrain_data/
  2) Run pre-training / its grounding eval (the scripts activate the env and
     set every path themselves; override the env name with PRETRAIN_CONDA_ENV
     if you changed it):
       bash pretrain/pretrain.sh
       CHECKPOINT_PATH=<run>/pretrain_epoch_<N>.pth bash pretrain/pretrain_eval.sh
  Note: this env is a strict subset of bridgevla_plus_gembench — if you built
  that one already, you can skip this env entirely and run
       PRETRAIN_CONDA_ENV=bridgevla_plus_gembench bash pretrain/pretrain.sh
============================================================================
EOF
