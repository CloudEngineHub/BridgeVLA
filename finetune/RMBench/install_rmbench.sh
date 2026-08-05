#!/usr/bin/env bash
# BridgeVLA++ — RMBench installer (SAPIEN sim env + shared policy env)
#
# One command sets up BOTH environments RMBench needs:
#   STEP 0     the shared policy env (default bridgevla_plus_gembench) via
#              finetune/GemBench/install_gembench.sh --policy-only — pip stack
#              + point-renderer build only, none of GemBench's CoppeliaSim /
#              RLBench / PyRep simulator stack;
#   STEP 1-6   the conda env `bridgevla_plus_rmbench` (python 3.10,
#              torch 2.4.1+cu121) with everything the RMBench simulation
#              client needs: SAPIEN, mplib, curobo, pytorch3d.
# Idempotent: finished steps are detected and skipped; safe to re-run after a
# network failure.
#
# Usage:
#   bash finetune/RMBench/install_rmbench.sh
#
# The `bridgevla_plus_rmbench` env runs ONLY the SAPIEN side (data collection
# + the eval client in policy/BridgeVLA_Plus/eval_double_env.sh). The BridgeVLA++
# policy side — training (finetune/RMBench_vla/train.sh) and the eval policy
# server — runs in the shared policy env from STEP 0 (also used by GemBench /
# memoryBench; a full GemBench install lands in the same env).
#
# The single `pip install -e` exception of this repository lives here:
# curobo's CUDA extension requires an editable install. The checkout goes to
# finetune/RMBench/envs/curobo (gitignored).
#
# Everything installs from official upstreams (pypi.org, github.com). If your
# network needs a mirror or proxy, configure it in your own environment
# (pip.conf, git config, https_proxy); this script does not modify any global
# configuration.
#
# Overrides:
#   RMBENCH_CONDA_ENV   conda env name          (default bridgevla_plus_rmbench)
#   GEMBENCH_CONDA_ENV  policy env name         (default bridgevla_plus_gembench)
#   RMBENCH_SKIP_POLICY_ENV=1   skip STEP 0 (bring your own policy env)
#   CONDA_BASE          conda install prefix     (default: auto-detect)
#   CUROBO_UPSTREAM     curobo git upstream — point at your own fork as
#                       insurance against upstream deletion
#   SKIP_VERIFY=1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"     # finetune/RMBench

log()  { printf '\n\033[1;34m[install]\033[0m %s\n' "$*"; }
warn() { printf '\n\033[1;33m[warn]\033[0m %s\n' "$*"; }
die()  { printf '\n\033[1;31m[fatal]\033[0m %s\n' "$*" >&2; exit 1; }
skip() { printf '\033[0;32m[skip]\033[0m %s already installed\n' "$*"; }

# quieter, deterministic pip: no "new release available" self-check chatter
export PIP_DISABLE_PIP_VERSION_CHECK=1

has_pip_pkg() {
    python -c "import importlib, sys; importlib.import_module(sys.argv[1])" "$1" 2>/dev/null
}

ENV_NAME="${RMBENCH_CONDA_ENV:-bridgevla_plus_rmbench}"
PY_VERSION="${PY_VERSION:-3.10}"
CUROBO_UPSTREAM="${CUROBO_UPSTREAM:-https://github.com/NVlabs/curobo.git}"

# STEP 0: shared policy env. The RMBench policy side (RMBench_vla/train.sh +
# the eval policy server started by eval_double_env.sh) runs there, not in the
# SAPIEN env this script builds below. --policy-only keeps it lean (pip stack
# + point-renderer build; no CoppeliaSim / RLBench / PyRep), and idempotence
# means an existing full GemBench env is verified and reused as-is.
if [ "${RMBENCH_SKIP_POLICY_ENV:-0}" = "1" ]; then
    warn "STEP 0 skipped (RMBENCH_SKIP_POLICY_ENV=1) — the policy side still needs the env from finetune/GemBench/install_gembench.sh"
else
    log "STEP 0: shared policy env [${GEMBENCH_CONDA_ENV:-bridgevla_plus_gembench}] (install_gembench.sh --policy-only)"
    bash "${SCRIPT_DIR}/../GemBench/install_gembench.sh" --policy-only
fi

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

# setuptools<81: sapien 3.0.0b1 imports pkg_resources on init, which
# setuptools>=81 dropped.
pip install --upgrade pip==25.0.1
pip install 'setuptools<81' wheel ninja

# STEP 2: python dependencies (torch 2.4.1 wheels on PyPI are cu121)
log "STEP 2: RMBench requirements (sapien, mplib, torch 2.4.1, ...)"
pip install -r "${SCRIPT_DIR}/script/requirements.txt"
# re-assert the pin — a requirement above may have dragged setuptools forward
pip install 'setuptools<81'

# STEP 3: pytorch3d — PREBUILT wheel for this exact torch/CUDA/python combo
# (py310 + cu121 + torch 2.4.1) instead of building from source. The source
# build (git+...@stable) is slow and brittle here, and when it silently fails
# envs/camera/camera.py stubs fps() to exit(), so point-cloud FPS
# downsampling hard-fails at run time.
# NOTE: if you change torch/CUDA/python, swap the py310_cu121_pyt241 channel
# accordingly (see the pytorch3d wheel index on github).
log "STEP 3: pytorch3d (prebuilt wheel)"
if has_pip_pkg pytorch3d; then
    skip "pytorch3d"
else
    # iopath is pytorch3d's runtime dep; --no-index won't fetch it, so install
    # it first from PyPI.
    pip install iopath
    pip install --no-index --no-cache-dir pytorch3d \
        -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt241/download.html
fi

# STEP 4: patch installed site-packages (sapien + mplib)
# Both seds are idempotent — once rewritten, the pattern no longer matches.
log "STEP 4: patching sapien/wrapper/urdf_loader.py (utf-8 file reads)"
SAPIEN_LOCATION="$(pip show sapien | grep 'Location' | awk '{print $2}')/sapien"
URDF_LOADER="${SAPIEN_LOCATION}/wrapper/urdf_loader.py"
[ -f "${URDF_LOADER}" ] || die "sapien not found at ${URDF_LOADER}"
# open(..., "r") -> open(..., "r", encoding="utf-8") for the urdf/srdf reads,
# so non-ASCII asset files load regardless of the system locale.
sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' "${URDF_LOADER}"
grep -q 'encoding="utf-8"' "${URDF_LOADER}" || die "urdf_loader.py patch did not apply (sapien layout changed?)"

log "STEP 4: patching mplib/planner.py (drop the collide guard)"
MPLIB_LOCATION="$(pip show mplib | grep 'Location' | awk '{print $2}')/mplib"
PLANNER="${MPLIB_LOCATION}/planner.py"
[ -f "${PLANNER}" ] || die "mplib not found at ${PLANNER}"
# "or collide" makes plan_screw reject almost every grasp pose whose target
# touches the object — RMBench (like RoboTwin) removes it.
sed -i -E 's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' "${PLANNER}"
grep -qF 'or collide or not within_joint_limit' "${PLANNER}" && die "planner.py patch did not apply (mplib layout changed?)"

# STEP 5: curobo (the repo's single editable install — its CUDA extension
# requires it). Clone is pinned to the v0.7.8 tag.
log "STEP 5: curobo v0.7.8"
if [ ! -d "${SCRIPT_DIR}/envs/curobo" ]; then
    git clone --branch v0.7.8 --depth 1 "${CUROBO_UPSTREAM}" "${SCRIPT_DIR}/envs/curobo"
else
    skip "curobo checkout (envs/curobo exists)"
fi
if has_pip_pkg curobo; then
    skip "curobo (importable)"
else
    log "building curobo's CUDA extension (several minutes) ..."
    (cd "${SCRIPT_DIR}/envs/curobo" && pip install -e . --no-build-isolation)
fi
# curobo's setup.cfg declares warp-lang>=0.9.0, so the editable install above
# pulls the LATEST warp (e.g. 1.14.0). But curobo v0.7.8 calls warp.torch.*,
# which warp>=1.5 removed -> "AttributeError: module 'warp' has no attribute
# 'torch'" -> the planner subprocess dies silently and eval floods with
# BrokenPipeError. Pin warp back down AFTER curobo so this pin wins.
log "STEP 5: pinning warp-lang==1.4.2 (curobo v0.7.8 needs <=1.4.x)"
pip install 'warp-lang==1.4.2'

# STEP 6: final import self-check
if [ "${SKIP_VERIFY:-0}" = "1" ]; then
    warn "STEP 6 skipped (SKIP_VERIFY=1)"
else
    log "STEP 6: import self-check"
    python - <<'PY'
import importlib, sys
import torch  # first, so libc10.so enters the loader path
MODS = ["torch", "sapien", "mplib", "curobo", "warp", "pytorch3d",
        "gymnasium", "trimesh", "open3d", "transforms3d", "h5py", "zarr",
        "imageio", "scipy", "numpy"]
fail = []
for m in MODS:
    try:
        importlib.import_module(m); print(f"  OK   {m}")
    except Exception as e:
        print(f"  FAIL {m}: {e}"); fail.append(m)
import warp
if not warp.__version__.startswith("1.4."):
    fail.append(f"warp-lang=={warp.__version__} (need 1.4.x for curobo v0.7.8)")
print(f"\ntorch={torch.__version__}  cuda={torch.cuda.is_available()}  warp={warp.__version__}")
if fail:
    sys.exit(f"self-check failed: {fail}")
print("ALL OK")
PY
    log "self-check passed"
fi

cat <<EOF

============================================================================
Done. Next steps:

  1) Download the RMBench assets + demos and the per-task checkpoints
     (see README — Download):
       bash scripts/download_datasets.sh --extract rmbench
       bash scripts/download_checkpoints_hf.sh rmbench paligemma
  2) The policy side (training + eval policy server) runs in the shared
     policy env already set up by STEP 0 — nothing more to install
     (GEMBENCH_CONDA_ENV overrides its name; a full GemBench install,
     should you ever add one, shares the same env).
  3) Train / evaluate:
       bash finetune/RMBench_vla/train.sh
       bash finetune/RMBench/policy/BridgeVLA_Plus/eval_double_env.sh

Notes / gotchas:
  * To run the simulator you must set the Vulkan ICD env var
    (machine-specific). On an Orion vGPU cluster:
      export VK_ICD_FILENAMES=/opt/orion/orion_runtime/gpu/vulkan/swf_icd.x86_64.json
  * KEEP warp-lang AT 1.4.2. Installing some baselines' requirements
    (e.g. DexVLA/TinyVLA pin warp-lang==1.8.0) silently breaks curobo
    -> BrokenPipeError during eval. If that happens, re-run:
      pip install 'warp-lang==1.4.2'
============================================================================
EOF
