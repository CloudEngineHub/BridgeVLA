#!/usr/bin/env bash
# Back-compat shim — the installer moved to finetune/RMBench/install_rmbench.sh,
# which creates and activates the conda env itself (like the other benchmark
# installers):
#   bash finetune/RMBench/install_rmbench.sh
#
# Old flow ("conda activate <env> && bash script/_install.sh") still works:
# an already-activated non-base env is reused instead of creating
# bridgevla_plus_rmbench.
set -euo pipefail
if [ -n "${CONDA_DEFAULT_ENV:-}" ] && [ "${CONDA_DEFAULT_ENV}" != "base" ]; then
    export RMBENCH_CONDA_ENV="${RMBENCH_CONDA_ENV:-${CONDA_DEFAULT_ENV}}"
    echo "[_install.sh] reusing activated conda env: ${RMBENCH_CONDA_ENV}"
fi
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/install_rmbench.sh" "$@"
