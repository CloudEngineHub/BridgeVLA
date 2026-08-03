#!/usr/bin/env bash
# Copy the 3 MemoryBench task definitions into every RLBench install we can
# locate. RLBench resolves task classes by `rlbench.tasks.<name>` and loads
# scene .ttm files from `rlbench/task_ttms/`, so we just drop the files in.
#
# Why "every install":
#   - The repo ships an editable RLBench at finetune/bridgevla/libs/RLBench,
#     but `bridgevla_plus_gembench` actually pip-installed a separate copy
#     into the env's site-packages. Whichever Python imports first wins.
#     Installing to BOTH keeps the project working regardless of how a future
#     env is set up.
#   - For a fresh env or different conda env, override CONDA_ENVS or
#     RLBENCH_DIR explicitly.
#
# Idempotent: cp -n only writes when the destination is missing.
set -euo pipefail

# BRIDGEVLA_ROOT is derived from this script's location (three levels above <repo>/finetune/memoryBench/scripts/).
BRIDGEVLA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# conda (CONDA_BASE overridable, auto-detected by default; only used to enumerate site-packages candidates)
if [ -z "${CONDA_BASE:-}" ]; then
    CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    [ -z "${CONDA_BASE}" ] && [ -d "${HOME}/miniconda3" ] && CONDA_BASE="${HOME}/miniconda3"
    [ -z "${CONDA_BASE}" ] && [ -d "${HOME}/anaconda3" ] && CONDA_BASE="${HOME}/anaconda3"
fi
# Task-definition source. Default to the repo's git-pinned copies
# (finetune/bridgevla/libs/rlbench_patches/tasks) — these are the exact
# versions all released checkpoints were trained/evaluated with. The upstream
# HF dataset (hqfang/memorybench data/files) has evolved since and its task
# files may differ; use SRC_DIR to point there explicitly if you want that.
SRC_DIR="${SRC_DIR:-${BRIDGEVLA_ROOT}/finetune/bridgevla/libs/rlbench_patches/tasks}"

if [ ! -d "${SRC_DIR}" ]; then
    echo "[install_memorybench_tasks] SRC_DIR missing: ${SRC_DIR}"
    echo "  Expected to find put_block_back.py/.ttm etc. there."
    exit 1
fi

# Discover candidate RLBench install dirs.
#   1. Explicit override via RLBENCH_DIR env var.
#   2. Editable repo copy.
#   3. site-packages of every conda env we know about.
candidates=()
if [ -n "${RLBENCH_DIR:-}" ]; then
    candidates+=("${RLBENCH_DIR}")
fi
candidates+=("${BRIDGEVLA_ROOT}/finetune/bridgevla/libs/RLBench/rlbench")
# conda env site-packages candidates: derived from the auto-detected CONDA_BASE, machine-independent.
for env_path in \
    "${CONDA_BASE}/envs/${GEMBENCH_CONDA_ENV:-bridgevla_plus_gembench}" \
    "${CONDA_BASE}" \
    "${CONDA_PREFIX:-}"; do
    [ -z "${env_path}" ] && continue
    for sp in "${env_path}"/lib/python*/site-packages/rlbench; do
        [ -d "${sp}" ] && candidates+=("${sp}")
    done
done

# De-dupe in order.
seen=()
unique=()
for c in "${candidates[@]}"; do
    skip=0
    for s in "${seen[@]:-}"; do
        [ "${s}" = "${c}" ] && skip=1 && break
    done
    if [ "${skip}" = 0 ]; then
        seen+=("${c}")
        unique+=("${c}")
    fi
done

installed_any=0
for rl_dir in "${unique[@]}"; do
    TASK_PY_DIR="${rl_dir}/tasks"
    TASK_TTM_DIR="${rl_dir}/task_ttms"
    if [ ! -d "${TASK_PY_DIR}" ] || [ ! -d "${TASK_TTM_DIR}" ]; then
        echo "[install_memorybench_tasks] skip (not a valid RLBench dir): ${rl_dir}"
        continue
    fi
    echo "[install_memorybench_tasks] target: ${rl_dir}"
    for task in put_block_back rearrange_block reopen_drawer; do
        py_src="${SRC_DIR}/${task}.py"
        ttm_src="${SRC_DIR}/${task}.ttm"
        if [ ! -f "${py_src}" ] || [ ! -f "${ttm_src}" ]; then
            echo "[install_memorybench_tasks] missing source for ${task}: ${py_src} or ${ttm_src}"
            exit 1
        fi
        # Overwrite on content mismatch (a stale/upstream-drifted task file
        # would silently change eval behaviour); no-op when already identical.
        for pair in "${py_src}:${TASK_PY_DIR}/${task}.py" "${ttm_src}:${TASK_TTM_DIR}/${task}.ttm"; do
            src="${pair%%:*}"; dst="${pair##*:}"
            if [ -f "${dst}" ] && cmp -s "${src}" "${dst}"; then
                continue
            elif [ -f "${dst}" ]; then
                echo "[install_memorybench_tasks]   ${dst} differed — replaced with pinned version"
            fi
            cp -f "${src}" "${dst}"
        done
        echo "[install_memorybench_tasks]   installed ${task}"
    done
    installed_any=1
done

if [ "${installed_any}" = 0 ]; then
    echo "[install_memorybench_tasks] WARN: no RLBench install dir was matched."
    echo "  Pass RLBENCH_DIR=/path/to/rlbench to target a specific install."
    exit 1
fi
echo "[install_memorybench_tasks] DONE."
