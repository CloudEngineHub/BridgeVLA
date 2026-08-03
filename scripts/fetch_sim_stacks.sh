#!/usr/bin/env bash
# fetch_sim_stacks.sh — rebuild the PyRep / RLBench simulation source stacks
#
# RLBench's license does not permit redistribution, so this repository ships
# *patches only*. This script clones the pinned upstream commits and applies
# the bundled patches, reproducing byte-for-byte the source trees that all
# released checkpoints were trained and evaluated with. It is the single
# source of truth used by both install_rlbench.sh and install_gembench.sh.
#
# Idempotent and interruption-safe: an existing tree is never trusted just
# because its directory exists. Every run re-verifies the pinned commit (when
# the tree is a git checkout), the patch state and the pinned task files,
# completes whatever is missing, and fails loudly with a rm-and-rerun
# instruction when a tree cannot be verified. Clones land in a temp dir and
# are moved into place only after the pinned checkout succeeds, so a killed
# run cannot leave a half-initialised tree that a re-run would silently trust.
#
# Two independent stacks coexist (they are NOT interchangeable — a checkpoint
# must be evaluated on the stack it was trained with; see
# finetune/bridgevla/libs/rlbench_patches/README.md):
#
#   shared stack  (GemBench / memoryBench / Colosseum)
#     libs/RLBench            rjgpinel/RLBench      @ ebdc339  + rlbench_patches/
#     libs/PyRep              cshizhe/PyRep         @ 7962b0e  (no patch)
#   peract stack  (RLBench benchmark — matches its checkpoints' training env)
#     libs/RLBench_peract587  buttomnutstoast/RLBench @ 587a6a0 + rlbench_patch_bundle/
#     libs/PyRep_stepjam231   stepjam/PyRep         @ 231a1ac  (no patch)
#
# Usage:
#   bash scripts/fetch_sim_stacks.sh [--shared|--peract|--all]   # default --all
#
# Overrides (point these at your own forks to guard against upstream deletion):
#   RLBENCH_UPSTREAM / PYREP_UPSTREAM                  (shared stack)
#   RLBENCH_PERACT_UPSTREAM / PYREP_STEPJAM_UPSTREAM   (peract stack)
#
# This script only clones + patches. C/CUDA extensions (PyRep cffi,
# point-renderer) are compiled by the install scripts inside the conda env.
# It never touches global git config; per-invocation `git -c` is used instead.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "${SCRIPT_DIR}")"
LIBS_DIR="${ROOT}/finetune/bridgevla/libs"
PATCH_SHARED_DIR="${LIBS_DIR}/rlbench_patches"
PATCH_PERACT_DIR="${ROOT}/finetune/RLBench/rlbench_patch_bundle"

RLBENCH_UPSTREAM="${RLBENCH_UPSTREAM:-https://github.com/rjgpinel/RLBench.git}"
PYREP_UPSTREAM="${PYREP_UPSTREAM:-https://github.com/cshizhe/PyRep.git}"
RLBENCH_PERACT_UPSTREAM="${RLBENCH_PERACT_UPSTREAM:-https://github.com/buttomnutstoast/RLBench.git}"
PYREP_STEPJAM_UPSTREAM="${PYREP_STEPJAM_UPSTREAM:-https://github.com/stepjam/PyRep.git}"

PIN_RLBENCH="ebdc3392c1a11c4cdcc9a440cd61ec345bef42ec"
PIN_PYREP="7962b0e04700315c2b0de87a994dbfe77c915c17"
PIN_RLBENCH_PERACT="587a6a0e6dc8cd36612a208724eb275fe8cb4470"
PIN_PYREP_STEPJAM="231a1ac6b0a179cff53c1d403d379260b9f05f2f"

log()  { printf '\033[1;34m[sim-stacks]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[sim-stacks]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[sim-stacks]\033[0m %s\n' "$*" >&2; exit 1; }

MODE="${1:---all}"
case "${MODE}" in
    --shared|--peract|--all) ;;
    *) die "unknown mode '${MODE}' (use --shared, --peract or --all)" ;;
esac

mkdir -p "${LIBS_DIR}"

# gapply <tree> <git-apply args...> — run `git apply` against <tree> without
# ever resolving the OUTER BridgeVLA repo: GIT_CEILING_DIRECTORIES stops
# upward repo discovery at libs/, so a stack with its own .git uses it, and a
# vendored tree without .git gets git-apply's outside-a-repo (GNU-patch-like,
# cwd-relative) behaviour instead of silently targeting this repo's root.
gapply() {
    local tree="$1"; shift
    GIT_CEILING_DIRECTORIES="${LIBS_DIR}" git -C "${tree}" apply "$@"
}

# clone_pinned <dest> <url> <commit> — clone with retries into <dest>.cloning,
# check out the pin, then atomically move into place: <dest> either does not
# exist or is a complete pinned checkout, never something in between.
# The larger http.postBuffer avoids GnuTLS recv errors behind some proxies;
# applied per-invocation so no global git config is modified.
clone_pinned() {
    local dest="$1" url="$2" pin="$3" tmp="$1.cloning" attempt
    rm -rf "${tmp}"
    for attempt in 1 2 3; do
        if git -c http.postBuffer=524288000 clone "${url}" "${tmp}"; then
            git -C "${tmp}" checkout --quiet "${pin}"
            mv "${tmp}" "${dest}"
            return 0
        fi
        warn "clone of ${url} failed (attempt ${attempt}/3), retrying in 5 s ..."
        rm -rf "${tmp}"
        sleep 5
    done
    die "failed to clone ${url} after 3 attempts — check your network (or set the *_UPSTREAM override to a mirror/fork)"
}

# verify_pin <dir> <pin> <label> — an existing tree with its own .git must sit
# exactly on <pin>; anything else (interrupted checkout, stray manual clone,
# tree copied from another machine) dies loudly instead of being silently
# accepted. NB the .git guard is load-bearing: on a vendored tree without
# .git, rev-parse would walk up and report the OUTER BridgeVLA repo's HEAD —
# such trees carry no commit id, so warn and fall through to content checks.
verify_pin() {
    local dir="$1" pin="$2" label="$3" head
    if [ ! -e "${dir}/.git" ]; then
        warn "${label}: no .git — cannot verify pin ${pin:0:7}; accepting vendored tree (content checks still apply where they exist)"
        return 0
    fi
    head="$(git -C "${dir}" rev-parse HEAD 2>/dev/null || echo '<unreadable>')"
    if [ "${head}" != "${pin}" ]; then
        die "${label} is at ${head:0:12}, expected ${pin:0:12} — remove it and re-run (rm -rf ${dir})"
    fi
}

# ensure_pinned <dest> <url> <pin> <label> — clone when absent, verify when
# present. Either way, on return <dest> exists and (where verifiable) is at
# the pinned commit.
ensure_pinned() {
    local dest="$1" url="$2" pin="$3" label="$4"
    if [ ! -d "${dest}" ]; then
        log "cloning ${label} @ ${pin:0:7} ..."
        clone_pinned "${dest}" "${url}" "${pin}"
    else
        verify_pin "${dest}" "${pin}" "${label}"
        log "${label} already present — skipping clone"
    fi
}

fetch_shared() {
    ensure_pinned "${LIBS_DIR}/PyRep"   "${PYREP_UPSTREAM}"   "${PIN_PYREP}"   "PyRep (cshizhe)"
    ensure_pinned "${LIBS_DIR}/RLBench" "${RLBENCH_UPSTREAM}" "${PIN_RLBENCH}" "RLBench (rjgpinel)"

    # Patch state is re-derived on EVERY run via a three-way check —
    # reverse-apply --check succeeding is proof the patch is FULLY applied
    # (a partial application fails both checks), which a marker grep cannot
    # distinguish. A run interrupted anywhere converges on the next run.
    local d="${LIBS_DIR}/RLBench" p="${PATCH_SHARED_DIR}/rlbench_rjgpinel_ebdc339.patch"
    if gapply "${d}" --reverse --check "${p}" 2>/dev/null; then
        log "RLBench patch already applied"
    elif gapply "${d}" --check "${p}" 2>/dev/null; then
        log "applying rlbench_patches to libs/RLBench ..."
        gapply "${d}" "${p}"
    else
        die "libs/RLBench is neither cleanly unpatched nor fully patched (partial apply or local edits) — remove it and re-run (rm -rf ${d})"
    fi

    # Pinned task files (.py + .ttm) are likewise verified per-file per-run;
    # a missing or drifted copy silently changes eval behaviour, so overwrite
    # on mismatch (byte-identical copies are left untouched).
    local src base dst
    for src in "${PATCH_SHARED_DIR}/tasks/"*; do
        base="$(basename "${src}")"
        case "${base}" in
            *.py)  dst="${d}/rlbench/tasks/${base}" ;;
            *.ttm) dst="${d}/rlbench/task_ttms/${base}" ;;
            *)     continue ;;
        esac
        if [ -f "${dst}" ] && cmp -s "${src}" "${dst}"; then
            continue
        fi
        if [ -f "${dst}" ]; then
            warn "task file ${base} differed from the pinned copy — replacing"
        fi
        cp -f "${src}" "${dst}"
    done
    log "shared RLBench stack ready (pin + patch + task files verified)"
}

fetch_peract() {
    ensure_pinned "${LIBS_DIR}/PyRep_stepjam231"  "${PYREP_STEPJAM_UPSTREAM}"  "${PIN_PYREP_STEPJAM}"  "PyRep_stepjam231 (stepjam)"
    ensure_pinned "${LIBS_DIR}/RLBench_peract587" "${RLBENCH_PERACT_UPSTREAM}" "${PIN_RLBENCH_PERACT}" "RLBench_peract587 (buttomnutstoast)"
    # apply_patch.sh runs on every invocation: SHA-256 of the patched files
    # short-circuits when already applied; otherwise it re-checks the baseline
    # commit and a clean working tree before applying, then verifies SHA-256
    # after — together with ensure_pinned this closes the interrupted-run
    # windows for this stack too.
    bash "${PATCH_PERACT_DIR}/apply_patch.sh" "${LIBS_DIR}/RLBench_peract587"
}

case "${MODE}" in
    --shared) fetch_shared ;;
    --peract) fetch_peract ;;
    --all)    fetch_shared; fetch_peract ;;
esac

log "done. Stacks under ${LIBS_DIR}:"
for d in RLBench PyRep RLBench_peract587 PyRep_stepjam231; do
    if [ -d "${LIBS_DIR}/${d}" ]; then
        # Only report a commit when the directory is its own git checkout
        # (a vendored tree without .git would resolve to the outer repo).
        if [ -e "${LIBS_DIR}/${d}/.git" ]; then
            printf '  %-22s %s\n' "${d}" "$(git -C "${LIBS_DIR}/${d}" rev-parse --short HEAD 2>/dev/null || echo '?')"
        else
            printf '  %-22s %s\n' "${d}" "present (no .git)"
        fi
    fi
done
