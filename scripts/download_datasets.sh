#!/usr/bin/env bash
# download_datasets.sh — fetch the third-party benchmark datasets
#
# The five simulation benchmarks use their authors' released data. This script
# downloads each into the exact layout the train/eval scripts expect and (with
# --extract) unpacks it. Everything is resumable; re-running skips finished
# files. Sources are HuggingFace datasets (if huggingface.co is slow, set
# HF_ENDPOINT to a mirror or use your own proxy).
#
# Usage:
#   bash scripts/download_datasets.sh [options] <target> [<target> ...]
#
# Targets                                     download    destination
#   rlbench         PerAct 18-task demos       ~116 GiB   data/bridgevla_data/RLBench/
#                   (hqfang/rlbench-18-tasks; train 100 ep/task + test 25 ep/task)
#   gembench        GemBench train + test      ~162 GiB   data/bridgevla_data/GemBench/
#                   (rjgpinel/GEMBench keysteps_bbox + microsteps; GEMBENCH_FULL=1
#                    mirrors the whole 177 GiB repo)
#   memorybench     MemoryBench 3 tasks         ~22 GiB   data/bridgevla_data/memorybench/data/
#                   (hqfang/memorybench; train 100 ep + test 25 ep + task files)
#   colosseum       COLOSSEUM training set      ~75 GiB   data/bridgevla_data/Colosseum/
#                   (colosseum/colosseum-challenge, train.tar.gz 2 parts)
#   colosseum_eval  COLOSSEUM eval variations  ~186 GiB   data/bridgevla_data/Colosseum/
#                   (per-task tars; subset via COLOSSEUM_EVAL_TASKS="close_box ...")
#   rmbench         RMBench assets + keyframe data ~31 GiB  data/bridgevla_data/RMBench/
#                   (upstream TianxingChen/RMBench assets ~1.3 GiB for eval, plus
#                    the BridgeVLA++ release's training-ready keyframe HDF5 +
#                    keyframe metadata — train/eval need NOTHING else. The raw
#                    37 GiB demo_clean demos are only required to re-extract
#                    keyframes / re-render; opt in with RMBENCH_RAW=1)
#
# rlbench / memorybench with --extract also fetch the released pre-built
# keyframe cache (scripts/download_checkpoints_hf.sh rlbench_cache /
# memorybench_cache): RLBench training REQUIRES it (strict cache mode; building
# locally instead takes hours), memoryBench merely builds it lazily on first
# run if absent. These release-hosted parts default to HuggingFace; set
# BRIDGEVLA_DL_SOURCE=ms to pull them from the ModelScope mirror instead
# (the third-party benchmark data itself exists on HuggingFace only).
#
# Options:
#   --dest DIR     data root (default <repo>/data)
#   --extract      also unpack / prepare after downloading
#   --dry-run      print what would be downloaded and exit
#
# The real-robot dataset is self-collected and not published; RLBench val zips
# are downloaded but only train/test are extracted (val is unused by the
# pipeline). Pretraining data (RoboPoint corpus) is part of the BridgeVLA++
# release instead: scripts/download_checkpoints_hf.sh pretrain_data.
#
# Requires: `hf` or `huggingface-cli` (pip install -U "huggingface_hub[cli]");
# any of this repo's conda envs already has it. RMBench additionally uses the
# python huggingface_hub API from the active env.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "${SCRIPT_DIR}")"

log()  { printf '\033[1;34m[datasets]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[datasets]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[datasets]\033[0m %s\n' "$*" >&2; exit 1; }

DEST_ROOT="${ROOT}/data"
EXTRACT=0
DRY_RUN=0
TARGETS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --dest)    DEST_ROOT="$2"; shift 2 ;;
        --dest=*)  DEST_ROOT="${1#*=}"; shift ;;
        --extract) EXTRACT=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        -*)        die "unknown option $1 (see --help)" ;;
        *)         TARGETS+=("$1"); shift ;;
    esac
done
[ "${#TARGETS[@]}" -gt 0 ] || die "no target given — see --help"

DATA_ROOT="${BRIDGEVLA_DATA_ROOT:-${DEST_ROOT}/bridgevla_data}"

# Release-hosted parts (keyframe caches / RMBench keyframe data) are fetched
# via the checkpoint downloader; BRIDGEVLA_DL_SOURCE=ms switches those pulls
# to the ModelScope mirror. The third-party datasets below are HF-only.
case "${BRIDGEVLA_DL_SOURCE:-hf}" in
    hf|ms) CKPT_DL="${SCRIPT_DIR}/download_checkpoints_${BRIDGEVLA_DL_SOURCE:-hf}.sh" ;;
    *)     die "BRIDGEVLA_DL_SOURCE must be hf or ms" ;;
esac

HF_BIN=""
if command -v hf >/dev/null 2>&1; then HF_BIN="hf";
elif command -v huggingface-cli >/dev/null 2>&1; then HF_BIN="huggingface-cli"; fi
if [ "${DRY_RUN}" = "0" ] && [ -z "${HF_BIN}" ]; then
    die 'huggingface CLI not found — pip install -U "huggingface_hub[cli]" (or conda activate one of this repo'\''s envs)'
fi

# hf_xet (rust) ignores proxy env vars and stalls at 0 B/s behind a proxy
if [ -n "${https_proxy:-}${HTTPS_PROXY:-}${all_proxy:-}${ALL_PROXY:-}" ]; then
    export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
fi

# hf_pull <repo> <local-dir> [include-pattern ...]
hf_pull() {
    local repo="$1" dest="$2"; shift 2
    local inc=()
    while [ $# -gt 0 ]; do inc+=(--include "$1"); shift; done
    if [ "${DRY_RUN}" = "1" ]; then
        echo "  DRY: ${HF_BIN:-hf} download ${repo} --repo-type dataset ${inc[*]:-} --local-dir ${dest}"
        return 0
    fi
    mkdir -p "${dest}"
    "${HF_BIN}" download "${repo}" --repo-type dataset "${inc[@]}" --local-dir "${dest}"
}

# unzip_split <zip-dir> <out-dir>  (zips contain <task>/... at their root)
unzip_split() {
    local zdir="$1" out="$2" z task
    [ -d "${zdir}" ] || { warn "no zips at ${zdir}"; return 0; }
    mkdir -p "${out}"
    for z in "${zdir}"/*.zip; do
        [ -f "${z}" ] || continue
        task="$(basename "${z}" .zip)"
        if [ -d "${out}/${task}/all_variations/episodes" ]; then
            log "extract: ${task} already present under ${out} — skipping"
        else
            log "extract: ${z} -> ${out}/"
            unzip -q -n "${z}" -d "${out}"
        fi
    done
}

dl_rlbench() {
    local dest="${DATA_ROOT}/RLBench"
    log "RLBench: downloading PerAct 18-task demos (train+val+test zips) ..."
    hf_pull "hqfang/rlbench-18-tasks" "${dest}"
    if [ "${DRY_RUN}" = "1" ]; then
        echo "  DRY: bash ${CKPT_DL} rlbench_cache"
        return 0
    fi
    if [ "${EXTRACT}" = "1" ]; then
        # train zips -> RLBench/train/<task>/..., test zips -> RLBench/<task>/...
        # (eval.sh reads test episodes from the RLBench root; see README)
        unzip_split "${dest}/data/train" "${dest}/train"
        unzip_split "${dest}/data/test"  "${dest}"
        # Training loads the canonical keyframe cache STRICTLY (no lazy build);
        # fetch the released pre-built cache into RLBench/_keyframe_cache.
        log "RLBench: fetching released keyframe cache (required for training) ..."
        bash "${CKPT_DL}" --dest "${DEST_ROOT}" rlbench_cache
    else
        log "RLBench: re-run with --extract (or unzip data/{train,test}/*.zip yourself) before training"
        log "RLBench: training also needs the keyframe cache: bash scripts/download_checkpoints_hf.sh rlbench_cache"
    fi
}

dl_gembench() {
    local dest="${DATA_ROOT}/GemBench"
    # Only what this pipeline consumes: train_dataset/keysteps_bbox (the
    # trainer reads keysteps_bbox/seed0) and test_dataset/microsteps (eval
    # client). The keysteps_bbox_pcd / motion_keysteps_bbox_pcd variants are
    # 3dlotus-specific — set GEMBENCH_FULL=1 to mirror the whole repo anyway.
    log "GemBench: downloading train keysteps + test microsteps ..."
    if [ "${GEMBENCH_FULL:-0}" = "1" ]; then
        hf_pull "rjgpinel/GEMBench" "${dest}"
    else
        hf_pull "rjgpinel/GEMBench" "${dest}" \
            "train_dataset/keysteps_bbox/*" "test_dataset/*"
    fi
    [ "${DRY_RUN}" = "1" ] && return 0
    # taskvars_*.json / taskvars_instructions_new.json ship with this repo
    # (finetune/GemBench/assets) — place the copies train.sh expects.
    local j
    for j in "${ROOT}/finetune/GemBench/assets/"*.json; do
        [ -f "${dest}/train_dataset/$(basename "${j}")" ] || {
            mkdir -p "${dest}/train_dataset"
            cp "${j}" "${dest}/train_dataset/"
        }
    done
    if [ "${EXTRACT}" = "1" ] && [ -f "${dest}/test_dataset/microsteps.tar.gz" ] \
        && [ ! -d "${dest}/test_dataset/microsteps" ]; then
        log "GemBench: extracting test_dataset/microsteps.tar.gz (needed by eval client) ..."
        tar -xzf "${dest}/test_dataset/microsteps.tar.gz" -C "${dest}/test_dataset/"
    fi
}

dl_memorybench() {
    local dest="${DATA_ROOT}/memorybench"
    log "memoryBench: downloading 3-task dataset ..."
    hf_pull "hqfang/memorybench" "${dest}"
    if [ "${DRY_RUN}" = "1" ]; then
        echo "  DRY: bash ${CKPT_DL} memorybench_cache"
        return 0
    fi
    if [ "${EXTRACT}" = "1" ]; then
        DATA_ROOT="${dest}/data" bash "${ROOT}/finetune/memoryBench/scripts/unzip_memorybench_data.sh"
        # Optional but saves the first-run lazy build (~hour): released cache.
        log "memoryBench: fetching released keyframe cache ..."
        bash "${CKPT_DL}" --dest "${DEST_ROOT}" memorybench_cache
    else
        log "memoryBench: re-run with --extract (or run finetune/memoryBench/scripts/unzip_memorybench_data.sh)"
    fi
    # The task .py/.ttm under data/files/ have drifted upstream; evaluation
    # always uses the repo-pinned copies in bridgevla/libs/rlbench_patches/tasks
    # (install_memorybench_tasks.sh enforces this automatically).
    warn "memoryBench: task definitions are pinned in-repo; upstream data/files may differ — that is expected"
}

dl_colosseum() {
    local droot="${DATA_ROOT}/Colosseum/.downloads/hf"
    log "COLOSSEUM: downloading training archives (2 x 37.5 GiB) ..."
    hf_pull "colosseum/colosseum-challenge" "${droot}" "training_dataset/*"
    [ "${DRY_RUN}" = "1" ] && return 0
    if [ "${EXTRACT}" = "1" ]; then
        log "COLOSSEUM: preparing training data (prepare_colosseum_data.sh) ..."
        TRAIN_ARCHIVE_DIR="${droot}/training_dataset" PREPARE_EVAL=false \
            bash "${ROOT}/finetune/Colosseum/prepare_colosseum_data.sh"
    else
        log "COLOSSEUM: next: TRAIN_ARCHIVE_DIR=${droot}/training_dataset PREPARE_EVAL=false bash finetune/Colosseum/prepare_colosseum_data.sh"
    fi
}

dl_colosseum_eval() {
    local droot="${DATA_ROOT}/Colosseum/.downloads/hf"
    local tasks="${COLOSSEUM_EVAL_TASKS:-}"
    if [ -n "${tasks}" ]; then
        local pats=() t
        for t in ${tasks}; do pats+=("${t}.tar.gz"); done
        log "COLOSSEUM eval: downloading ${#pats[@]} task archive(s): ${tasks}"
        hf_pull "colosseum/colosseum-challenge" "${droot}" "${pats[@]}"
    else
        log "COLOSSEUM eval: downloading ALL per-task eval archives (~200 GiB; set COLOSSEUM_EVAL_TASKS=\"close_box ...\" for a subset)"
        hf_pull "colosseum/colosseum-challenge" "${droot}" "*.tar.gz"
    fi
    [ "${DRY_RUN}" = "1" ] && return 0
    if [ "${EXTRACT}" = "1" ]; then
        log "COLOSSEUM eval: extracting via prepare_colosseum_data.sh ..."
        EVAL_ARCHIVE_DIR="${droot}" PREPARE_TRAIN=false \
            bash "${ROOT}/finetune/Colosseum/prepare_colosseum_data.sh"
    else
        log "COLOSSEUM eval: next: EVAL_ARCHIVE_DIR=${droot} PREPARE_TRAIN=false bash finetune/Colosseum/prepare_colosseum_data.sh"
    fi
}

dl_rmbench() {
    # Train/eval need only: upstream assets (eval sim) + the released
    # keyframe data (training HDF5 + keyframe metadata). The raw demo_clean
    # demos (37 GiB) are needed ONLY to re-extract keyframes / re-render —
    # opt in with RMBENCH_RAW=1.
    if [ "${DRY_RUN}" = "1" ]; then
        echo "  DRY: bash finetune/RMBench/script/_download_assets.sh"
        echo "  DRY: bash ${CKPT_DL} rmbench_data"
        [ "${RMBENCH_RAW:-0}" = "1" ] && echo "  DRY: bash finetune/RMBench/script/_download_data.sh"
        return 0
    fi
    bash "${ROOT}/finetune/RMBench/script/_download_assets.sh"
    bash "${CKPT_DL}" --dest "${DEST_ROOT}" rmbench_data
    if [ "${RMBENCH_RAW:-0}" = "1" ]; then
        log "RMBench: RMBENCH_RAW=1 — also downloading raw demo_clean demos (37 GiB) ..."
        bash "${ROOT}/finetune/RMBench/script/_download_data.sh"
    else
        log "RMBench: skipped raw demo_clean demos (only needed to re-extract keyframes; RMBENCH_RAW=1 to fetch)"
    fi
}

log "dest=${DATA_ROOT}  extract=${EXTRACT}  dry_run=${DRY_RUN}"
for t in "${TARGETS[@]}"; do
    case "${t}" in
        rlbench)        dl_rlbench ;;
        gembench)       dl_gembench ;;
        memorybench)    dl_memorybench ;;
        colosseum)      dl_colosseum ;;
        colosseum_eval) dl_colosseum_eval ;;
        rmbench)        dl_rmbench ;;
        all)            dl_rlbench; dl_gembench; dl_memorybench; dl_colosseum; dl_colosseum_eval; dl_rmbench ;;
        *)              die "unknown target '${t}' (see --help)" ;;
    esac
done
log "all requested targets done."
