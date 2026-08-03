#!/usr/bin/env bash
# _download_checkpoints_impl.sh — shared engine behind the two release
# downloaders (do NOT run this file directly, run one of):
#
#   scripts/download_checkpoints_hf.sh   HuggingFace  https://huggingface.co/datasets/LPY/BridgeVLA
#   scripts/download_checkpoints_ms.sh   ModelScope   https://modelscope.cn/models/susetiankong/bridgevla_plus
#
# Both hubs mirror the release with identical content — pick whichever is fast
# for you; every target and option below behaves the same on either.
#
# Files are staged under <data-root>/.dl_stage/ (resumable) and moved into the
# exact layout the train/eval scripts expect. Re-running is safe: finished
# targets are detected and skipped (FORCE=1 re-downloads).
#
# Usage:
#   bash scripts/download_checkpoints_hf.sh [options] <target> [<target> ...]
#   bash scripts/download_checkpoints_ms.sh [options] <target> [<target> ...]
#
# Targets                                              size     destination
#   pretrain        grounding warm start (both gens)   6.6 GiB  data/bridgevla_ckpt/pretrain
#   rlbench         BridgeVLA++ RLBench checkpoint     7.9 GiB  data/bridgevla_ckpt/bridgevla_plus/rlbench
#   colosseum       BridgeVLA++ COLOSSEUM checkpoint   7.9 GiB  data/bridgevla_ckpt/bridgevla_plus/colosseum
#   gembench        BridgeVLA++ GemBench checkpoint    7.8 GiB  data/bridgevla_ckpt/bridgevla_plus/gembench
#   memorybench     BridgeVLA++ memoryBench checkpoint 7.8 GiB  data/bridgevla_ckpt/bridgevla_plus/memorybench
#   rmbench         all 9 RMBench task checkpoints    ~83 GiB  data/bridgevla_ckpt/bridgevla_plus/rmbench/<task>
#   rmbench:<task>  a single RMBench task checkpoint   9.2 GiB  (tasks: run with --list)
#   bridgevla       original BridgeVLA checkpoints      22 GiB  data/bridgevla_ckpt/bridgevla
#                   (for the original codebase only — NOT loadable here)
#   pretrain_data   RoboPoint grounding corpus          18 GiB  data/bridgevla_data/pretrain_data
#   rlbench_cache   RLBench canonical keyframe cache    12 GiB  data/bridgevla_data/RLBench/_keyframe_cache
#                   (pre-built npz+meta; skips the multi-hour local build and
#                    pins the canonical majority-vote keyframes)
#   memorybench_cache  memoryBench keyframe cache      2.3 GiB  data/bridgevla_data/memorybench/data/train/_keyframe_cache
#   rmbench_data    RMBench keyframe training data      30 GiB  data/bridgevla_data/RMBench/data/{keyframe_data,keyframes}
#                   (keyframe-only HDF5 + keyframe metadata; training needs NO
#                    other RMBench demo data — eval additionally needs the
#                    upstream assets, see scripts/download_datasets.sh rmbench)
#   rmbench_data:<task>  a single task's keyframe data  0.5-7.5 GiB
#   paligemma       google/paligemma-3b-pt-224 base     11 GiB  data/bridgevla_ckpt/paligemma-3b-pt-224
#                   (gated on HF: accept license + `hf auth login`; the _ms
#                    variant uses the ungated mirror AI-ModelScope/paligemma-3b-pt-224)
#   clip            OpenAI CLIP RN50.pt                244 MiB  data/bridgevla_ckpt/clip/RN50.pt
#   all             pretrain + 5 benchmark ckpts + paligemma + clip (~120 GiB)
#
# Options:
#   --dest DIR       data root (default <repo>/data)
#   --config-only    only the small cfg files (exp_cfg/mvt_cfg yaml, *.json) —
#                    lets you inspect a model's architecture before an 8 GiB pull
#   --dry-run        print what would be downloaded and exit
#   --list           list targets / RMBench task names and exit
#
# Env:  FORCE=1                    re-download finished targets
#       BRIDGEVLA_HF_REPO / BRIDGEVLA_MS_REPO   override repo ids
#       HF_ENDPOINT                e.g. https://hf-mirror.com
#
# Requires: `hf` or `huggingface-cli` (pip install -U "huggingface_hub[cli]")
#           for the _hf variant; `modelscope` (pip install modelscope) for the
#           _ms variant. Any of this repo's conda envs already has the former.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "${SCRIPT_DIR}")"

HF_REPO="${BRIDGEVLA_HF_REPO:-LPY/BridgeVLA}"
MS_REPO="${BRIDGEVLA_MS_REPO:-susetiankong/bridgevla_plus}"
HF_PALIGEMMA="google/paligemma-3b-pt-224"
MS_PALIGEMMA="AI-ModelScope/paligemma-3b-pt-224"
CLIP_RN50_SHA="afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762"
CLIP_RN50_URL="https://openaipublic.azureedge.net/clip/models/${CLIP_RN50_SHA}/RN50.pt"
RMBENCH_TASKS=(battery_try blocks_ranking_try cover_blocks observe_and_pickup
               press_button put_back_block rearrange_blocks swap_T swap_blocks)

log()  { printf '\033[1;34m[download]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[download]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[download]\033[0m %s\n' "$*" >&2; exit 1; }

SOURCE="${BRIDGEVLA_DL_SOURCE:-}"
case "${SOURCE}" in hf|ms) ;; *)
    die "internal engine — run scripts/download_checkpoints_hf.sh or scripts/download_checkpoints_ms.sh instead"
esac
DEST_ROOT="${ROOT}/data"
CONFIG_ONLY=0
DRY_RUN=0
TARGETS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --source|--source=*)
            die "--source was removed — the hub is picked by the script name: download_checkpoints_hf.sh / download_checkpoints_ms.sh" ;;
        --dest)        DEST_ROOT="$2"; shift 2 ;;
        --dest=*)      DEST_ROOT="${1#*=}"; shift ;;
        --config-only) CONFIG_ONLY=1; shift ;;
        --dry-run)     DRY_RUN=1; shift ;;
        --list)
            echo "targets: pretrain rlbench colosseum gembench memorybench rmbench"
            echo "         rmbench:<task> bridgevla pretrain_data paligemma clip all"
            echo "         rlbench_cache memorybench_cache rmbench_data rmbench_data:<task>"
            echo "rmbench tasks: ${RMBENCH_TASKS[*]}"
            exit 0 ;;
        -h|--help)     awk 'NR==1{next} /^#/{sub(/^# ?/,""); print; next} {exit}' "${BASH_SOURCE[0]}"; exit 0 ;;
        -*)            die "unknown option $1 (see --help)" ;;
        *)             TARGETS+=("$1"); shift ;;
    esac
done
[ "${#TARGETS[@]}" -gt 0 ] || die "no target given — see --help or --list"

CKPT_ROOT="${BRIDGEVLA_CKPT_ROOT:-${DEST_ROOT}/bridgevla_ckpt}"
DATA_ROOT="${BRIDGEVLA_DATA_ROOT:-${DEST_ROOT}/bridgevla_data}"
STAGE_ROOT="${DEST_ROOT}/.dl_stage/${SOURCE}"

# --- tooling ----------------------------------------------------------------
HF_BIN=""
if command -v hf >/dev/null 2>&1; then HF_BIN="hf";
elif command -v huggingface-cli >/dev/null 2>&1; then HF_BIN="huggingface-cli"; fi
MS_BIN=""
command -v modelscope >/dev/null 2>&1 && MS_BIN="modelscope"

if [ "${DRY_RUN}" = "0" ]; then
    if [ "${SOURCE}" = "hf" ] && [ -z "${HF_BIN}" ]; then
        die 'huggingface CLI not found — pip install -U "huggingface_hub[cli]" (or conda activate one of this repo'\''s envs)'
    fi
    if [ "${SOURCE}" = "ms" ] && [ -z "${MS_BIN}" ]; then
        die "modelscope CLI not found — pip install modelscope"
    fi
fi

# hf_xet (rust) ignores proxy env vars and stalls at 0 B/s behind a proxy;
# fall back to the plain HTTP downloader there. Respect an explicit override.
if [ "${SOURCE}" = "hf" ] && [ -n "${https_proxy:-}${HTTPS_PROXY:-}${all_proxy:-}${ALL_PROXY:-}" ]; then
    export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
    log "proxy detected -> HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET}"
fi

# --- engine -----------------------------------------------------------------
# hub_pull <include-pattern> <stage-dir>  — one resumable transfer
hub_pull() {
    local pat="$1" stage="$2"
    mkdir -p "${stage}"
    if [ "${DRY_RUN}" = "1" ]; then
        if [ "${SOURCE}" = "hf" ]; then
            echo "  DRY: ${HF_BIN:-hf} download ${HF_REPO} --repo-type dataset --include '${pat}' --local-dir ${stage}"
        else
            echo "  DRY: modelscope download --model ${MS_REPO} --include '${pat}' --local_dir ${stage}"
        fi
        return 0
    fi
    if [ "${SOURCE}" = "hf" ]; then
        "${HF_BIN}" download "${HF_REPO}" --repo-type dataset \
            --include "${pat}" --local-dir "${stage}"
    else
        "${MS_BIN}" download --model "${MS_REPO}" \
            --include "${pat}" --local_dir "${stage}"
    fi
}

# merge_move <src-dir> <dst-dir> — move every file, preserving relative paths
merge_move() {
    local src="$1" dst="$2" f rel
    [ -d "${src}" ] || return 0
    while IFS= read -r -d '' f; do
        rel="${f#"${src}"/}"
        mkdir -p "${dst}/$(dirname "${rel}")"
        mv -f "${f}" "${dst}/${rel}"
    done < <(find "${src}" -type f -not -path '*/.cache/*' -print0)
    find "${src}" -type d -empty -delete 2>/dev/null || true
}

# fetch_tree <repo-path> <dest-dir> <done-marker-glob> <label>
fetch_tree() {
    local rpath="$1" dest="$2" marker="$3" label="$4" pat
    if [ "${FORCE:-0}" != "1" ] && [ "${CONFIG_ONLY}" = "0" ] \
        && compgen -G "${dest}/${marker}" >/dev/null 2>&1; then
        log "${label}: already present at ${dest} — skipping (FORCE=1 to re-download)"
        return 0
    fi
    pat="${rpath}/*"
    if [ "${CONFIG_ONLY}" = "1" ]; then
        case "${rpath}" in
            checkpoints/pretrain) pat="${rpath}/*.json" ;;
            *)                    pat="${rpath}/*.yaml" ;;
        esac
    fi
    log "${label}: downloading ${pat} (${SOURCE}) ..."
    hub_pull "${pat}" "${STAGE_ROOT}"
    [ "${DRY_RUN}" = "1" ] && return 0
    merge_move "${STAGE_ROOT}/${rpath}" "${dest}"
    log "${label}: ready at ${dest}"
}

fetch_paligemma() {
    local dest="${CKPT_ROOT}/paligemma-3b-pt-224"
    if [ "${FORCE:-0}" != "1" ] && [ -f "${dest}/model-00003-of-00003.safetensors" ]; then
        log "paligemma: already present at ${dest} — skipping"
        return 0
    fi
    mkdir -p "${dest}"
    if [ "${DRY_RUN}" = "1" ]; then
        echo "  DRY: download ${SOURCE} paligemma snapshot -> ${dest}"
        return 0
    fi
    if [ "${SOURCE}" = "hf" ]; then
        log "paligemma: downloading ${HF_PALIGEMMA} (gated — needs accepted license + hf auth login) ..."
        "${HF_BIN}" download "${HF_PALIGEMMA}" --local-dir "${dest}" \
            || die "paligemma download failed. On HF the repo is gated: accept the license at https://huggingface.co/${HF_PALIGEMMA} and run '${HF_BIN} auth login'. Alternatively use scripts/download_checkpoints_ms.sh (ungated mirror)."
    else
        log "paligemma: downloading ${MS_PALIGEMMA} ..."
        "${MS_BIN}" download --model "${MS_PALIGEMMA}" --local_dir "${dest}"
    fi
    log "paligemma: ready at ${dest}"
}

fetch_clip() {
    local dest="${CKPT_ROOT}/clip/RN50.pt" tmp sum
    if [ "${FORCE:-0}" != "1" ] && [ -f "${dest}" ]; then
        log "clip: already present at ${dest} — skipping"
        return 0
    fi
    mkdir -p "$(dirname "${dest}")"
    if [ "${DRY_RUN}" = "1" ]; then
        echo "  DRY: curl ${CLIP_RN50_URL} -> ${dest}"
        return 0
    fi
    tmp="${dest}.part"
    log "clip: downloading RN50.pt ..."
    curl -L --retry 3 -C - -o "${tmp}" "${CLIP_RN50_URL}"
    sum="$(sha256sum "${tmp}" | cut -d' ' -f1)"
    [ "${sum}" = "${CLIP_RN50_SHA}" ] || die "clip: sha256 mismatch (${sum}) — corrupted download, remove ${tmp} and retry"
    mv "${tmp}" "${dest}"
    log "clip: ready at ${dest} (sha256 verified)"
}

fetch_pretrain_data() {
    local dest="${DATA_ROOT}/pretrain_data"
    if [ "${FORCE:-0}" != "1" ] && [ -f "${dest}/detection_data.json" ] \
        && { [ -d "${dest}/coco" ] || [ -f "${dest}/coco.tar.gz" ]; }; then
        log "pretrain_data: already present at ${dest} — skipping"
    else
        log "pretrain_data: downloading (${SOURCE}) ..."
        hub_pull "pretrain_data/*" "${STAGE_ROOT}"
        [ "${DRY_RUN}" = "1" ] && return 0
        merge_move "${STAGE_ROOT}/pretrain_data" "${dest}"
    fi
    if [ -f "${dest}/coco.tar.gz" ] && [ ! -d "${dest}/coco" ]; then
        cat <<EOF
[download] pretrain_data: now extract the COCO images (only needed to re-run
           the grounding pretraining yourself):
             tar -xzf ${dest}/coco.tar.gz -C ${dest}/
EOF
    fi
}

# --- targets ----------------------------------------------------------------
run_target() {
    case "$1" in
        pretrain)
            fetch_tree "checkpoints/pretrain" "${CKPT_ROOT}/pretrain" \
                "model.safetensors.index.json" "pretrain" ;;
        rlbench|colosseum|gembench|memorybench)
            fetch_tree "checkpoints/bridgevla_plus/$1" \
                "${CKPT_ROOT}/bridgevla_plus/$1" "model_*.pth" "$1" ;;
        rmbench)
            local t
            for t in "${RMBENCH_TASKS[@]}"; do run_target "rmbench:${t}"; done ;;
        rmbench:*)
            local task="${1#rmbench:}"
            printf '%s\n' "${RMBENCH_TASKS[@]}" | grep -qx "${task}" \
                || die "unknown RMBench task '${task}' (see --list)"
            fetch_tree "checkpoints/bridgevla_plus/rmbench/${task}" \
                "${CKPT_ROOT}/bridgevla_plus/rmbench/${task}" "model_*.pth" "rmbench/${task}" ;;
        bridgevla)
            local b
            for b in rlbench colosseum gembench; do
                fetch_tree "checkpoints/bridgevla/${b}" \
                    "${CKPT_ROOT}/bridgevla/${b}" "model_*.pth" "bridgevla/${b} (legacy)"
            done ;;
        pretrain_data)  fetch_pretrain_data ;;
        rlbench_cache)
            # Pre-built canonical keyframe cache (npz + .meta). The .meta files
            # pin the majority-vote canonical keyframes; training loads this
            # strictly (allow_build=False).
            fetch_tree "datasets/rlbench/keyframe_cache" \
                "${DATA_ROOT}/RLBench/_keyframe_cache" \
                "size128_v2/*/episode99.npz.meta" "rlbench_cache" ;;
        memorybench_cache)
            # Dest matches memoryBench train.sh's MEMORYBENCH_DATA_FOLDER
            # (data/train), whose parent dir hosts the _keyframe_cache.
            fetch_tree "datasets/memorybench/keyframe_cache" \
                "${DATA_ROOT}/memorybench/data/train/_keyframe_cache" \
                "size128_v3/*/episode99.npz.meta" "memorybench_cache" ;;
        rmbench_data)
            # keyframes/ MUST live next to keyframe_data/ — the dataset derives
            # keyframes_dir as a sibling of data_root (rmbench_dataset.py).
            fetch_tree "datasets/rmbench/keyframes" \
                "${DATA_ROOT}/RMBench/data/keyframes" \
                "press_button.json" "rmbench keyframes-json"
            fetch_tree "datasets/rmbench/keyframe_data" \
                "${DATA_ROOT}/RMBench/data/keyframe_data" \
                "*/keyframe_depth/data/episode0.hdf5" "rmbench keyframe_data" ;;
        rmbench_data:*)
            local dtask="${1#rmbench_data:}"
            printf '%s\n' "${RMBENCH_TASKS[@]}" | grep -qx "${dtask}" \
                || die "unknown RMBench task '${dtask}' (see --list)"
            fetch_tree "datasets/rmbench/keyframes" \
                "${DATA_ROOT}/RMBench/data/keyframes" \
                "press_button.json" "rmbench keyframes-json"
            fetch_tree "datasets/rmbench/keyframe_data/${dtask}" \
                "${DATA_ROOT}/RMBench/data/keyframe_data/${dtask}" \
                "keyframe_depth/data/episode0.hdf5" "rmbench keyframe_data/${dtask}" ;;
        paligemma)      fetch_paligemma ;;
        clip)           fetch_clip ;;
        all)
            local t
            for t in pretrain rlbench colosseum gembench memorybench rmbench paligemma clip; do
                run_target "${t}"
            done ;;
        *) die "unknown target '$1' (see --list)" ;;
    esac
}

log "source=${SOURCE}  dest=${DEST_ROOT}  config_only=${CONFIG_ONLY}  dry_run=${DRY_RUN}"
mkdir -p "${CKPT_ROOT}" "${DATA_ROOT}"
for t in "${TARGETS[@]}"; do
    run_target "${t}"
done
log "all requested targets done."
