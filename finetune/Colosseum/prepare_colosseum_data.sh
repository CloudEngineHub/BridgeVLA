#!/usr/bin/env bash
# Resume-safe Colosseum data preparation for BridgeVLA.
#
# The archives are expected to have already been downloaded under:
#   data/bridgevla_data/Colosseum/.downloads/official_train
#   data/bridgevla_data/Colosseum/.downloads/clean_eval
#
# Typical use:
#   bash finetune/Colosseum/prepare_colosseum_data.sh
#
# Train first, then prepare eval separately:
#   PREPARE_EVAL=false bash finetune/Colosseum/prepare_colosseum_data.sh
#   PREPARE_TRAIN=false bash finetune/Colosseum/prepare_colosseum_data.sh
#
# Optional knobs:
#   STAGE_ROOT=<temporary unpack directory, default <ROOT>/.cache/colosseum_stage>
#   CACHE_JOBS=10 COPY_JOBS=20 EVAL_EXTRACT_JOBS=10
#   VERIFY_SHA256=true   # expensive; all archives were already verified once
#
# IMPORTANT — do NOT point STAGE_ROOT at /dev/shm (tmpfs = RAM). Extracting the
# ~80G train + ~190G eval archives into tmpfs exhausts physical RAM (this box
# has no swap) and OOM-kills the whole machine. STAGE_ROOT must be real disk.
# The eval path no longer stages at all: archives are extracted straight to the
# final ${COLOSSEUM_ROOT}/eval (single write, no RAM, no second copy).

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Paths are derived from the repository root (overridable via environment variables).
FINETUNE_DIR="$(dirname "${SCRIPT_DIR}")"
BRIDGEVLA_ROOT="$(dirname "${FINETUNE_DIR}")"
export BRIDGEVLA_DATA_ROOT="${BRIDGEVLA_DATA_ROOT:-${BRIDGEVLA_ROOT}/data/bridgevla_data}"

COLOSSEUM_ROOT="${COLOSSEUM_ROOT:-${BRIDGEVLA_DATA_ROOT}/Colosseum}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-${COLOSSEUM_ROOT}/.downloads}"
TRAIN_ARCHIVE_DIR="${TRAIN_ARCHIVE_DIR:-${DOWNLOAD_ROOT}/official_train}"
EVAL_ARCHIVE_DIR="${EVAL_ARCHIVE_DIR:-${DOWNLOAD_ROOT}/clean_eval}"
# Real-disk staging root (NEVER /dev/shm — see header note; tmpfs staging OOMs
# the box). Only the train path uses this; eval extracts straight to final.
STAGE_ROOT="${STAGE_ROOT:-${BRIDGEVLA_ROOT:-.}/.cache/colosseum_stage}"
CACHE_DIR="${CACHE_DIR:-${COLOSSEUM_ROOT}/_keyframe_cache/size128_v2}"
TRAIN_FINAL_ROOT="${TRAIN_FINAL_ROOT:-${COLOSSEUM_ROOT}}"
PREPARE_TRAIN="${PREPARE_TRAIN:-true}"
PREPARE_EVAL="${PREPARE_EVAL:-true}"
VERIFY_SHA256="${VERIFY_SHA256:-false}"
CACHE_JOBS="${CACHE_JOBS:-10}"
COPY_JOBS="${COPY_JOBS:-20}"
EVAL_EXTRACT_JOBS="${EVAL_EXTRACT_JOBS:-10}"

TRAIN_PART_00="${TRAIN_ARCHIVE_DIR}/train.tar.gz_part00"
TRAIN_PART_01="${TRAIN_ARCHIVE_DIR}/train.tar.gz_part01"
TRAIN_STAGE_ROOT="${STAGE_ROOT}/train"
TRAIN_STAGE_MARKER="${STAGE_ROOT}/.train_extract_complete"
# Eval is extracted straight to its final home (no /dev/shm, no rsync copy).
# Inside each eval archive the tree carries the 7 leading path components of the
# machine the archive was packed on, i.e.
#   <7 path components ending in .../colosseum_data/eval>/<task>/<task>_<var>/...
# so we strip those 7 components (see EVAL_STRIP_COMPONENTS) and land <task>/... in
# ${EVAL_FINAL_ROOT}. Each archive is extracted into a temp dir on the SAME
# filesystem, then atomically mv'd into place so a crash never leaves a half
# task visible to training/eval.
EVAL_FINAL_ROOT="${EVAL_FINAL_ROOT:-${COLOSSEUM_ROOT}/eval}"
EVAL_INCOMING="${EVAL_FINAL_ROOT}/.incoming"
EVAL_MARKER_DIR="${EVAL_FINAL_ROOT}/.markers"
EVAL_STRIP_COMPONENTS="${EVAL_STRIP_COMPONENTS:-7}"
CACHE_LOG_DIR="${COLOSSEUM_ROOT}/prepare_logs/cache"

COLOSSEUM_TASKS=(
  basketball_in_hoop
  close_box
  empty_dishwasher
  get_ice_from_fridge
  hockey
  meat_on_grill
  move_hanger
  wipe_desk
  open_drawer
  slide_block_to_target
  reach_and_drag
  put_money_in_safe
  place_wine_at_rack_location
  insert_onto_square_peg
  turn_oven_on
  straighten_rope
  setup_chess
  scoop_with_spatula
  close_laptop_lid
  stack_cups
)

on_error() {
  local line="$1"
  echo "[prepare] ERROR at line ${line}. Re-run the same command to resume." >&2
}
trap 'on_error "$LINENO"' ERR

is_true() {
  case "${1,,}" in
    true|1|yes|y|on) return 0 ;;
    false|0|no|n|off) return 1 ;;
    *) echo "[prepare] invalid boolean: $1" >&2; return 2 ;;
  esac
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "[prepare] missing required command: $1" >&2
    exit 1
  }
}

check_size() {
  local path="$1"
  local expected="$2"
  if [[ ! -f "${path}" ]]; then
    echo "[prepare] missing archive: ${path}" >&2
    exit 1
  fi
  local actual
  actual="$(stat -c %s "${path}")"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "[prepare] bad size: ${path}: expected ${expected}, got ${actual}" >&2
    exit 1
  fi
}

check_sha256() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(sha256sum "${path}")"
  actual="${actual%% *}"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "[prepare] bad SHA-256: ${path}" >&2
    echo "[prepare] expected ${expected}, got ${actual}" >&2
    exit 1
  fi
}

activate_environment() {
  # conda activation reads CONDA_PREFIX and PS1 without defaults. They are
  # normally unset in non-interactive shells, so temporarily relax nounset.
  set +u
  if [[ -z "${CONDA_BASE:-}" ]]; then
    CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    [[ -z "${CONDA_BASE}" && -d "${HOME}/miniconda3" ]] && CONDA_BASE="${HOME}/miniconda3"
    [[ -z "${CONDA_BASE}" && -d "${HOME}/anaconda3" ]] && CONDA_BASE="${HOME}/anaconda3"
  fi
  source "${CONDA_BASE}/bin/activate" "${COLOSSEUM_CONDA_ENV:-bridgevla_plus_rlbench}"
  set -u

  export COPPELIASIM_ROOT="${FINETUNE_DIR}/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
  export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"
  export QT_QPA_PLATFORM=offscreen
  export PYTHONPATH="${FINETUNE_DIR}/Colosseum:${FINETUNE_DIR}:${FINETUNE_DIR}/Colosseum/robot-colosseum:${FINETUNE_DIR}/bridgevla/libs/point-renderer:${FINETUNE_DIR}/bridgevla/libs/peract_colab:${FINETUNE_DIR}/bridgevla/libs/YARR:${FINETUNE_DIR}/bridgevla/libs/peract:${FINETUNE_DIR}/GemBench:${PYTHONPATH:-}"
}

check_train_archives() {
  check_size "${TRAIN_PART_00}" 40286722294
  check_size "${TRAIN_PART_01}" 40286722295
  if is_true "${VERIFY_SHA256}"; then
    check_sha256 "${TRAIN_PART_00}" bb8ca05fb29918e429a82dce3f94d2d06ac886113345b5d8fc6362fcc0ba6340
    check_sha256 "${TRAIN_PART_01}" 2db8e3e32e60e2d30b72dec75a1bdd63bc6ae6b029dbea033ce9110fc81d050d
  fi
}

check_eval_archives() {
  while IFS=$'\t' read -r filename size sha256; do
    [[ -n "${filename}" ]] || continue
    local path="${EVAL_ARCHIVE_DIR}/${filename}"
    check_size "${path}" "${size}"
    if is_true "${VERIFY_SHA256}"; then
      check_sha256 "${path}" "${sha256}"
    fi
  done <<'EOF'
basketball_in_hoop.tar.gz	8893958316	fc1d87e8bc16dbe24a044a2a765c6a6f4e3cbceb039e73a58f28f635048d4bd2
close_box.tar.gz	10542422905	3b0e687474ff01e6df71d242f082dacdf5662cec11d1059b343945a36efaaa31
close_laptop_lid.tar.gz	6508387363	eb6a46833778152408bca9f721080e34c5fc03e1271a528e3078894fd15e360a
empty_dishwasher.tar.gz	26448924032	81585651bf3f1470998e8b65ea68a16225f8d9d999a87078247bfd5636f45d59
get_ice_from_fridge.tar.gz	12082706699	49a974bdd2fbf229363117de2e9efe8421c8e6f5d0c78b6f7f17d23b926ce0a2
hockey.tar.gz	13006839537	a0da6964fe80f54251cc8d249d31cb76703fc6b1cf19310348a617d40e0350fd
insert_onto_square_peg.tar.gz	9812887592	be45d1b9f9428d6c09cd39e07a95495d2d1adc1873d21abde48eb5c40cb39036
meat_on_grill.tar.gz	6551445751	d93df33409727abac02a73dfb10f68625d24c2c894ba4e10215795235c1858ec
move_hanger.tar.gz	8498795522	cb36295c8d65d2b1def597943016caf76c72f06721bc84e5936cf993b67d9393
open_drawer.tar.gz	4179841745	7f91bb8c0022d666383ff1668af108804da57a56f41173e9b9ba8551ceaf669d
place_wine_at_rack_location.tar.gz	12383946527	d93bf122f2564df31f7b76e691805ad62bfe6e7a627513fd337c621c2e14ce3c
put_money_in_safe.tar.gz	11617659354	c7fc68ddf0760ca87f6d3d4057d25926e06e77a341ce7ae75136b6ddd2774439
reach_and_drag.tar.gz	11985009133	6e08414b2d12bc1dab9d1b6dce943d3b6868c9ad5d5962deb9423034da05ce76
scoop_with_spatula.tar.gz	8101271973	4896a9825deda4da4e940909a7dbaa05fd636a8772ccfd898ab641c3ffcbe02d
setup_chess.tar.gz	7527999056	d065655c34584c871f1d3ed73a2856e531de96a7060951531e20a02a661dab7e
slide_block_to_target.tar.gz	4062063782	3009f0220b81b5f7f2459b7370bef7e0ddbcd6d1b19fcee76d3c3d6bedbe6b40
stack_cups.tar.gz	11970064648	eb2894b7314efb5da6dbc6de60f39788bca1ab3881f8199a284bd59bd0675470
straighten_rope.tar.gz	9483579590	912737804d1250222c8d7183921786c827c0dbaedc2d580187effe41a8be8aeb
turn_oven_on.tar.gz	6359057741	078076bd8f3f12b2def122ffbb9e51348da15f340e15b41fbb2e8aad3e52eaa4
wipe_desk.tar.gz	10727228118	eb5e33373f8a46114c68d3ae4e1c182cbc871cc6742b9038017ee5f6eefd7e99
EOF
}

count_task_files() {
  local root="$1"
  local name="$2"
  if [[ ! -d "${root}" ]]; then
    printf '0\n'
    return 0
  fi
  find "${root}" -type f -name "${name}" 2>/dev/null | wc -l
}

extract_train() {
  local count
  count="$(count_task_files "${TRAIN_STAGE_ROOT}" low_dim_obs.pkl)"
  if [[ -f "${TRAIN_STAGE_MARKER}" && "${count}" == 2000 ]]; then
    echo "[prepare] train stage already complete (2000 episodes)"
    return
  fi

  echo "[prepare] extracting official train archive to ${STAGE_ROOT}"
  mkdir -p "${STAGE_ROOT}"
  cat "${TRAIN_PART_00}" "${TRAIN_PART_01}" | tar -xzf - -C "${STAGE_ROOT}"

  count="$(count_task_files "${TRAIN_STAGE_ROOT}" low_dim_obs.pkl)"
  [[ "${count}" == 2000 ]] || {
    echo "[prepare] train extraction incomplete: ${count}/2000 episodes" >&2
    exit 1
  }
  touch "${TRAIN_STAGE_MARKER}"
}

build_caches() {
  echo "[prepare] building missing keyframe caches with ${CACHE_JOBS} workers"
  mkdir -p "${CACHE_DIR}" "${CACHE_LOG_DIR}"
  export BRIDGEVLA_ROOT FINETUNE_DIR STAGE_ROOT TRAIN_STAGE_ROOT CACHE_DIR CACHE_LOG_DIR

  printf '%s\n' "${COLOSSEUM_TASKS[@]}" | xargs -P "${CACHE_JOBS}" -I{} bash -c '
    task="$1"
    task_cache="${CACHE_DIR}/${task}"
    complete=$(find "${task_cache}" -maxdepth 1 -type f -name "episode*.npz.meta" 2>/dev/null | wc -l)
    if [[ "${complete}" == 100 ]]; then
      echo "[cache] ${task}: already complete"
      exit 0
    fi
    echo "[cache] ${task}: resuming from ${complete}/100"
    python "${FINETUNE_DIR}/Colosseum/utils/colosseum_keyframe_dataset.py" \
      --data_root "${STAGE_ROOT}" \
      --cache_dir "${CACHE_DIR}" \
      --ep_per_task 100 \
      --image_size 128 \
      --memory_enabled \
      --tasks "${task}" \
      > "${CACHE_LOG_DIR}/${task}.log" 2>&1
  ' _ {}

  local task meta_count meta_file cache_file
  for task in "${COLOSSEUM_TASKS[@]}"; do
    meta_count="$(find "${CACHE_DIR}/${task}" -maxdepth 1 -type f -name 'episode*.npz.meta' | wc -l)"
    [[ "${meta_count}" == 100 ]] || {
      echo "[prepare] cache incomplete: ${task}: ${meta_count}/100" >&2
      exit 1
    }
    while IFS= read -r meta_file; do
      cache_file="${meta_file%.meta}"
      [[ -s "${cache_file}" ]] || {
        echo "[prepare] cache payload missing: ${cache_file}" >&2
        exit 1
      }
    done < <(find "${CACHE_DIR}/${task}" -maxdepth 1 -type f -name 'episode*.npz.meta')
  done
  echo "[prepare] keyframe cache complete (20 tasks x 100 episodes)"
}

sync_train_raw() {
  echo "[prepare] syncing train raw data with ${COPY_JOBS} workers"
  mkdir -p "${TRAIN_FINAL_ROOT}"
  export TRAIN_FINAL_ROOT TRAIN_STAGE_ROOT
  printf '%s\n' "${COLOSSEUM_TASKS[@]}" | xargs -P "${COPY_JOBS}" -I{} bash -c '
    task="$1"
    mkdir -p "${TRAIN_FINAL_ROOT}/${task}"
    rsync -a "${TRAIN_STAGE_ROOT}/${task}/" "${TRAIN_FINAL_ROOT}/${task}/"
    echo "[train-sync] ${task}: done"
  ' _ {}

  local task low_dim descriptions
  for task in "${COLOSSEUM_TASKS[@]}"; do
    low_dim="$(count_task_files "${TRAIN_FINAL_ROOT}/${task}" low_dim_obs.pkl)"
    descriptions="$(count_task_files "${TRAIN_FINAL_ROOT}/${task}" variation_descriptions.pkl)"
    [[ "${low_dim}" == 100 && "${descriptions}" == 100 ]] || {
      echo "[prepare] train raw validation failed: ${task}: low_dim=${low_dim}, descriptions=${descriptions}" >&2
      exit 1
    }
  done
  echo "[prepare] train raw data complete (20 tasks x 100 episodes)"
}

extract_eval() {
  echo "[prepare] extracting eval archives directly to ${EVAL_FINAL_ROOT}" \
       "(${EVAL_EXTRACT_JOBS} workers, no /dev/shm, no second copy)"
  mkdir -p "${EVAL_FINAL_ROOT}" "${EVAL_INCOMING}" "${EVAL_MARKER_DIR}"
  export EVAL_FINAL_ROOT EVAL_INCOMING EVAL_MARKER_DIR EVAL_STRIP_COMPONENTS
  find "${EVAL_ARCHIVE_DIR}" -maxdepth 1 -type f -name '*.tar.gz' -print0 \
    | xargs -0 -P "${EVAL_EXTRACT_JOBS}" -n1 bash -c '
        set -Eeuo pipefail
        archive="$1"
        base="$(basename "${archive}" .tar.gz)"
        marker="${EVAL_MARKER_DIR}/${base}.complete"
        if [[ -f "${marker}" ]]; then
          echo "[eval-extract] ${base}: already complete"
          exit 0
        fi
        tmp="${EVAL_INCOMING}/${base}.$$"
        rm -rf "${tmp}"
        mkdir -p "${tmp}"
        # Strip the packer-side prefix (7 components, ends in .../colosseum_data/eval)
        tar -xzf "${archive}" --strip-components="${EVAL_STRIP_COMPONENTS}" -C "${tmp}"
        shopt -s nullglob
        dirs=("${tmp}"/*/)
        if [[ ${#dirs[@]} -ne 1 ]]; then
          echo "[eval-extract] ${base}: expected exactly one task dir, got ${#dirs[@]}" \
               "(wrong EVAL_STRIP_COMPONENTS?)" >&2
          rm -rf "${tmp}"
          exit 1
        fi
        task_src="${dirs[0]%/}"
        task="$(basename "${task_src}")"
        dest="${EVAL_FINAL_ROOT}/${task}"
        rm -rf "${dest}"
        mv "${task_src}" "${dest}"   # same-fs rename: instant, atomic
        rm -rf "${tmp}"
        touch "${marker}"
        echo "[eval-extract] ${base}: done -> ${dest}"
      ' _

  local task
  for task in "${COLOSSEUM_TASKS[@]}"; do
    [[ -d "${EVAL_FINAL_ROOT}/${task}" ]] || {
      echo "[prepare] eval extraction missing task: ${task}" >&2
      exit 1
    }
  done
  rm -rf "${EVAL_INCOMING}"
}

validate_eval() {
  local task variation variation_count=0 episode_count
  for task in "${COLOSSEUM_TASKS[@]}"; do
    [[ -d "${COLOSSEUM_ROOT}/eval/${task}" ]] || {
      echo "[prepare] eval task missing: ${task}" >&2
      exit 1
    }
    while IFS= read -r variation; do
      [[ -n "${variation}" ]] || continue
      [[ -d "${variation}/variation0/episodes" ]] || {
        echo "[prepare] invalid eval variation: ${variation}" >&2
        exit 1
      }
      episode_count="$(find "${variation}/variation0/episodes" -mindepth 1 -maxdepth 1 -type d -name 'episode*' | wc -l)"
      [[ "${episode_count}" == 25 ]] || {
        echo "[prepare] eval variation has ${episode_count}/25 episodes: ${variation}" >&2
        exit 1
      }
      variation_count=$((variation_count + 1))
    done < <(find "${COLOSSEUM_ROOT}/eval/${task}" -mindepth 1 -maxdepth 1 -type d -name "${task}_*" | sort)
  done
  [[ "${variation_count}" -gt 0 ]] || {
    echo "[prepare] no eval variations found" >&2
    exit 1
  }
  echo "[prepare] eval data complete (${variation_count} variations x 25 episodes)"
}

main() {
  require_command python
  require_command rsync
  require_command sha256sum
  require_command tar
  require_command xargs
  activate_environment

  echo "[prepare] Colosseum root: ${COLOSSEUM_ROOT}"
  echo "[prepare] staging root:   ${STAGE_ROOT}"

  if is_true "${PREPARE_TRAIN}"; then
    check_train_archives
    extract_train
    build_caches
    sync_train_raw
  fi

  if is_true "${PREPARE_EVAL}"; then
    check_eval_archives
    extract_eval
    validate_eval
  fi

  echo "[prepare] all requested Colosseum data is ready"
  echo "[prepare] train: ${TRAIN_FINAL_ROOT}/<task>/all_variations/episodes"
  echo "[prepare] cache: ${CACHE_DIR}"
  echo "[prepare] eval:  ${COLOSSEUM_ROOT}/eval"
}

main "$@"
