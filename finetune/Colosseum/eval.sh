#!/usr/bin/env bash
#
# Colosseum eval launcher (ported from RLBench/eval.sh), multi-GPU parallel scheduler edition.
#
# Colosseum eval is organised by (variation, base_task): one directory per task, with per-variation
# subdirectories <task>_<variation>/variation0/episodes. One task per process is a hard constraint
# (multitask_rlbench_env reads its config from base_cfg_name[-1] at launch, so several tasks in one
# process would pick up the wrong perturbation config), which makes one (task, variation) pair = one
# eval.py process the natural unit of work.
#
# Scheduling: pick GPUs with CUDA_VISIBLE_DEVICES (default 0,1,2,3); at most MAX_PROCS_PER_GPU (3) eval.py
# processes per GPU, each new job going to the least-loaded GPU. All jobs evaluate the same checkpoint under
# one EVAL_LOG_NAME, so results collect into {MODEL_FOLDER}/eval/{model_N}/{EVAL_LOG_NAME}/: one shared
# flock-protected eval_results_*.csv, a result.json / result_detail.txt per <task>_<var>, and per-process
# stdout in joblogs/. At the end the task x variation success-rate matrix is written to summary_matrix.csv.
#
# Usage examples (override model/tasks/devices via the environment):
#   bash eval.sh                                  # everything: 20 tasks x variations 0..14
#   VARIATIONS="0 3 7" bash eval.sh               # only these variations
#   VARIATIONS=0-4 bash eval.sh                   # range syntax
#   VARIATION=0 bash eval.sh                      # legacy single-variation form
#   CUDA_VISIBLE_DEVICES=2,5,6,7 bash eval.sh    # pick physical GPUs yourself
#   MAX_PROCS_PER_GPU=2 bash eval.sh
#   EVAL_TASKS="basketball_in_hoop close_box" VARIATIONS=0-14 bash eval.sh
#   MODEL_FOLDER=/path/to/run MODEL_NAME=model_40.pth bash eval.sh
#   SAVE_VIDEO=true VARIATIONS=0 bash eval.sh
#   SAVE_POINTCLOUD=true ... bash eval.sh          # save each step's merged multi-camera point cloud (.ply)
#   EVAL_LOG_NAME=20260722_sweep ... bash eval.sh  # name the experiment dir; reuse it to append re-runs
set -e

# ---- Paths and environment: all derived from the repository root, no machine-specific config ----
FINETUNE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # <repo>/finetune
BRIDGEVLA_ROOT="$(dirname "${FINETUNE_DIR}")"                      # <repo>
export BRIDGEVLA_DATA_ROOT="${BRIDGEVLA_DATA_ROOT:-${BRIDGEVLA_ROOT}/data/bridgevla_data}"
export BRIDGEVLA_LOG_DIR="${BRIDGEVLA_LOG_DIR:-${BRIDGEVLA_DATA_ROOT}/logs}"
export BRIDGEVLA_CKPT_ROOT="${BRIDGEVLA_CKPT_ROOT:-${BRIDGEVLA_ROOT}/data/bridgevla_ckpt}"
# Released checkpoint root: one subdirectory per bench holding model_<E>.pth + exp_cfg.yaml + mvt_cfg.yaml.
# Your own training runs share that layout, so just override MODEL_FOLDER=<run dir>.
export BRIDGEVLA_RELEASE_CKPT_DIR="${BRIDGEVLA_RELEASE_CKPT_DIR:-${BRIDGEVLA_CKPT_ROOT}/bridgevla_plus}"
export PALIGEMMA_PATH="${PALIGEMMA_PATH:-${BRIDGEVLA_CKPT_ROOT}/paligemma-3b-pt-224}"
export CLIP_CACHE_DIR="${CLIP_CACHE_DIR:-${BRIDGEVLA_CKPT_ROOT}/clip}"
export CLIP_CHECKPOINT_DIR="${CLIP_CHECKPOINT_DIR:-${BRIDGEVLA_CKPT_ROOT}/clip}"
export HF_HOME="${HF_HOME:-${BRIDGEVLA_ROOT}/.cache/hf}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
# Colosseum training run root (where MODEL_FOLDER looks by default).
export COLOSSEUM_LOG_DIR="${COLOSSEUM_LOG_DIR:-${BRIDGEVLA_LOG_DIR}/train_colosseum}"

# conda (CONDA_BASE overridable, auto-detected by default). Colosseum reuses RLBench's python
# environment (same as train.sh; robot-colosseum is vendored and comes in via PYTHONPATH).
if [ -z "${CONDA_BASE:-}" ]; then
    CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    [ -z "${CONDA_BASE}" ] && [ -d "${HOME}/miniconda3" ] && CONDA_BASE="${HOME}/miniconda3"
    [ -z "${CONDA_BASE}" ] && [ -d "${HOME}/anaconda3" ] && CONDA_BASE="${HOME}/anaconda3"
fi
source "${CONDA_BASE}/bin/activate" "${COLOSSEUM_CONDA_ENV:-bridgevla_plus_rlbench}"

# ---- Model & data — override via the environment, no need to edit the script ----
MODEL_FOLDER="${MODEL_FOLDER:-${BRIDGEVLA_RELEASE_CKPT_DIR}/colosseum}"
MODEL_NAME="${MODEL_NAME:-model_290.pth}"
# A released directory usually holds a single weight: when MODEL_NAME does not match, the only model_*.pth
# is picked automatically. With several candidates the value is kept and the sanity check below errors out.
if [ ! -f "${MODEL_FOLDER}/${MODEL_NAME}" ]; then
    _CKPT_CANDS=( "${MODEL_FOLDER}"/model_*.pth )
    if [ "${#_CKPT_CANDS[@]}" -eq 1 ] && [ -f "${_CKPT_CANDS[0]}" ]; then
        MODEL_NAME="$(basename "${_CKPT_CANDS[0]}")"
        echo "[Info] MODEL_NAME auto-resolved to ${MODEL_NAME} (the only weight in ${MODEL_FOLDER})"
    fi
fi

# Colosseum variation set (BridgeVLA eval protocol: spreadsheet indices 0..14 per task). 0=no_variations is
# the clean baseline, 1=all_mixed stacks every perturbation, 2..12 are individual factors,
# 13=rlbench_variations, 14=camera_pose; friction(16)/mass(17) are excluded, matching cal_statics.py.
# VARIATIONS takes a space-separated list mixed with lo-hi ranges ("0 3 7" / "0-14" / "0-4 13"); unset it
# falls back to the legacy single VARIATION, then to the full 0-14.
_var_raw="${VARIATIONS:-${VARIATION:-0-14}}"
if [ "${_var_raw}" = "all" ]; then _var_raw="0-14"; fi
VARIATION_LIST=""
for _tok in ${_var_raw}; do
    case "${_tok}" in
        *-*)
            _lo="${_tok%-*}"; _hi="${_tok#*-}"
            case "${_lo}${_hi}" in *[!0-9]*|"")
                echo "[Error] bad VARIATIONS token: '${_tok}'"; exit 1 ;; esac
            VARIATION_LIST+="$(seq -s' ' "${_lo}" "${_hi}") " ;;
        *)
            case "${_tok}" in *[!0-9]*|"")
                echo "[Error] bad VARIATIONS token: '${_tok}'"; exit 1 ;; esac
            VARIATION_LIST+="${_tok} " ;;
    esac
done
VARIATION_LIST="${VARIATION_LIST% }"
N_VARIATIONS=$(wc -w <<< "${VARIATION_LIST}")

# Results go to {MODEL_FOLDER}/eval/{model_N}/{EVAL_LOG_NAME} (see eval.py); the whole parallel sweep
# shares one directory = "one experiment", with the CSV appended across tasks and variations.
if [ -z "${EVAL_LOG_NAME:-}" ]; then
    if [ "${N_VARIATIONS}" -eq 1 ]; then
        EVAL_LOG_NAME="$(date +%Y%m%d_%H%M%S)_var${VARIATION_LIST}"
    else
        EVAL_LOG_NAME="$(date +%Y%m%d_%H%M%S)_sweep${N_VARIATIONS}vars"
    fi
fi

# Eval data root: one subdirectory per base task, containing <task>_<var>/variation0/episodes. Official data:
# https://huggingface.co/datasets/colosseum/colosseum-challenge (cleaned by clean_colosseum_eval_data.py).
COLOSSEUM_EVAL_DATA_ROOT="${COLOSSEUM_EVAL_DATA_ROOT:-${BRIDGEVLA_DATA_ROOT}/Colosseum/eval}"
DEFAULT_EVAL_TASKS="basketball_in_hoop close_box empty_dishwasher get_ice_from_fridge hockey meat_on_grill move_hanger wipe_desk open_drawer slide_block_to_target reach_and_drag put_money_in_safe place_wine_at_rack_location insert_onto_square_peg turn_oven_on straighten_rope setup_chess scoop_with_spatula close_laptop_lid stack_cups"
EVAL_TASKS="${EVAL_TASKS:-${DEFAULT_EVAL_TASKS}}"

# ---- GPU selection and concurrency limit ----
# An exported CUDA_VISIBLE_DEVICES wins, otherwise the default 0,1,2,3 below is used. Internally the
# scheduler uses logical indices 0..N-1 within the visible set (what eval.py --device expects), and
# EVAL_DEVICES can restrict it further (for debugging).
MAX_PROCS_PER_GPU="${MAX_PROCS_PER_GPU:-3}"   # max concurrent processes per GPU
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
if [ -n "${EVAL_DEVICES:-}" ]; then
    GPU_LIST="${EVAL_DEVICES//,/ }"
elif [ -n "${EVAL_DEVICE:-}" ]; then
    GPU_LIST="${EVAL_DEVICE//,/ }"            # legacy single-GPU form (logical index)
else
    _n=$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")
    GPU_LIST="$(seq -s' ' 0 $(( _n - 1 )))"
fi
LAUNCH_STAGGER="${LAUNCH_STAGGER:-5}"         # seconds between consecutive launches (stagger model loading / simulator startup)
# Cap the CPU threads per process when running many in parallel, so 3*NGPU processes do not fight over cores.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

EVAL_EPISODES="${EVAL_EPISODES:-25}"
EPISODE_LENGTH="${EPISODE_LENGTH:-25}"           # global fallback step_limit (used when a task is absent from the table)
# Per-task step_limit table (by base task name), configs/eval_step_limit.yml by default. Non-null values
# override EPISODE_LENGTH; set it empty to always use EPISODE_LENGTH.
STEP_LIMIT_CONFIG="${STEP_LIMIT_CONFIG:-configs/eval_step_limit.yml}"
# Per-step heatmap visualisation (on by default): lands in {visualize_root_dir}/{task}/episode_{ep}/.
VISUALIZE="${VISUALIZE:-true}"
# Save a rollout mp4 (needs ffmpeg, off by default); independent of VISUALIZE (which is the heatmap).
SAVE_VIDEO="${SAVE_VIDEO:-false}"
# Save the point cloud of every step (merged multi-camera xyz+rgb, .ply, needs open3d).
SAVE_POINTCLOUD="${SAVE_POINTCLOUD:-false}"
# --- Memory ablation switches: must match the memory.temporal_memory / memory.spatial_memory recorded in
# MODEL_FOLDER's mvt_cfg.yaml, or eval.py fails on load. ---
TEMPORAL_MEMORY="${TEMPORAL_MEMORY:-true}"
SPATIAL_MEMORY="${SPATIAL_MEMORY:-true}"

# PALIGEMMA_PATH / HF offline flags are already derived at the top of this script.

# PYTHONPATH (Colosseum first so utils / visualize resolve here; vendored robot-colosseum provides `colosseum`).
export PYTHONPATH="${FINETUNE_DIR}/Colosseum:${FINETUNE_DIR}:${FINETUNE_DIR}/Colosseum/robot-colosseum:${FINETUNE_DIR}/bridgevla/libs/point-renderer:${FINETUNE_DIR}/bridgevla/libs/peract_colab:${FINETUNE_DIR}/bridgevla/libs/YARR:${FINETUNE_DIR}/bridgevla/libs/peract:${FINETUNE_DIR}/GemBench:${FINETUNE_DIR}/bridgevla/libs/PyRep:${FINETUNE_DIR}/bridgevla/libs/RLBench:${PYTHONPATH:-}"

# ---- CoppeliaSim / Qt / display (eval launches the simulator, which needs a display) ----
# All parallel processes share one Xvfb display: mesa software rendering happens inside each client, so
# the X server only moves images around and is not a bottleneck.
export COPPELIASIM_ROOT="${FINETUNE_DIR}/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}"
# Qt xcb needs the system Qt5 GLX plugin (apt-get install -y libqt5gui5).
export QT_PLUGIN_PATH="${COPPELIASIM_ROOT}:/usr/lib/x86_64-linux-gnu/qt5/plugins"

# Xvfb: the virtual X display used for headless rendering (reuse :99 or create a new one).
XVFB_DISPLAY="${XVFB_DISPLAY:-:99}"
if [ -z "${DISPLAY:-}" ] || ! xdpyinfo -display "${DISPLAY}" >/dev/null 2>&1; then
    if xdpyinfo -display "${XVFB_DISPLAY}" >/dev/null 2>&1; then
        echo "[Info] Reusing existing Xvfb on ${XVFB_DISPLAY}"
    else
        echo "[Info] Starting Xvfb on ${XVFB_DISPLAY} ..."
        Xvfb ${XVFB_DISPLAY} -screen 0 1024x768x24 -ac +extension GLX +render -noreset &
        XVFB_PID=$!
        sleep 2
        if ! kill -0 ${XVFB_PID} 2>/dev/null; then
            echo "[Error] Xvfb failed to start"; exit 1
        fi
        echo "[Info] Xvfb started (PID=${XVFB_PID})"
    fi
    export DISPLAY="${XVFB_DISPLAY}"
else
    echo "[Info] Using existing DISPLAY=${DISPLAY}"
fi

cd "${FINETUNE_DIR}/Colosseum"

# Sanity checks
[ -d "${MODEL_FOLDER}" ]               || { echo "[Error] MODEL_FOLDER missing: ${MODEL_FOLDER}"; exit 1; }
[ -f "${MODEL_FOLDER}/${MODEL_NAME}" ] || { echo "[Error] checkpoint missing: ${MODEL_FOLDER}/${MODEL_NAME}"; exit 1; }
for _cfg in exp_cfg.yaml mvt_cfg.yaml; do
    [ -f "${MODEL_FOLDER}/${_cfg}" ] || {
        echo "[Error] ${_cfg} missing in ${MODEL_FOLDER}"
        echo "        the weights must sit next to exp_cfg.yaml + mvt_cfg.yaml (they define the model structure)."
        echo "        your own run directories already have them; to keep a standalone copy, put all three files in one directory:"
        echo "          model_<E>.pth  exp_cfg.yaml  mvt_cfg.yaml"
        exit 1
    }
done
[ -d "${COLOSSEUM_EVAL_DATA_ROOT}" ]   || { echo "[Error] COLOSSEUM_EVAL_DATA_ROOT missing: ${COLOSSEUM_EVAL_DATA_ROOT}"; exit 1; }
[ -d "${PALIGEMMA_PATH}" ]             || { echo "[Error] PALIGEMMA_PATH missing: ${PALIGEMMA_PATH}"; exit 1; }

# Experiment directory (matches eval.py's eval/<model_stem>/<log_name> layout).
MODEL_STEM="$(basename "${MODEL_NAME}" .pth)"
RESULTS_DIR="${MODEL_FOLDER}/eval/${MODEL_STEM}/${EVAL_LOG_NAME}"
JOBLOG_DIR="${RESULTS_DIR}/joblogs"
mkdir -p "${JOBLOG_DIR}"

echo "[Info] MODEL_FOLDER             = ${MODEL_FOLDER}"
echo "[Info] MODEL_NAME               = ${MODEL_NAME}"
echo "[Info] VARIATIONS               = ${VARIATION_LIST} (${N_VARIATIONS} total)"
echo "[Info] EVAL_LOG_NAME            = ${EVAL_LOG_NAME}"
echo "[Info] RESULTS_DIR              = ${RESULTS_DIR}"
echo "[Info] COLOSSEUM_EVAL_DATA_ROOT = ${COLOSSEUM_EVAL_DATA_ROOT}"
echo "[Info] EVAL_TASKS               = ${EVAL_TASKS}"
echo "[Info] CUDA_VISIBLE_DEVICES     = ${CUDA_VISIBLE_DEVICES} (physical GPUs)"
echo "[Info] GPUS(logical)            = ${GPU_LIST}"
echo "[Info] MAX_PROCS_PER_GPU        = ${MAX_PROCS_PER_GPU}"
echo "[Info] EVAL_EPISODES            = ${EVAL_EPISODES}"
echo "[Info] EPISODE_LENGTH           = ${EPISODE_LENGTH}"
echo "[Info] STEP_LIMIT_CONFIG        = ${STEP_LIMIT_CONFIG}"
echo "[Info] VISUALIZE                = ${VISUALIZE}"
echo "[Info] SAVE_VIDEO               = ${SAVE_VIDEO}"
echo "[Info] SAVE_POINTCLOUD          = ${SAVE_POINTCLOUD}"
echo "[Info] TEMPORAL_MEMORY          = ${TEMPORAL_MEMORY}"
echo "[Info] SPATIAL_MEMORY           = ${SPATIAL_MEMORY}"
echo "[Info] COPPELIASIM_ROOT         = ${COPPELIASIM_ROOT}"
echo "[Info] DISPLAY                  = ${DISPLAY}"

EVAL_EXTRA_ARGS=()
case "${VISUALIZE,,}" in
    true|1|yes|y|on)   EVAL_EXTRA_ARGS+=(--visualize) ;;
    false|0|no|n|off)  ;;
    *) echo "[Error] VISUALIZE must be true/false, got: ${VISUALIZE}"; exit 1 ;;
esac
case "${SAVE_VIDEO,,}" in
    true|1|yes|y|on)   EVAL_EXTRA_ARGS+=(--save-video) ;;
    false|0|no|n|off)  ;;
    *) echo "[Error] SAVE_VIDEO must be true/false, got: ${SAVE_VIDEO}"; exit 1 ;;
esac
case "${SAVE_POINTCLOUD,,}" in
    true|1|yes|y|on)   EVAL_EXTRA_ARGS+=(--save-pointcloud) ;;
    false|0|no|n|off)  ;;
    *) echo "[Error] SAVE_POINTCLOUD must be true/false, got: ${SAVE_POINTCLOUD}"; exit 1 ;;
esac
PASSTHRU_ARGS=("$@")

# Some placement-prone variants wedge CoppeliaSim across episodes under the default persistent env, so
# they are detected at the end and re-run with --relaunch-env-each-episode. Two flags prevent a loop:
# RELAUNCH_ACTIVE = this run already carries the flag; _HEAL_JOBS = this run *is* the heal pass.
RELAUNCH_ACTIVE=0
for _a in "$@"; do
    case "${_a}" in --relaunch-env-each-episode) RELAUNCH_ACTIVE=1 ;; esac
done

# ---- Job list: one (base_task, variation) = one eval.py process ----
# Combos without data are skipped (the official data does not cover every variation of every task). The
# agent's memory + 6D settings come from the checkpoint's mvt_cfg.yaml, and the YARR rollout calls
# agent.reset() each episode to clear the MemoryBank. In the heal pass the given "task:var" list is run
# directly, skipping the cross-product.
JOBS=()
SKIPPED_TASKS=()
if [ -n "${_HEAL_JOBS:-}" ]; then
    echo "[Heal] heal pass: only re-running the wedged combos: ${_HEAL_JOBS}"
    for _tok in ${_HEAL_JOBS}; do
        t="${_tok%:*}"; v="${_tok##*:}"
        if [ -e "${COLOSSEUM_EVAL_DATA_ROOT}/${t}/${t}_${v}" ]; then
            JOBS+=("${t} ${v}")
        else
            SKIPPED_TASKS+=("${t}_${v}")
        fi
    done
else
for v in ${VARIATION_LIST}; do
    for t in ${EVAL_TASKS}; do
        if [ -e "${COLOSSEUM_EVAL_DATA_ROOT}/${t}/${t}_${v}" ]; then
            JOBS+=("${t} ${v}")
        else
            SKIPPED_TASKS+=("${t}_${v}")
        fi
    done
done
fi
NJOBS=${#JOBS[@]}
if [ "${#SKIPPED_TASKS[@]}" -gt 0 ]; then
    echo "[Warn] Skipping ${#SKIPPED_TASKS[@]} combos with missing data: ${SKIPPED_TASKS[*]}"
fi
if [ "${NJOBS}" -eq 0 ]; then
    echo "[Error] nothing to evaluate (all (task,variation) combos missing data)"; exit 1
fi

# Scheduler: slot = (GPU, process index), MAX_PROCS_PER_GPU slots per GPU; each new job takes a free slot
# on the GPU currently running the fewest processes.
GPU_ARR=(${GPU_LIST})
NGPU=${#GPU_ARR[@]}
SLOT_GPU=()
for (( _p = 0; _p < MAX_PROCS_PER_GPU; _p++ )); do
    for _g in "${GPU_ARR[@]}"; do SLOT_GPU+=("${_g}"); done
done
NSLOTS=${#SLOT_GPU[@]}
SLOT_PID=()
SLOT_JOB=()
for (( _s = 0; _s < NSLOTS; _s++ )); do SLOT_PID[_s]=""; SLOT_JOB[_s]=""; done

echo "[Info] ${NJOBS} jobs across ${NGPU} GPU(s), ${NSLOTS} parallel slots."
echo "[Info] Per-job stdout: ${JOBLOG_DIR}/<task>_<var>.log  (tail -f to follow a single task)"

pick_slot() {   # print a free slot index, preferring the least loaded GPU; print -1 when none is free
    local -A load=()
    local s g best=-1 bestload=999999
    for (( s = 0; s < NSLOTS; s++ )); do
        if [ -n "${SLOT_PID[s]}" ]; then
            g="${SLOT_GPU[s]}"
            load[$g]=$(( ${load[$g]:-0} + 1 ))
        fi
    done
    for (( s = 0; s < NSLOTS; s++ )); do
        if [ -z "${SLOT_PID[s]}" ]; then
            g="${SLOT_GPU[s]}"
            if [ "${load[$g]:-0}" -lt "${bestload}" ]; then
                bestload=${load[$g]:-0}
                best=${s}
            fi
        fi
    done
    echo "${best}"
}

launch_job() {  # $1=slot $2=task $3=variation
    local slot=$1 task=$2 var=$3
    local gpu="${SLOT_GPU[slot]}"
    local name="${task}_${var}"
    python3 eval.py \
        --model-folder    "${MODEL_FOLDER}" \
        --eval-datafolder "${COLOSSEUM_EVAL_DATA_ROOT}/${task}" \
        --model-name      "${MODEL_NAME}" \
        --tasks           "${name}" \
        --eval-episodes   "${EVAL_EPISODES}" \
        --episode-length  "${EPISODE_LENGTH}" \
        --step-limit-config "${STEP_LIMIT_CONFIG}" \
        --log-name        "${EVAL_LOG_NAME}" \
        --device          "${gpu}" \
        --temporal_memory "${TEMPORAL_MEMORY}" \
        --spatial_memory  "${SPATIAL_MEMORY}" \
        --headless \
        "${EVAL_EXTRA_ARGS[@]}" \
        "${PASSTHRU_ARGS[@]}" \
        > "${JOBLOG_DIR}/${name}.log" 2>&1 &
    SLOT_PID[slot]=$!
    SLOT_JOB[slot]="${name}"
    launched=$(( launched + 1 ))
    echo "[$(date +%H:%M:%S)] [launch ${launched}/${NJOBS}] ${name} -> GPU ${gpu} (pid ${SLOT_PID[slot]})"
}

reap_slots() {  # reclaim finished slots, recording success/failure
    local s pid st
    for (( s = 0; s < NSLOTS; s++ )); do
        pid="${SLOT_PID[s]}"
        if [ -z "${pid}" ]; then continue; fi
        if kill -0 "${pid}" 2>/dev/null; then continue; fi
        st=0; wait "${pid}" 2>/dev/null || st=$?
        done_count=$(( done_count + 1 ))
        if [ "${st}" -eq 0 ]; then
            echo "[$(date +%H:%M:%S)] [done ${done_count}/${NJOBS}] ${SLOT_JOB[s]} OK (GPU ${SLOT_GPU[s]})"
        else
            FAILED_JOBS+=("${SLOT_JOB[s]}")
            echo "[$(date +%H:%M:%S)] [done ${done_count}/${NJOBS}] ${SLOT_JOB[s]} FAILED exit=${st} (GPU ${SLOT_GPU[s]})  log: ${JOBLOG_DIR}/${SLOT_JOB[s]}.log"
        fi
        SLOT_PID[s]=""
        SLOT_JOB[s]=""
    done
}

cleanup_on_signal() {
    trap - INT TERM
    echo ""
    echo "[Warn] Interrupted — killing running eval jobs..."
    local s
    for (( s = 0; s < NSLOTS; s++ )); do
        if [ -n "${SLOT_PID[s]}" ]; then kill "${SLOT_PID[s]}" 2>/dev/null || true; fi
    done
    wait 2>/dev/null
    exit 130
}
trap cleanup_on_signal INT TERM

# Individual job failures are only recorded and reported at the end, so turn off set -e.
set +e
START_TS=$(date +%s)
launched=0
done_count=0
FAILED_JOBS=()
ji=0
while : ; do
    reap_slots
    while [ "${ji}" -lt "${NJOBS}" ]; do
        slot=$(pick_slot)
        if [ "${slot}" -lt 0 ]; then break; fi
        read -r jt jv <<< "${JOBS[ji]}"
        ji=$(( ji + 1 ))
        launch_job "${slot}" "${jt}" "${jv}"
        sleep "${LAUNCH_STAGGER}"
        reap_slots
    done
    if [ "${ji}" -ge "${NJOBS}" ]; then
        busy=0
        for (( _s = 0; _s < NSLOTS; _s++ )); do
            if [ -n "${SLOT_PID[_s]}" ]; then busy=1; break; fi
        done
        if [ "${busy}" -eq 0 ]; then break; fi
    fi
    sleep 5
done
ELAPSED=$(( $(date +%s) - START_TS ))

# ---- Wrap-up: task x variation success-rate matrix + hints for re-running failures ----
echo ""
echo "[Info] All jobs finished in $(( ELAPSED / 3600 ))h$(( ELAPSED % 3600 / 60 ))m: $(( done_count - ${#FAILED_JOBS[@]} ))/${NJOBS} OK, ${#FAILED_JOBS[@]} failed, ${#SKIPPED_TASKS[@]} skipped (missing data)."

# Automatically heal wedged combos.
# Placement-prone variants (especially size perturbation v6 / all_mixed v1) wedge CoppeliaSim across
# episodes under the default persistent env: after one placement failure the score is polluted and
# irreproducible (see eval.py's relaunch comment). Every combo's result.json is scanned for
# TaskEnvironmentError and, together with FAILED_JOBS, re-run once with --relaunch-env-each-episode
# (fresh scene per episode). Results append under the same names and the matrix overwrites polluted rows
# last-wins. Only the main sweep triggers this, so the heal pass cannot loop.
if [ -z "${_HEAL_JOBS:-}" ] && [ "${RELAUNCH_ACTIVE}" -eq 0 ]; then
    HEAL_SET=()
    declare -A _heal_seen=()
    for _js in "${JOBS[@]}"; do
        read -r _ht _hv <<< "${_js}"
        _rj="${RESULTS_DIR}/${_ht}_${_hv}/result.json"
        if [ -f "${_rj}" ] && grep -q "TaskEnvironmentError" "${_rj}"; then
            _k="${_ht}:${_hv}"
            [ -z "${_heal_seen[${_k}]:-}" ] && { HEAL_SET+=("${_k}"); _heal_seen[${_k}]=1; }
        fi
    done
    for _f in "${FAILED_JOBS[@]}"; do
        _fv="${_f##*_}"; _fb="${_f%_*}"; _k="${_fb}:${_fv}"
        [ -z "${_heal_seen[${_k}]:-}" ] && { HEAL_SET+=("${_k}"); _heal_seen[${_k}]=1; }
    done
    if [ "${#HEAL_SET[@]}" -gt 0 ]; then
        echo ""
        echo "[Heal] ${#HEAL_SET[@]} combo(s) wedged/failed, re-running them with --relaunch-env-each-episode into the same experiment: ${HEAL_SET[*]}"
        _HEAL_JOBS="${HEAL_SET[*]}" \
        MODEL_FOLDER="${MODEL_FOLDER}" MODEL_NAME="${MODEL_NAME}" \
        COLOSSEUM_EVAL_DATA_ROOT="${COLOSSEUM_EVAL_DATA_ROOT}" \
        EVAL_LOG_NAME="${EVAL_LOG_NAME}" \
        EVAL_EPISODES="${EVAL_EPISODES}" EPISODE_LENGTH="${EPISODE_LENGTH}" \
        STEP_LIMIT_CONFIG="${STEP_LIMIT_CONFIG}" \
        TEMPORAL_MEMORY="${TEMPORAL_MEMORY}" SPATIAL_MEMORY="${SPATIAL_MEMORY}" \
        MAX_PROCS_PER_GPU="${MAX_PROCS_PER_GPU}" \
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
        VISUALIZE="${VISUALIZE}" SAVE_VIDEO="${SAVE_VIDEO}" SAVE_POINTCLOUD="${SAVE_POINTCLOUD}" \
            bash "${FINETUNE_DIR}/Colosseum/eval.sh" --relaunch-env-each-episode "${PASSTHRU_ARGS[@]}"
        exit $?
    fi
    echo "[Info] no wedged combos, skipping the automatic heal pass."
fi

# Summary matrix + paper-protocol average (var0 is reference only and excluded; see colosseum_matrix.py).
python3 colosseum_matrix.py "${RESULTS_DIR}"

if [ "${#FAILED_JOBS[@]}" -gt 0 ]; then
    echo ""
    echo "[Warn] ${#FAILED_JOBS[@]} job(s) FAILED (logs in ${JOBLOG_DIR}/). Re-run them into the same experiment"
    echo "       (add --relaunch-env-each-episode for placement wedges; the main sweep already healed those, so what is left is usually other errors):"
    for f in "${FAILED_JOBS[@]}"; do
        fv="${f##*_}"; fb="${f%_*}"
        echo "  VARIATIONS=${fv} EVAL_TASKS=${fb} EVAL_LOG_NAME=${EVAL_LOG_NAME} bash eval.sh --relaunch-env-each-episode"
    done
fi
echo "[Info] Done. Results in ${RESULTS_DIR}/"
if [ "${#FAILED_JOBS[@]}" -gt 0 ]; then exit 1; fi
