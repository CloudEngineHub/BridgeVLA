#!/usr/bin/env bash
#
# eval_multi_gpu.sh — GemBench multi-GPU eval scheduler (server + client architecture).
#
# Same lineage as RLBench/eval_multi_gpu_v2.sh (see "Design notes" at the end), but GemBench needs two
# processes per job: start a server, wait for its health check, run the client, tear the server down.
#
# Usage: list the checkpoints in the MODELS array below, then `bash finetune/GemBench/eval_multi_gpu.sh`.
#
# One resident worker per GPU claims jobs off a flock-protected shared queue, so fast GPUs take more work
# and none sit idle. Server port = BASE_PORT + GPU slot, so ports never collide. TEMPORAL_MEMORY /
# SPATIAL_MEMORY are hard-coded below and forced onto every server and client, and each ckpt's
# mvt_cfg.yaml is validated before launch — so MODELS may only hold runs of one memory ablation.
# Logs land in {ckpt dir}/eval/gembench/model_{EPOCH}/seed{SEED}/multi_gpu_launch/{ts}_job{idx}/, while
# the client's result.json keeps run_client.sh's own layout. Every job runs in its own process group, and
# the EXIT/SIGINT trap tears down all workers, servers, clients, xvfb and CoppeliaSim.
#
# Environment overrides: BASE_PORT / SERVER_READY_TIMEOUT_SEC / HEALTH_POLL_INTERVAL_SEC / KILL_GRACE_SEC,
# plus run_client.sh's SPLITS / MAX_EPISODES / RECORD_VIDEO / VISUALIZE, which are inherited through setsid.
# TEMPORAL_MEMORY / SPATIAL_MEMORY are the exception and cannot be overridden from outside.
set -uo pipefail

# Derive the checkpoint root first (the MODELS array expands it at definition time).
FINETUNE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # <repo>/finetune
BRIDGEVLA_ROOT="$(dirname "${FINETUNE_DIR}")"                      # <repo>
export BRIDGEVLA_DATA_ROOT="${BRIDGEVLA_DATA_ROOT:-${BRIDGEVLA_ROOT}/data/bridgevla_data}"
export BRIDGEVLA_LOG_DIR="${BRIDGEVLA_LOG_DIR:-${BRIDGEVLA_DATA_ROOT}/logs}"
export BRIDGEVLA_CKPT_ROOT="${BRIDGEVLA_CKPT_ROOT:-${BRIDGEVLA_ROOT}/data/bridgevla_ckpt}"
# Release checkpoint root (one subdirectory per bench, holding model_<E>.pth + the two cfg yamls).
export BRIDGEVLA_RELEASE_CKPT_DIR="${BRIDGEVLA_RELEASE_CKPT_DIR:-${BRIDGEVLA_CKPT_ROOT}/bridgevla_plus}"

# ---- Checkpoints to evaluate — one per line: "<MODEL_FOLDER> <MODEL_EPOCH> [<SEED>]" ----
#   MODEL_FOLDER : ckpt directory (model_<EPOCH>.pth / exp_cfg.yaml / mvt_cfg.yaml); a released ckpt
#                  directory and one of your own runs have identical layouts.
#   SEED         : optional, defaults to DEFAULT_SEED. GemBench splits its test set into microsteps by seed.
# The client runs all SPLITS x taskvars in one go and does not shard by taskvar, so parallelism is across
# (model, seed) jobs — never list the same pair twice, or both write the same result.json. The paper
# protocol is 4 seeds (200/400/500/600) over one ckpt.
GEMBENCH_CKPT="${GEMBENCH_CKPT:-${BRIDGEVLA_RELEASE_CKPT_DIR}/gembench}"
GEMBENCH_EPOCH="${GEMBENCH_EPOCH:-200}"
MODELS=(
    "${GEMBENCH_CKPT} ${GEMBENCH_EPOCH} 200"
    "${GEMBENCH_CKPT} ${GEMBENCH_EPOCH} 400"
    "${GEMBENCH_CKPT} ${GEMBENCH_EPOCH} 500"
    "${GEMBENCH_CKPT} ${GEMBENCH_EPOCH} 600"
)

# ---- Memory ablation switches: hard-coded here and pushed to every server and client, over both the
# run_server.sh / run_client.sh defaults and the environment. ----
#   temporal_memory: memory 1 = stage-1 (mvt1) temporal memory
#   spatial_memory:  memory 2 = stage-2 (mvt2) spatial anchor
# Must match the mvt_cfg.yaml of every ckpt in MODELS, or the pre-flight check fails.
TEMPORAL_MEMORY="true"
SPATIAL_MEMORY="true"

# ---- Defaults (each overridable by the environment variable of the same name) ----
DEFAULT_SEED="${DEFAULT_SEED:-300}"           # used when a MODELS entry omits SEED
BASE_PORT="${BASE_PORT:-13200}"               # server port on GPU slot s = BASE_PORT + s
SERVER_READY_TIMEOUT_SEC="${SERVER_READY_TIMEOUT_SEC:-900}"  # max wait for a server (big model, slow load)
HEALTH_POLL_INTERVAL_SEC="${HEALTH_POLL_INTERVAL_SEC:-5}"    # health-check poll interval
HEALTH_CURL_TIMEOUT_SEC="${HEALTH_CURL_TIMEOUT_SEC:-5}"      # timeout of a single curl probe
HEALTH_HOST="${HEALTH_HOST:-localhost}"       # host to probe/connect to (the server binds localhost)
KILL_GRACE_SEC="${KILL_GRACE_SEC:-10}"        # grace period between TERM and KILL of a process group
TEARDOWN_SETTLE_SEC="${TEARDOWN_SETTLE_SEC:-3}"  # settle time between tearing down a server and the next job
WORKER_START_STAGGER_SEC="${WORKER_START_STAGGER_SEC:-2}"    # stagger worker starts to ease xvfb-run -a display contention
MICROSTEP_DATA_DIR="${MICROSTEP_DATA_DIR:-}"  # empty = let run_client.sh use the seed-derived default microsteps dir

# One server+client per GPU at a time. MODELS may hold far more entries than there are GPUs — they simply
# queue, so there is no MAX_PER_GPU cap like RLBench v2.

# This script needs python3 to read mvt_cfg.yaml; run_server.sh / run_client.sh activate again (harmless).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEMBENCH_DIR="${SCRIPT_DIR}"
# conda (CONDA_BASE overridable, auto-detected). activate reads undefined variables, which blows up under nounset.
set +u
if [ -z "${CONDA_BASE:-}" ]; then
    CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    [ -z "${CONDA_BASE}" ] && [ -d "${HOME}/miniconda3" ] && CONDA_BASE="${HOME}/miniconda3"
    [ -z "${CONDA_BASE}" ] && [ -d "${HOME}/anaconda3" ] && CONDA_BASE="${HOME}/anaconda3"
fi
source "${CONDA_BASE}/bin/activate" "${GEMBENCH_CONDA_ENV:-bridgevla_plus_gembench}"
set -u

# GPU list: an injected CUDA_VISIBLE_DEVICES is respected; otherwise GPUS_DEFAULT below is used.
GPUS_DEFAULT=(0 1 2 3)
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES}"
else
    GPUS=("${GPUS_DEFAULT[@]}")
fi
NGPU="${#GPUS[@]}"
if (( NGPU == 0 )); then
    echo "[Error] no GPU in list (set CUDA_VISIBLE_DEVICES or edit GPUS_DEFAULT)"; exit 1
fi

# ---- Parse MODELS -> parallel arrays JOB_FOLDER/JOB_EPOCH/JOB_SEED, with pre-flight checks ----
JOB_FOLDER=(); JOB_EPOCH=(); JOB_SEED=()
for entry in "${MODELS[@]}"; do
    read -r _f _e _s <<< "${entry}" || true
    [ -n "${_f:-}" ] || { echo "[Error] empty MODEL_FOLDER in entry: '${entry}'"; exit 1; }
    [ -n "${_e:-}" ] || { echo "[Error] missing MODEL_EPOCH in entry: '${entry}'"; exit 1; }
    _s="${_s:-${DEFAULT_SEED}}"
    JOB_FOLDER+=("${_f}"); JOB_EPOCH+=("${_e}"); JOB_SEED+=("${_s}")
done
NJOB="${#JOB_FOLDER[@]}"
if (( NJOB == 0 )); then
    echo "[Error] MODELS list is empty — add checkpoint entries at the top of this script"; exit 1
fi

# Every ckpt and its companion yamls must exist.
for ((i = 0; i < NJOB; i++)); do
    folder="${JOB_FOLDER[$i]}"; epoch="${JOB_EPOCH[$i]}"
    [ -d "${folder}" ]                          || { echo "[Error] MODEL_FOLDER missing: ${folder}"; exit 1; }
    [ -f "${folder}/model_${epoch}.pth" ]       || { echo "[Error] checkpoint missing: ${folder}/model_${epoch}.pth"; exit 1; }
    for _cfg in exp_cfg.yaml mvt_cfg.yaml; do
        [ -f "${folder}/${_cfg}" ] || {
            echo "[Error] ${_cfg} missing in ${folder}"
            echo "        weights must sit next to exp_cfg.yaml + mvt_cfg.yaml (they define the model structure);"
            echo "        to assemble one yourself: copy model_<E>.pth + exp_cfg.yaml + mvt_cfg.yaml from a run into one directory."
            exit 1
        }
    done
done

# Identical to bridgevla.utils.memory_switches.model_memory_switches: effective switch = memory.enabled
# AND the per-group switch (a missing key counts as True).
memory_switches_for() {
    python3 - "$1" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1])) or {}
mem = cfg.get("memory") or {}
enabled = bool(mem.get("enabled", False))
t = enabled and bool(mem.get("temporal_memory", True))
s = enabled and bool(mem.get("spatial_memory", True))
print(("true" if t else "false"), ("true" if s else "false"))
PY
}

# One-off pre-flight check: the switches hard-coded here must match every ckpt's mvt_cfg.yaml.
for ((i = 0; i < NJOB; i++)); do
    folder="${JOB_FOLDER[$i]}"
    if ! mem_out="$(memory_switches_for "${folder}/mvt_cfg.yaml")"; then
        echo "[Error] cannot parse memory switches from ${folder}/mvt_cfg.yaml"; exit 1
    fi
    read -r ckpt_tmem ckpt_smem <<< "${mem_out}"
    if [ "${ckpt_tmem}" != "${TEMPORAL_MEMORY}" ] || [ "${ckpt_smem}" != "${SPATIAL_MEMORY}" ]; then
        echo "[Error] memory switches mismatch: ${folder}/model_${JOB_EPOCH[$i]}.pth"
        echo "        script: temporal=${TEMPORAL_MEMORY} spatial=${SPATIAL_MEMORY}"
        echo "        ckpt  : temporal=${ckpt_tmem} spatial=${ckpt_smem}  (${folder}/mvt_cfg.yaml)"
        exit 1
    fi
done

# ---- Shared state directory + result table + queue cursor ----
STATE_DIR="$(mktemp -d /tmp/gembench_eval_multi.XXXXXX)"
RESULTS_TSV="${STATE_DIR}/results.tsv"
: > "${RESULTS_TSV}"
CURSOR_FILE="${STATE_DIR}/cursor"
echo 0 > "${CURSOR_FILE}"

echo "[Plan] ${NJOB} job(s) on ${NGPU}-GPU pool [${GPUS[*]}], base_port=${BASE_PORT}, state=${STATE_DIR}"
echo "[Info] readiness endpoint : GET http://${HEALTH_HOST}:<port>/memory_config (HTTP 200 => server up AND model loaded)"
echo "[Info] memory switches (forced): temporal=${TEMPORAL_MEMORY} spatial=${SPATIAL_MEMORY} (matches all ${NJOB} ckpt(s))"
echo "[Info] scheduling : dynamic pool — one worker per GPU, each claims the next queued job when it finishes one."
for ((i = 0; i < NJOB; i++)); do
    echo "  [job ${i}] folder=${JOB_FOLDER[$i]} epoch=${JOB_EPOCH[$i]} seed=${JOB_SEED[$i]}"
done

# Queue: flock-protected atomic cursor. Returns the next unclaimed job index, or -1 when empty.
claim_next_job() {
    (
        flock -x 9
        cur="$(cat "${CURSOR_FILE}")"
        if [ "${cur}" -ge "${NJOB}" ]; then
            printf '%s\n' "-1"
        else
            printf '%s\n' "$(( cur + 1 ))" > "${CURSOR_FILE}"
            printf '%s\n' "${cur}"
        fi
    ) 9>"${CURSOR_FILE}.lock"
}

# Concurrent appends to the result table (many workers write; flock serialises them).
record_result() {
    (
        flock -x 9
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$@" >> "${RESULTS_TSV}"
    ) 9>"${RESULTS_TSV}.lock"
}

# Tear down one process group (setsid means pid == pgid): TERM -> grace -> KILL. xvfb / CoppeliaSim /
# python are all in the group and go down with it.
cleanup_group() {
    local pgid="${1:-}" i
    [ -n "${pgid}" ] || return 0
    if kill -0 -- "-${pgid}" 2>/dev/null; then
        kill -TERM -- "-${pgid}" 2>/dev/null || true
        for ((i = 0; i < KILL_GRACE_SEC; i++)); do
            kill -0 -- "-${pgid}" 2>/dev/null || break
            sleep 1
        done
        kill -KILL -- "-${pgid}" 2>/dev/null || true
    fi
}

# ---- One job: start server -> health check -> start client -> wait -> tear down server. ----
# slot decides the port and pgid filenames; gpu is the physical GPU id.
run_one_job() {
    local slot="$1" gpu="$2" idx="$3"
    local port=$(( BASE_PORT + slot ))
    local folder="${JOB_FOLDER[$idx]}" epoch="${JOB_EPOCH[$idx]}" seed="${JOB_SEED[$idx]}"
    local tag="model_${epoch}/seed${seed} (${folder})"
    local output_root="${folder}/eval/gembench"
    local ts log_dir server_log client_log
    ts="$(date +%Y%m%d_%H%M%S)"
    log_dir="${output_root}/model_${epoch}/seed${seed}/multi_gpu_launch/${ts}_job${idx}"
    mkdir -p "${log_dir}"
    server_log="${log_dir}/server.log"
    client_log="${log_dir}/client.log"

    local server_pgf="${STATE_DIR}/gpu${slot}.server.pgid"
    local client_pgf="${STATE_DIR}/gpu${slot}.client.pgid"

    local t0 t1 dur
    t0="$(date +%s)"
    echo "[gpu ${gpu}] START job ${idx}: ${tag} port=${port}"
    echo "[gpu ${gpu}]   logs: ${log_dir}/{server.log,client.log}"

    # --- Start the server (background, own process group). The pgid file is written as soon as the pid is known.
    setsid env \
        CUDA_VISIBLE_DEVICES="${gpu}" \
        PORT="${port}" \
        TEMPORAL_MEMORY="${TEMPORAL_MEMORY}" \
        SPATIAL_MEMORY="${SPATIAL_MEMORY}" \
        bash "${GEMBENCH_DIR}/run_server.sh" "${epoch}" "${folder}" \
        > "${server_log}" 2>&1 &
    local server_pid=$!
    echo "${server_pid}" > "${server_pgf}"

    # --- Health check: poll /memory_config until HTTP 200, while watching for a server that dies early.
    local ready=0 deadline
    deadline=$(( $(date +%s) + SERVER_READY_TIMEOUT_SEC ))
    while [ "$(date +%s)" -lt "${deadline}" ]; do
        if ! kill -0 "${server_pid}" 2>/dev/null; then
            echo "[gpu ${gpu}] server for job ${idx} EXITED during startup — see ${server_log}"
            break
        fi
        if curl -sf --max-time "${HEALTH_CURL_TIMEOUT_SEC}" -o /dev/null \
                "http://${HEALTH_HOST}:${port}/memory_config"; then
            ready=1; break
        fi
        sleep "${HEALTH_POLL_INTERVAL_SEC}"
    done

    if [ "${ready}" -ne 1 ]; then
        echo "[gpu ${gpu}] FAILED job ${idx}: server not ready within ${SERVER_READY_TIMEOUT_SEC}s"
        rm -f "${server_pgf}"
        cleanup_group "${server_pid}"
        wait "${server_pid}" 2>/dev/null || true
        t1="$(date +%s)"; dur="$(( t1 - t0 ))s"
        record_result "${gpu}" "${idx}" "${tag}" "FAILED(server_not_ready)" "${dur}" "${log_dir}"
        sleep "${TEARDOWN_SETTLE_SEC}"
        return 1
    fi
    echo "[gpu ${gpu}] server ready for job ${idx} (port ${port}); launching client"

    # --- Start the client (background, own process group). IP/PORT point at this GPU's server and the
    #     memory switches match it. arg3 = MICROSTEP_DATA_DIR (empty => the seed-derived default);
    #     arg4 = OUTPUT_ROOT, which must be explicit since run_client.sh's own MODEL_FOLDER is a fallback.
    setsid env \
        CUDA_VISIBLE_DEVICES="${gpu}" \
        PORT="${port}" \
        IP="${HEALTH_HOST}" \
        TEMPORAL_MEMORY="${TEMPORAL_MEMORY}" \
        SPATIAL_MEMORY="${SPATIAL_MEMORY}" \
        bash "${GEMBENCH_DIR}/run_client.sh" "${seed}" "${epoch}" "${MICROSTEP_DATA_DIR}" "${output_root}" \
        > "${client_log}" 2>&1 &
    local client_pid=$!
    echo "${client_pid}" > "${client_pgf}"

    local status=0
    wait "${client_pid}" || status=$?
    rm -f "${client_pgf}"
    cleanup_group "${client_pid}"   # sweep up CoppeliaSim/xvfb remnants left by the client

    # --- Tear down the server, freeing this GPU for the next job.
    rm -f "${server_pgf}"
    cleanup_group "${server_pid}"
    wait "${server_pid}" 2>/dev/null || true
    sleep "${TEARDOWN_SETTLE_SEC}"

    t1="$(date +%s)"; dur="$(( t1 - t0 ))s"
    if [ "${status}" -eq 0 ]; then
        echo "[gpu ${gpu}] DONE  job ${idx}: ${tag} in ${dur}"
        record_result "${gpu}" "${idx}" "${tag}" "OK" "${dur}" "${log_dir}"
        return 0
    else
        echo "[gpu ${gpu}] FAILED job ${idx}: ${tag} (client exit=${status}) in ${dur} — see ${client_log}"
        record_result "${gpu}" "${idx}" "${tag}" "FAILED(client=${status})" "${dur}" "${log_dir}"
        return 1
    fi
}

# One worker per GPU: keep claiming jobs until the queue is empty.
run_worker() {
    local slot="$1" gpu="$2" idx
    while :; do
        idx="$(claim_next_job)"
        [ "${idx}" -lt 0 ] && break
        run_one_job "${slot}" "${gpu}" "${idx}" || true
    done
}

# Signal/exit handling: tear down all workers plus any running server/client process groups. EXIT is
# trapped too, as an idempotent backstop after each job has already cleaned up its own pgid file.
WORKER_PIDS=()
_CLEANED=0
cleanup_all() {
    [ "${_CLEANED}" = 1 ] && return 0
    _CLEANED=1
    local w f pgid
    for w in "${WORKER_PIDS[@]:-}"; do [ -n "${w}" ] && kill -TERM "${w}" 2>/dev/null || true; done
    for f in "${STATE_DIR}"/gpu*.pgid; do
        [ -e "${f}" ] || continue
        read -r pgid < "${f}" 2>/dev/null || continue
        [ -n "${pgid}" ] && kill -TERM -- "-${pgid}" 2>/dev/null || true
    done
    sleep 2
    for f in "${STATE_DIR}"/gpu*.pgid; do
        [ -e "${f}" ] || continue
        read -r pgid < "${f}" 2>/dev/null || continue
        [ -n "${pgid}" ] && kill -KILL -- "-${pgid}" 2>/dev/null || true
    done
    for w in "${WORKER_PIDS[@]:-}"; do [ -n "${w}" ] && kill -KILL "${w}" 2>/dev/null || true; done
}
trap cleanup_all EXIT
trap 'echo "[Warn] caught signal, tearing down all servers/clients ..."; cleanup_all; exit 143' INT TERM

# Start the workers (surplus GPUs stay idle when there are fewer jobs than GPUs), staggered to ease
# xvfb-run -a display contention.
for ((slot = 0; slot < NGPU && slot < NJOB; slot++)); do
    run_worker "${slot}" "${GPUS[$slot]}" &
    WORKER_PIDS+=("$!")
    [ "${WORKER_START_STAGGER_SEC}" -gt 0 ] 2>/dev/null && sleep "${WORKER_START_STAGGER_SEC}"
done

OVERALL=0
for w in "${WORKER_PIDS[@]}"; do
    wait "${w}" || OVERALL=1
done

# ---- Summary (sorted by job index) ----
echo ""
echo "==================== GEMBENCH EVAL SUMMARY ===================="
printf '%-4s %-5s %-26s %-9s %s\n' "JOB" "GPU" "STATUS" "TIME" "MODEL / LOG_DIR"
n_ok=0; n_fail=0
while IFS=$'\t' read -r gpu idx tag status dur log_dir; do
    [ -n "${idx:-}" ] || continue
    printf '%-4s %-5s %-26s %-9s %s\n' "${idx}" "${gpu}" "${status}" "${dur}" "${tag}"
    printf '%-4s %-5s %-26s %-9s %s\n' ""      ""       ""          ""       "-> ${log_dir}"
    case "${status}" in OK) n_ok=$(( n_ok + 1 )) ;; *) n_fail=$(( n_fail + 1 )) ;; esac
done < <(sort -t$'\t' -k2,2n "${RESULTS_TSV}")
echo "--------------------------------------------------------------"
echo "[Info] OK=${n_ok}  FAILED=${n_fail}  TOTAL=${NJOB}"
echo "=============================================================="

if [ "${OVERALL}" -ne 0 ] || [ "${n_fail}" -ne 0 ]; then
    echo "[Error] some jobs FAILED — check the server.log / client.log paths above"
    exit 1
fi
echo "[Info] all ${NJOB} job(s) finished OK"
exit 0

# Design notes / relationship to RLBench/eval_multi_gpu_v2.sh (the canonical script)
# Inherited from v2: the user-editable list + hard-coded memory switches forced downstream, the
# pre-flight per-ckpt mvt_cfg.yaml validation, setsid process groups with TERM -> grace -> KILL cleanup,
# the mktemp'd STATE_DIR / TSV summary, and logs anchored under {ckpt dir}/eval/.
# GemBench-specific: two processes per job (server polled on /memory_config until HTTP 200, then client,
# then teardown), each in its own process group; run_server.sh / run_client.sh are reused as-is and manage
# their own display via `xvfb-run -a`; scheduling is a flock-backed dynamic queue rather than v2's static
# round-robin, so v2's MAX_PER_GPU cap is dropped.
# Limits: all MODELS entries must share one memory ablation; parallelism is per (model, seed) job, since
# the client does not shard by taskvar; server and client must be on the same host.
