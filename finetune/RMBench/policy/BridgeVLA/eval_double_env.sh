#!/bin/bash
# Client-Server eval for the BridgeVLA dual-arm policy.
#   * server: bridgevla (gembench) conda env — holds PaliGemma + the agent.
#   * client: RMBench conda env — runs the SAPIEN sim.
#
# Usage (recommended):
#   bash eval_double_env.sh
#   -- which task / which ckpt / which GPU are all set in eval_config.yml next to this script
#     (task_name / task_config / ckpt_setting / seed / gpu_id); no command-line arguments needed.
#
# Usage (optional, a one-off override of the yml):
#   bash eval_double_env.sh <task_name> <task_config> <ckpt_dir_or_pth> <seed> <gpu_id>
#   -- positional arguments take priority; omitted positions fall back to eval_config.yml.
#
# <ckpt_dir_or_pth>: a training run dir (containing model_*.pth + exp_cfg.yaml
#   + mvt_cfg.yaml) or a direct .pth path. Passed through as ckpt_setting.
#
# Eval outputs (result.txt / videos / log) are written UNDER the training run
# dir, mirroring memoryBench's MODEL_FOLDER/eval/<benchmark>/ convention:
#   <run_dir>/eval/rmbench/<task>/<task_config>/<model_name>/<timestamp>_<flags>/
# (model_name is the name of the ckpt under evaluation, e.g. model_105; timestamp looks like 06_11_11_50_13,
#  flags look like force_center_0_sparsePC_force_center_0, always with an explicit 0/1)
# (when ckpt_setting is a .pth, dirname(.pth) is used as the run dir; when it is
#  neither a dir nor a file, it falls back to the old relative eval_result/.)
#
# ---- Visualisation / step-count control panel ----
# The knobs you tune often live in eval_config.yml next to this script (edit that file; no need to
# dig into task_config/demo_clean.yml or task_config/_eval_step_limit.yml):
#   1. eval_video_log   — enable visualisation recording (true/false). The single master switch for
#                         third_view rendering: turning it off skips the whole third_view chain, a free speed-up that saves CPU.
#   2. eval_video_mode  — keyframe visualisation (keyframe) or continuous-frame visualisation (smooth)
#   3. step_limit       — max execution steps per task (a task name -> step count table)
#   4. eval_overlay_viz — enable overlay heatmap visualisation (per-arm, per-stage predicted heatmaps
#                         overlaid on the three views, assembled into a grid at the end of each episode;
#                         matches training). It comes from the network forward, never touches third_view,
#                         and is fully decoupled from recording. When enabled it also writes
#                         action_pose.txt / plan_status.txt (Curobo planning diagnostics).
# Priority: the environment variables below > eval_config.yml > the task_config defaults. For a one-off override set:
#   RMBENCH_EVAL_VIDEO=1            enable recording (off by default, see eval_config.yml)
#   RMBENCH_EVAL_VIDEO_MODE=smooth  continuous frames (or keyframe)
#   RMBENCH_EVAL_TEST_NUM=5         run only 5 seeds per task (quick check)
#   RMBENCH_EVAL_START_SEED=100064  start from a given sim seed (to reproduce a single episode)
#   RMBENCH_EVAL_OVERLAY_VIZ=0      disable overlay heatmap visualisation (on by default; lands in <run_dir>/eval/.../viz/)
#   stage2_force_center=1  force the stage-2 translation to the zoom centre (the stage-1 wpt), skipping
#                          coordinate inference from the mvt2 heatmap; mvt2 still runs a full forward and
#                          rotation/gripper features still come from stage 2 (=0 disables it, the default).
#   stage2_sparsePC_force_center=1  fall back to the stage-1 translation (the zoom centre) when the zoom has too few points
#                                   (=0 disables that fallback and always uses the mvt2 translation).

policy_name=BridgeVLA

# Uniform bool environment-variable parsing: 1/true/yes/on => 1, anything else => 0
to_bool01() {
  case "${1,,}" in
    1|true|yes|on) echo 1 ;;
    *) echo 0 ;;
  esac
}

# Force the stage-2 translation to the zoom centre (the stage-1 waypoint): mvt2 still runs to completion,
# but the translation no longer comes from the mvt2 heatmap argmax — it always uses wpt_local1 (the zoom
# cube centre). Rotation/gripper still come from mvt2.
stage2_force_center="${stage2_force_center:-0}"
export stage2_force_center="${stage2_force_center}"
stage2_sparsePC_force_center="${stage2_sparsePC_force_center:-0}"
export stage2_sparsePC_force_center="${stage2_sparsePC_force_center}"
stage2_force_center_on="$(to_bool01 "${stage2_force_center}")"
stage2_sparsePC_force_center_on="$(to_bool01 "${stage2_sparsePC_force_center}")"

# Paths and environment: all derived from the repository root, no machine-specific config;
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FINETUNE_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"               # <repo>/finetune
BRIDGEVLA_ROOT="$(dirname "${FINETUNE_DIR}")"                      # <repo>
export BRIDGEVLA_DATA_ROOT="${BRIDGEVLA_DATA_ROOT:-${BRIDGEVLA_ROOT}/data/bridgevla_data}"
export BRIDGEVLA_LOG_DIR="${BRIDGEVLA_LOG_DIR:-${BRIDGEVLA_DATA_ROOT}/logs}"
export BRIDGEVLA_CKPT_ROOT="${BRIDGEVLA_CKPT_ROOT:-${BRIDGEVLA_ROOT}/data/bridgevla_ckpt}"
export PALIGEMMA_PATH="${PALIGEMMA_PATH:-${BRIDGEVLA_CKPT_ROOT}/paligemma-3b-pt-224}"
export HF_HOME="${HF_HOME:-${BRIDGEVLA_ROOT}/.cache/hf}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
# Real location of the RMBench data/assets (envs/_GLOBAL_CONFIGS.py reads these two variables first).
export RMBENCH_DATA_PATH="${RMBENCH_DATA_PATH:-${BRIDGEVLA_DATA_ROOT}/RMBench/data}"
export RMBENCH_ASSETS_PATH="${RMBENCH_ASSETS_PATH:-${BRIDGEVLA_DATA_ROOT}/RMBench/assets}"

# conda (CONDA_BASE overridable, auto-detected by default)
if [ -z "${CONDA_BASE:-}" ]; then
    CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    [ -z "${CONDA_BASE}" ] && [ -d "${HOME}/miniconda3" ] && CONDA_BASE="${HOME}/miniconda3"
    [ -z "${CONDA_BASE}" ] && [ -d "${HOME}/anaconda3" ] && CONDA_BASE="${HOME}/anaconda3"
fi
GEMBENCH_CONDA_ENV="${GEMBENCH_CONDA_ENV:-bridgevla_plus_gembench}"
RMBENCH_CONDA_ENV="${RMBENCH_CONDA_ENV:-bridgevla_plus_rmbench}"

# == Run parameters: read from eval_config.yml by default, overridden by positional arguments if given ==
EVAL_CONFIG="${SCRIPT_DIR}/eval_config.yml"
# A snapshot of eval_config is taken at startup and every task/episode in this run reads only the
# snapshot; editing eval_config.yml on disk mid-run does not affect an already-started evaluation (across tasks either).
EVAL_CONFIG_SNAPSHOT="$(mktemp "${TMPDIR:-/tmp}/rmbench_eval_config.XXXXXX.yml")"
cp "${EVAL_CONFIG}" "${EVAL_CONFIG_SNAPSHOT}"
export RMBENCH_EVAL_CONFIG_SNAPSHOT="${EVAL_CONFIG_SNAPSHOT}"
SERVER_PID=""
# The server is started with set -m in its own process group (see below), so signals go to the whole
# group and python (plus any children) is killed rather than just the outer subshell, which would
# orphan python and leak VRAM. After TERM we wait at most 10s, then escalate to SIGKILL.
cleanup_eval() {
  if [ -n "${SERVER_PID}" ] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo -e "\033[31m[cleanup] Killing server process group (PGID=${SERVER_PID})\033[0m"
    kill -TERM -- -"${SERVER_PID}" 2>/dev/null || kill -TERM "${SERVER_PID}" 2>/dev/null
    for _ in $(seq 1 10); do
      kill -0 "${SERVER_PID}" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo -e "\033[31m[cleanup] Server still alive after 10s; sending SIGKILL\033[0m"
      kill -KILL -- -"${SERVER_PID}" 2>/dev/null || kill -KILL "${SERVER_PID}" 2>/dev/null
    fi
  fi
  rm -f "${EVAL_CONFIG_SNAPSHOT}"
}
trap cleanup_eval EXIT
# Ctrl-C / kill of this script also goes through the EXIT trap to finish the cleanup.
trap 'exit 130' INT
trap 'exit 143' TERM

# Read top-level yml scalars in pure bash (conda is not active yet, so PyYAML may be missing). Only
# matches unindented "key: value" lines at the start of a line (indented keys inside the step_limit
# table are never hit), strips inline # comments and quotes; a null/empty value returns the empty string.
yaml_get() {
  local key="$1" line val
  line=$(grep -E "^${key}:" "${EVAL_CONFIG_SNAPSHOT}" 2>/dev/null | head -1)
  val="${line#*:}"          # drop "key:"
  val="${val%%#*}"          # drop the inline # comment
  val="$(echo "${val}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"  # trim
  val="${val%\"}"; val="${val#\"}"   # strip double quotes
  val="${val%\'}"; val="${val#\'}"   # strip single quotes
  [ "${val}" = "null" ] && val=""
  printf '%s' "${val}"
}

task_name=${1:-$(yaml_get task_name)}
task_config=${2:-$(yaml_get task_config)}
ckpt_setting=${3:-$(yaml_get ckpt_setting)}
seed=${4:-$(yaml_get seed)}
gpu_id=${5:-$(yaml_get gpu_id)}

# Validate the required fields; a missing one errors out clearly (instead of pushing an empty value into --overrides and making the server raise StopIteration).
missing=()
[ -z "${task_name}" ]   && missing+=("task_name")
[ -z "${task_config}" ] && missing+=("task_config")
[ -z "${ckpt_setting}" ]&& missing+=("ckpt_setting")
[ -z "${seed}" ]        && missing+=("seed")
[ -z "${gpu_id}" ]      && missing+=("gpu_id")
if [ ${#missing[@]} -gt 0 ]; then
  echo -e "\033[31m[eval] missing run parameters: ${missing[*]}\033[0m"
  echo -e "\033[31m       set them in ${EVAL_CONFIG}, or pass them as positional arguments:\033[0m"
  echo -e "\033[31m       bash eval_double_env.sh <task_name> <task_config> <ckpt_dir_or_pth> <seed> <gpu_id>\033[0m"
  exit 1
fi

# ckpt_setting may be written relative to the repository root (that is how eval_config.yml writes it),
# but the server/client are started after cd'ing into finetune/RMBench, so expand it to an absolute path first.
case "${ckpt_setting}" in
  /*) : ;;
  *)  ckpt_setting="${BRIDGEVLA_ROOT}/${ckpt_setting}" ;;
esac
if [ ! -e "${ckpt_setting}" ]; then
  echo -e "\033[31m[eval] ckpt_setting does not exist: ${ckpt_setting}\033[0m"
  echo -e "\033[31m       it should be a directory holding model_*.pth + exp_cfg.yaml + mvt_cfg.yaml (or a single .pth).\033[0m"
  echo -e "\033[31m       training run directories already satisfy this; to keep a standalone copy, put model_*.pth + exp_cfg.yaml + mvt_cfg.yaml in one directory.\033[0m"
  exit 1
fi

server_gpu=$(echo $gpu_id | cut -d',' -f1)
client_gpu=$(echo $gpu_id | cut -d',' -f2)
[ -z "$client_gpu" ] && client_gpu=$server_gpu

echo -e "\033[36m[eval] task=${task_name}  config=${task_config}  seed=${seed}\033[0m"
echo -e "\033[36m[eval] ckpt=${ckpt_setting}\033[0m"

echo -e "\033[33mgpu id (to use): server=${server_gpu}, client=${client_gpu}\033[0m"
if [ "${stage2_force_center_on}" = "1" ]; then
  echo -e "\033[35m[eval] Stage-2 translation: FORCE zoom center (skip mvt2 trans argmax; rot/grip still from mvt2)\033[0m"
elif [ "${stage2_sparsePC_force_center_on}" = "1" ]; then
  echo -e "\033[35m[eval] Stage-2 translation: sparse-PC fallback ON (fallback to stage1 when zoom too empty)\033[0m"
else
  echo -e "\033[35m[eval] Stage-2 translation: mvt2 only (sparse-PC fallback OFF)\033[0m"
fi

cd "${FINETUNE_DIR}/RMBench"   # repo root for script/*.py + envs/* imports

# task_name accepts: a single task / several space-separated tasks / all (= every non-underscore task under envs/).
# It expands into the TASKS array; the server is task-independent and started once, while the client loops over the tasks.
if [ "${task_name}" = "all" ]; then
  TASKS=()
  for f in envs/*.py; do
    b="$(basename "${f}" .py)"
    case "${b}" in _*) continue;; esac   # skip _base_task / _GLOBAL_CONFIGS etc.
    TASKS+=("${b}")
  done
else
  read -r -a TASKS <<< "${task_name}"    # split on whitespace into an array
fi
echo -e "\033[36m[eval] tasks(${#TASKS[@]}): ${TASKS[*]}\033[0m"
echo -e "\033[36m[eval] config frozen at start -> ${EVAL_CONFIG_SNAPSHOT}\033[0m"

# Server needs bridgevla on PYTHONPATH (model code + RMBench_vla utils).
SERVER_PYTHONPATH="${FINETUNE_DIR}/RMBench_vla:${FINETUNE_DIR}:${FINETUNE_DIR}/RMBench/policy/BridgeVLA:${FINETUNE_DIR}/bridgevla/libs/point-renderer:${FINETUNE_DIR}/bridgevla/libs/peract_colab:${FINETUNE_DIR}/bridgevla/libs/YARR:${FINETUNE_DIR}/bridgevla/libs/peract:${FINETUNE_DIR}/GemBench"

# Find a free port.
FREE_PORT=$(python3 - << 'EOF'
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(('', 0))
    print(s.getsockname()[1])
EOF
)
echo -e "\033[33mUsing socket port: ${FREE_PORT}\033[0m"

# ---- Server (bridgevla / gembench env) ----
# The model (PaliGemma) is task-independent (the server's get_model only uses ckpt_setting), so it is
# started once even for several tasks; the instruction is sent by the client on every step. --task_name
# only receives the first task (a single token, otherwise a multi-word value would break the --overrides key/val pairing); the server does not actually use it.
echo -e "\033[32m[server] Activating conda env: ${GEMBENCH_CONDA_ENV}\033[0m"
# set -m: makes the background subshell its own process-group leader (PGID = $!) so cleanup can kill the
# whole group; exec inside the subshell means python replaces the subshell (same PID), so $! is python's PID.
set -m
(
  source "${CONDA_BASE}/bin/activate" "${GEMBENCH_CONDA_ENV}"
  export PYTHONPATH="${SERVER_PYTHONPATH}:${PYTHONPATH:-}"
  export CUDA_VISIBLE_DEVICES=${server_gpu}
  export PYTHONWARNINGS=ignore::UserWarning
  export stage2_force_center="${stage2_force_center}"
  export stage2_sparsePC_force_center="${stage2_sparsePC_force_center}"
  exec python script/policy_model_server.py \
      --port ${FREE_PORT} \
      --config policy/${policy_name}/deploy_policy.yml \
      --overrides \
      --task_name ${TASKS[0]} \
      --task_config ${task_config} \
      --ckpt_setting "${ckpt_setting}" \
      --seed ${seed} \
      --policy_name ${policy_name}
) &
SERVER_PID=$!
set +m

# Give the server time to load PaliGemma + the checkpoint.
sleep 5

# ---- Client (RMBench env, runs SAPIEN) ----
# The server can accept several client connections in sequence (each client disconnects when done), so the tasks are run in a loop.
echo -e "\033[34m[client] Activating conda env: ${RMBENCH_CONDA_ENV}\033[0m"
source "${CONDA_BASE}/bin/activate" "${RMBENCH_CONDA_ENV}"
# SAPIEN rendering needs a Vulkan ICD; defaults to the standard location of the system NVIDIA driver, overridable.
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export CUDA_VISIBLE_DEVICES=${client_gpu}

for t in "${TASKS[@]}"; do
  echo -e "\033[34m[client] ===== task: ${t} (${task_config}, seed=${seed}) =====\033[0m"
  PYTHONWARNINGS=ignore::UserWarning \
  python script/eval_policy_client.py \
      --port ${FREE_PORT} \
      --config policy/${policy_name}/deploy_policy.yml \
      --overrides \
      --task_name ${t} \
      --task_config ${task_config} \
      --ckpt_setting "${ckpt_setting}" \
      --seed ${seed} \
      --policy_name ${policy_name}
done

echo -e "\033[33m[main] all client tasks finished; server will be terminated.\033[0m"
