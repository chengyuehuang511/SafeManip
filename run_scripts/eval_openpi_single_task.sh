#!/usr/bin/bash
#SBATCH --job-name=eval_openpi_single_task
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=16
#SBATCH --gpus-per-node="a40:1"
#SBATCH --qos="short"
#SBATCH --mem-per-gpu=45G
#SBATCH --time=24:00:00
#SBATCH --array=0

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  if [[ "$(basename "${SLURM_SUBMIT_DIR}")" == "run_scripts" ]]; then
    PROJECT_ROOT=$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)
  else
    PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
  fi
else
  PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
fi
RUN_SCRIPTS_DIR="${PROJECT_ROOT}/run_scripts"
if [[ -f "${RUN_SCRIPTS_DIR}/.local_paths.sh" ]]; then
  # shellcheck disable=SC1091
  source "${RUN_SCRIPTS_DIR}/.local_paths.sh"
fi
OPENPI_ROOT="${PROJECT_ROOT}/openpi"
cd "${OPENPI_ROOT}"

OPENPI_CONDA_ENV_NAME="${OPENPI_CONDA_ENV_NAME:-${CONDA_ENV_NAME:-openpi-robocasa}}"
ROBOCASA_CONDA_ENV_NAME="${ROBOCASA_CONDA_ENV_NAME:-robocasa}"

if [[ -f "${CONDA_SH}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
else
  echo "Missing conda initialization script: ${CONDA_SH}" >&2
  exit 1
fi

conda activate "${OPENPI_CONDA_ENV_NAME}"

ROBOCASA_ROOT="${ROBOCASA_ROOT:-${PROJECT_ROOT}/robocasa}"
ROBOSUITE_ROOT="${ROBOSUITE_ROOT:-${PROJECT_ROOT}/robosuite}"
export PYTHONPATH="${OPENPI_ROOT}:${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src:${ROBOCASA_ROOT}:${ROBOSUITE_ROOT}:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
if [[ -z "${MUJOCO_EGL_DEVICE_ID:-}" ]]; then
  export MUJOCO_EGL_DEVICE_ID="${CUDA_VISIBLE_DEVICES%%,*}"
fi
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-${SEED:-42}}"
export NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-1}"

TASK="${TASK:-}"
TASK_LIST="${TASK_LIST:-}"
if [[ -z "${TASK}" && -n "${TASK_LIST}" ]]; then
  IFS=':' read -r -a TASK_LIST_ITEMS <<< "${TASK_LIST}"
  if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "TASK_LIST provided but SLURM_ARRAY_TASK_ID is missing." >&2
    exit 1
  fi
  if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#TASK_LIST_ITEMS[@]} )); then
    echo "Invalid SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}; expected 0-$(( ${#TASK_LIST_ITEMS[@]} - 1 ))" >&2
    exit 1
  fi
  TASK="${TASK_LIST_ITEMS[SLURM_ARRAY_TASK_ID]}"
fi
TASK="${TASK:-PickPlaceCounterToCabinet}"

SPLIT="${SPLIT:-target}"
SEED="${SEED:-42}"
N_EPISODES="${N_EPISODES:-10}"
RESIZE_SIZE="${RESIZE_SIZE:-224}"
REPLAN_STEPS="${REPLAN_STEPS:-5}"
SAVE_PRIVILEGED_INFO="${SAVE_PRIVILEGED_INFO:-1}"
RUN_MONITOR="${RUN_MONITOR:-1}"
PRIVILEGED_TRAJECTORY_HORIZON="${PRIVILEGED_TRAJECTORY_HORIZON:-128}"
SCENEFLOW_ROOT="${SCENEFLOW_ROOT:-${PROJECT_ROOT}}"
MONITOR_ROOT="${MONITOR_ROOT:-${SCENEFLOW_ROOT}/SafeManip/monitor}"
if [[ ! -f "${MONITOR_ROOT}/run_monitor_on_privileged.py" ]]; then
  if [[ -f "${SCENEFLOW_ROOT}/run_monitor_on_privileged.py" ]]; then
    MONITOR_ROOT="${SCENEFLOW_ROOT}"
  elif [[ -f "${SCENEFLOW_ROOT}/monitor/run_monitor_on_privileged.py" ]]; then
    MONITOR_ROOT="${SCENEFLOW_ROOT}/monitor"
  fi
fi
OPENPI_MODEL_VARIANT="${OPENPI_MODEL_VARIANT:-pi0}"
OPENPI_MODEL_FAMILY="${OPENPI_MODEL_FAMILY:-pretraining}"
OPENPI_CHECKPOINT_ROOT="${OPENPI_CHECKPOINT_ROOT:-${OPENPI_ROOT}/checkpoints}"
LOG_DIR="${LOG_DIR:-${OPENPI_ROOT}/logs/eval/${OPENPI_MODEL_VARIANT}/${OPENPI_MODEL_FAMILY}}"
if [[ -z "${PORT:-}" ]]; then
  PORT="$((8000 + (${SLURM_JOB_ID:-0} % 20000)))"
fi

infer_task_set() {
  case "$1" in
    CloseBlenderLid|CloseFridge|CloseToasterOvenDoor|CoffeeSetupMug|NavigateKitchen|OpenCabinet|OpenDrawer|OpenStandMixerHead|PickPlaceCounterToCabinet|PickPlaceCounterToStove|PickPlaceDrawerToCounter|PickPlaceSinkToCounter|PickPlaceToasterToCounter|SlideDishwasherRack|TurnOffStove|TurnOnElectricKettle|TurnOnMicrowave|TurnOnSinkFaucet)
      echo "atomic_seen"
      ;;
    DeliverStraw|GetToastedBread|KettleBoiling|LoadDishwasher|PackIdenticalLunches|PreSoakPan|PrepareCoffee|RinseSinkBasin|ScrubCuttingBoard|SearingMeat|SetUpCuttingStation|StackBowlsCabinet|SteamInMicrowave|StirVegetables|StoreLeftoversInBowl|WashLettuce)
      echo "composite_seen"
      ;;
    ArrangeBreadBasket|ArrangeTea|BreadSelection|CategorizeCondiments|CuttingToolSelection|GarnishPancake|GatherTableware|HeatKebabSandwich|MakeIceLemonade|PanTransfer|PortionHotDogs|RecycleBottlesByType|SeparateFreezerRack|WaffleReheat|WashFruitColander|WeighIngredients)
      echo "composite_unseen"
      ;;
    *)
      return 1
      ;;
  esac
}

config_for_task_set() {
  local task_set="$1"
  # OpenPI RoboCasa evals use one checkpoint/config per model family. Unlike
  # GR00T, the atomic/composite task set only affects task routing.
  case "${OPENPI_MODEL_VARIANT}:${OPENPI_MODEL_FAMILY}:${task_set}" in
    pi0:pretraining:*)
      echo "pi0_robocasa_pretrain_human300"
      ;;
    pi0.5:pretraining:*|pi05:pretraining:*)
      echo "pi05_pretrain_human300"
      ;;
    pi0:target_only:*|pi0:target_posttraining:*)
      echo "${OPENPI_TARGET_CONFIG:-pi0_robocasa_target50}"
      ;;
    pi0.5:target_only:*|pi05:target_only:*|pi0.5:target_posttraining:*|pi05:target_posttraining:*)
      echo "${OPENPI_TARGET_CONFIG:-pi05_pretrain_human300}"
      ;;
    *)
      return 1
      ;;
  esac
}

snapshot_checkpoint_for_variant() {
  case "${OPENPI_MODEL_VARIANT}:${OPENPI_MODEL_FAMILY}" in
    pi0:pretraining)
      echo "${OPENPI_HF_SNAPSHOT}/pi0/pi0_robocasa_pretrain_human300/multitask_learning/75000"
      ;;
    pi0.5:pretraining|pi05:pretraining)
      echo "${OPENPI_HF_SNAPSHOT}/pi05_pretrain_human300/multitask_learning/75000"
      ;;
    *)
      return 1
      ;;
  esac
}

latest_checkpoint_step() {
  local exp_dir="$1"
  find "${exp_dir}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null \
    | grep -E '^[0-9]+$' \
    | sort -n \
    | tail -n 1 \
    || true
}

TASK_SET="${TASK_SET:-}"
if [[ -z "${TASK_SET}" ]]; then
  if ! TASK_SET="$(infer_task_set "${TASK}")"; then
    echo "Unable to infer TASK_SET for TASK=${TASK}; set TASK_SET." >&2
    exit 1
  fi
fi

OPENPI_CONFIG="${OPENPI_CONFIG:-}"
if [[ -z "${OPENPI_CONFIG}" ]]; then
  if ! OPENPI_CONFIG="$(config_for_task_set "${TASK_SET}")"; then
    echo "No default OPENPI_CONFIG for OPENPI_MODEL_VARIANT=${OPENPI_MODEL_VARIANT} OPENPI_MODEL_FAMILY=${OPENPI_MODEL_FAMILY} TASK_SET=${TASK_SET}." >&2
    echo "Set OPENPI_CONFIG and OPENPI_CHECKPOINT_DIR explicitly for this model." >&2
    exit 1
  fi
fi

OPENPI_EXP_NAME="${OPENPI_EXP_NAME:-${OPENPI_MODEL_FAMILY}}"
OPENPI_CHECKPOINT_DIR="${OPENPI_CHECKPOINT_DIR:-}"
if [[ -z "${OPENPI_CHECKPOINT_DIR}" ]]; then
  if snapshot_checkpoint="$(snapshot_checkpoint_for_variant)"; then
    OPENPI_CHECKPOINT_DIR="${snapshot_checkpoint}"
  else
    exp_dir="${OPENPI_CHECKPOINT_ROOT}/${OPENPI_CONFIG}/${OPENPI_EXP_NAME}"
    latest_step="$(latest_checkpoint_step "${exp_dir}")"
    if [[ -z "${latest_step}" ]]; then
      echo "Could not infer checkpoint step under ${exp_dir}." >&2
      echo "Set OPENPI_CHECKPOINT_DIR to a checkpoint directory containing params/ and assets/." >&2
      exit 1
    fi
    OPENPI_CHECKPOINT_DIR="${exp_dir}/${latest_step}"
  fi
fi

if [[ ! -d "${OPENPI_CHECKPOINT_DIR}/params" ]]; then
  echo "OPENPI_CHECKPOINT_DIR must contain params/: ${OPENPI_CHECKPOINT_DIR}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"

echo "Hostname: $(hostname)"
echo "Working directory: ${OPENPI_ROOT}"
echo "OPENPI_CONDA_ENV_NAME=${OPENPI_CONDA_ENV_NAME}"
echo "ROBOCASA_CONDA_ENV_NAME=${ROBOCASA_CONDA_ENV_NAME}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "MUJOCO_GL=${MUJOCO_GL}"
echo "MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID}"
echo "OPENPI_MODEL_VARIANT=${OPENPI_MODEL_VARIANT}"
echo "OPENPI_MODEL_FAMILY=${OPENPI_MODEL_FAMILY}"
echo "OPENPI_HF_SNAPSHOT=${OPENPI_HF_SNAPSHOT}"
echo "OPENPI_CONFIG=${OPENPI_CONFIG}"
echo "OPENPI_CHECKPOINT_DIR=${OPENPI_CHECKPOINT_DIR}"
echo "TASK=${TASK}"
echo "TASK_SET=${TASK_SET}"
echo "SPLIT=${SPLIT}"
echo "LOG_DIR=${LOG_DIR}"
echo "PORT=${PORT}"
echo "SEED=${SEED}"
echo "N_EPISODES=${N_EPISODES}"
echo "RESIZE_SIZE=${RESIZE_SIZE}"
echo "REPLAN_STEPS=${REPLAN_STEPS}"
echo "SAVE_PRIVILEGED_INFO=${SAVE_PRIVILEGED_INFO}"
echo "RUN_MONITOR=${RUN_MONITOR}"
echo "PRIVILEGED_TRAJECTORY_HORIZON=${PRIVILEGED_TRAJECTORY_HORIZON}"
echo "SCENEFLOW_ROOT=${SCENEFLOW_ROOT}"
echo "NUMBA_DISABLE_JIT=${NUMBA_DISABLE_JIT}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L || true
fi

server_log="${LOG_DIR}/server-${SLURM_JOB_ID:-local}-${SLURM_ARRAY_TASK_ID:-0}.log"
python scripts/serve_policy.py \
  --port="${PORT}" \
  policy:checkpoint \
  --policy.config="${OPENPI_CONFIG}" \
  --policy.dir="${OPENPI_CHECKPOINT_DIR}" \
  > "${server_log}" 2>&1 &
server_pid="$!"

cleanup() {
  if kill -0 "${server_pid}" 2>/dev/null; then
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

server_ready=0
for _ in $(seq 1 120); do
  if python - "${PORT}" <<'PY'
import socket
import sys

port = int(sys.argv[1])
sock = socket.socket()
sock.settimeout(1.0)
try:
    sock.connect(("127.0.0.1", port))
except OSError:
    sys.exit(1)
finally:
    sock.close()
PY
  then
    server_ready=1
    break
  fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    echo "Policy server exited before accepting connections. Log follows:" >&2
    tail -n 120 "${server_log}" >&2 || true
    exit 1
  fi
  sleep 5
done

if [[ "${server_ready}" != "1" ]]; then
  echo "Policy server did not accept connections on port ${PORT}. Log follows:" >&2
  tail -n 120 "${server_log}" >&2 || true
  exit 1
fi

if ! kill -0 "${server_pid}" 2>/dev/null; then
  echo "Policy server is not running. Log follows:" >&2
  tail -n 120 "${server_log}" >&2 || true
  exit 1
fi

CLIENT_ARGS=(
  python examples/robocasa/eval_single_task.py
  --task "${TASK}" \
  --split "${SPLIT}" \
  --log-dir "${LOG_DIR}" \
  --num-trials "${N_EPISODES}" \
  --resize-size "${RESIZE_SIZE}" \
  --replan-steps "${REPLAN_STEPS}" \
  --host "127.0.0.1" \
  --port "${PORT}" \
  --seed "${SEED}" \
  --privileged-trajectory-horizon "${PRIVILEGED_TRAJECTORY_HORIZON}" \
  --sceneflow-root "${SCENEFLOW_ROOT}"
)
if [[ "${SAVE_PRIVILEGED_INFO}" == "0" ]]; then
  CLIENT_ARGS+=(--no-save-privileged-info)
fi
if [[ "${RUN_MONITOR}" == "0" ]]; then
  CLIENT_ARGS+=(--no-run-monitor)
fi

conda run -n "${ROBOCASA_CONDA_ENV_NAME}" "${CLIENT_ARGS[@]}"
