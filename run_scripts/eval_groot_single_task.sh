#!/usr/bin/bash
#SBATCH --job-name=eval_groot_single_task
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=16
#SBATCH --gpus-per-node="l40s:1"
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
GROOT_ROOT="${PROJECT_ROOT}/Isaac-GR00T"
cd "${GROOT_ROOT}"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-robocasa}"

if [[ -f "${CONDA_SH}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
else
  echo "Missing conda initialization script: ${CONDA_SH}" >&2
  exit 1
fi

conda activate "${CONDA_ENV_NAME}"

export PYTHONPATH="${GROOT_ROOT}:${PYTHONPATH:-}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/${USER}/triton-${SLURM_JOB_ID:-local}}"
mkdir -p "${TRITON_CACHE_DIR}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
if [[ -z "${MUJOCO_EGL_DEVICE_ID:-}" ]]; then
  export MUJOCO_EGL_DEVICE_ID="${CUDA_VISIBLE_DEVICES%%,*}"
fi
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"

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
TASK="${TASK:-PrepareCoffee}"
SPLIT="${SPLIT:-target}"
TASK_SET="${TASK_SET:-}"
MODEL_PATH="${MODEL_PATH:-}"
VIDEO_DIR="${VIDEO_DIR:-${GROOT_ROOT}/videos_single_task}"
if [[ -z "${PORT:-}" ]]; then
  PORT="$((5555 + (${SLURM_JOB_ID:-0} % 20000)))"
fi
SEED="${SEED:-42}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-${SEED}}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG-:4096:8}"
export NVIDIA_TF32_OVERRIDE="${NVIDIA_TF32_OVERRIDE-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM-false}"
export GR00T_POLICY_COMPUTE_DTYPE="${GR00T_POLICY_COMPUTE_DTYPE-float32}"
export GR00T_POLICY_DISABLE_AUTOCAST="${GR00T_POLICY_DISABLE_AUTOCAST-1}"
export GR00T_DISABLE_FLASH_ATTN="${GR00T_DISABLE_FLASH_ATTN-1}"
export GR00T_SEED_PYTHON_RANDOM="${GR00T_SEED_PYTHON_RANDOM-1}"
export GR00T_SEED_NUMPY="${GR00T_SEED_NUMPY-1}"
export GR00T_SEED_TORCH="${GR00T_SEED_TORCH-1}"
export GR00T_SEED_CUDA="${GR00T_SEED_CUDA-1}"
export GR00T_CUDNN_BENCHMARK="${GR00T_CUDNN_BENCHMARK-0}"
export GR00T_CUDNN_DETERMINISTIC="${GR00T_CUDNN_DETERMINISTIC-1}"
export GR00T_MATMUL_ALLOW_TF32="${GR00T_MATMUL_ALLOW_TF32-0}"
export GR00T_CUDNN_ALLOW_TF32="${GR00T_CUDNN_ALLOW_TF32-0}"
export GR00T_TORCH_DETERMINISTIC="${GR00T_TORCH_DETERMINISTIC-1}"
export GR00T_TORCH_DETERMINISTIC_WARN_ONLY="${GR00T_TORCH_DETERMINISTIC_WARN_ONLY-1}"
export GR00T_PER_ACTION_SEED="${GR00T_PER_ACTION_SEED-1}"
N_EPISODES="${N_EPISODES:-10}"
N_ENVS="${N_ENVS:-1}"
N_ACTION_STEPS="${N_ACTION_STEPS:-16}"
SAVE_REPLAY_PACKAGE="${SAVE_REPLAY_PACKAGE:-1}"
SAVE_PRIVILEGED_INFO="${SAVE_PRIVILEGED_INFO:-1}"
RENDER_PRIVILEGED_VIDEO="${RENDER_PRIVILEGED_VIDEO:-0}"
ISOLATE_PRIVILEGED_INFO="${ISOLATE_PRIVILEGED_INFO:-0}"
ROLLOUT_VIDEO_SOURCE="${ROLLOUT_VIDEO_SOURCE:-obs}"
ROLLOUT_SIM_RENDER_WARMUP_PASSES="${ROLLOUT_SIM_RENDER_WARMUP_PASSES:-0}"
STABILIZE_ROLLOUT_VIDEO="${STABILIZE_ROLLOUT_VIDEO:-1}"
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

FOUNDATION_MODEL_ROOT="${FOUNDATION_MODEL_ROOT:-${GROOT_CHECKPOINT_ROOT}/foundation_model_learning}"
MODEL_FAMILY="${MODEL_FAMILY:-target_posttraining}"
PRETRAINING_MODEL_PATH="${PRETRAINING_MODEL_PATH:-${FOUNDATION_MODEL_ROOT}/pretraining/checkpoint-80000}"
ATOMIC_SEEN_MODEL_PATH="${ATOMIC_SEEN_MODEL_PATH:-${FOUNDATION_MODEL_ROOT}/${MODEL_FAMILY}/atomic_seen/checkpoint-60000}"
COMPOSITE_SEEN_MODEL_PATH="${COMPOSITE_SEEN_MODEL_PATH:-${FOUNDATION_MODEL_ROOT}/${MODEL_FAMILY}/composite_seen/checkpoint-60000}"
COMPOSITE_UNSEEN_MODEL_PATH="${COMPOSITE_UNSEEN_MODEL_PATH:-${FOUNDATION_MODEL_ROOT}/${MODEL_FAMILY}/composite_unseen/checkpoint-60000}"

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

model_path_for_task_set() {
  if [[ "${MODEL_FAMILY}" == "pretraining" ]]; then
    echo "${PRETRAINING_MODEL_PATH}"
    return 0
  fi
  if [[ "${MODEL_FAMILY}" == "multitask_learning" ]]; then
    echo "${MULTITASK_LEARNING_MODEL_PATH}"
    return 0
  fi
  case "$1" in
    atomic_seen)
      echo "${ATOMIC_SEEN_MODEL_PATH}"
      ;;
    composite_seen)
      echo "${COMPOSITE_SEEN_MODEL_PATH}"
      ;;
    composite_unseen)
      echo "${COMPOSITE_UNSEEN_MODEL_PATH}"
      ;;
    *)
      return 1
      ;;
  esac
}

if [[ -z "${TASK_SET}" ]]; then
  if ! TASK_SET="$(infer_task_set "${TASK}")"; then
    echo "Unable to infer TASK_SET for TASK=${TASK}; set TASK_SET or MODEL_PATH explicitly." >&2
    exit 1
  fi
fi

if [[ -z "${MODEL_PATH}" ]]; then
  if ! MODEL_PATH="$(model_path_for_task_set "${TASK_SET}")"; then
    echo "Unable to select MODEL_PATH for TASK_SET=${TASK_SET}; set MODEL_PATH explicitly." >&2
    exit 1
  fi
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH does not exist or is not a directory: ${MODEL_PATH}" >&2
  echo "MODEL_FAMILY=${MODEL_FAMILY} FOUNDATION_MODEL_ROOT=${FOUNDATION_MODEL_ROOT} TASK_SET=${TASK_SET}" >&2
  exit 1
fi

echo "Hostname: $(hostname)"
echo "Working directory: ${GROOT_ROOT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "MUJOCO_GL=${MUJOCO_GL}"
echo "MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID}"
echo "MODEL_FAMILY=${MODEL_FAMILY}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "TASK=${TASK}"
echo "TASK_SET=${TASK_SET}"
echo "SPLIT=${SPLIT}"
echo "VIDEO_DIR=${VIDEO_DIR}"
echo "PORT=${PORT}"
echo "SEED=${SEED}"
echo "PYTHONHASHSEED=${PYTHONHASHSEED}"
echo "CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG}"
echo "NVIDIA_TF32_OVERRIDE=${NVIDIA_TF32_OVERRIDE}"
echo "TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM}"
echo "GR00T_POLICY_COMPUTE_DTYPE=${GR00T_POLICY_COMPUTE_DTYPE}"
echo "GR00T_POLICY_DISABLE_AUTOCAST=${GR00T_POLICY_DISABLE_AUTOCAST}"
echo "GR00T_DISABLE_FLASH_ATTN=${GR00T_DISABLE_FLASH_ATTN}"
echo "GR00T_SEED_PYTHON_RANDOM=${GR00T_SEED_PYTHON_RANDOM}"
echo "GR00T_SEED_NUMPY=${GR00T_SEED_NUMPY}"
echo "GR00T_SEED_TORCH=${GR00T_SEED_TORCH}"
echo "GR00T_SEED_CUDA=${GR00T_SEED_CUDA}"
echo "GR00T_CUDNN_BENCHMARK=${GR00T_CUDNN_BENCHMARK}"
echo "GR00T_CUDNN_DETERMINISTIC=${GR00T_CUDNN_DETERMINISTIC}"
echo "GR00T_MATMUL_ALLOW_TF32=${GR00T_MATMUL_ALLOW_TF32}"
echo "GR00T_CUDNN_ALLOW_TF32=${GR00T_CUDNN_ALLOW_TF32}"
echo "GR00T_TORCH_DETERMINISTIC=${GR00T_TORCH_DETERMINISTIC}"
echo "GR00T_TORCH_DETERMINISTIC_WARN_ONLY=${GR00T_TORCH_DETERMINISTIC_WARN_ONLY}"
echo "GR00T_PER_ACTION_SEED=${GR00T_PER_ACTION_SEED}"
echo "N_EPISODES=${N_EPISODES}"
echo "N_ENVS=${N_ENVS}"
echo "N_ACTION_STEPS=${N_ACTION_STEPS}"
echo "SAVE_REPLAY_PACKAGE=${SAVE_REPLAY_PACKAGE}"
echo "SAVE_PRIVILEGED_INFO=${SAVE_PRIVILEGED_INFO}"
echo "RENDER_PRIVILEGED_VIDEO=${RENDER_PRIVILEGED_VIDEO}"
echo "ISOLATE_PRIVILEGED_INFO=${ISOLATE_PRIVILEGED_INFO}"
echo "ROLLOUT_VIDEO_SOURCE=${ROLLOUT_VIDEO_SOURCE}"
echo "ROLLOUT_SIM_RENDER_WARMUP_PASSES=${ROLLOUT_SIM_RENDER_WARMUP_PASSES}"
echo "STABILIZE_ROLLOUT_VIDEO=${STABILIZE_ROLLOUT_VIDEO}"
echo "PRIVILEGED_TRAJECTORY_HORIZON=${PRIVILEGED_TRAJECTORY_HORIZON}"
echo "SCENEFLOW_ROOT=${SCENEFLOW_ROOT}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L || true
fi

EXTRA_ARGS=()
if [[ "${SAVE_REPLAY_PACKAGE}" == "1" ]]; then
  EXTRA_ARGS+=(--save_replay_package)
fi
if [[ "${SAVE_PRIVILEGED_INFO}" == "1" ]]; then
  EXTRA_ARGS+=(--save_privileged_info)
fi
if [[ "${RENDER_PRIVILEGED_VIDEO}" == "1" ]]; then
  EXTRA_ARGS+=(--render_privileged_video)
fi

srun_status=0
srun python scripts/run_single_task.py \
  --model_path "${MODEL_PATH}" \
  --task "${TASK}" \
  --split "${SPLIT}" \
  --video_dir "${VIDEO_DIR}" \
  --port "${PORT}" \
  --seed "${SEED}" \
  --n_episodes "${N_EPISODES}" \
  --n_envs "${N_ENVS}" \
  --n_action_steps "${N_ACTION_STEPS}" \
  --privileged_trajectory_horizon "${PRIVILEGED_TRAJECTORY_HORIZON}" \
  "${EXTRA_ARGS[@]}" || srun_status="$?"

exit "${srun_status}"
