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

# Runs eval/single_task/run_single_task_groot.py against the PRISTINE
# eval/models/Isaac-GR00T submodule (robocasa-benchmark/Isaac-GR00T, no
# SafeManip additions) and eval/simulators/robocasa (pristine
# robocasa/robocasa). This intentionally does NOT reproduce the determinism
# env-var plumbing (GR00T_TORCH_DETERMINISTIC, GR00T_POLICY_COMPUTE_DTYPE,
# ...) or the privileged-info/monitor wiring this script used to have --
# none of that exists in the pristine gr00t.model.policy/gr00t.eval.simulation,
# since it was a *_safemanip-only addition (see git history for the previous
# version of this file if you need that feature set again). Replayable
# rollout saving here goes through eval/single_task/replay_capture.py
# (robosuite.wrappers.DataCollectionWrapper composed from the outside), not
# the old privileged-info JSON path.

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  if [[ "$(basename "${SLURM_SUBMIT_DIR}")" == "run_scripts" ]]; then
    PROJECT_ROOT=$(cd "${SLURM_SUBMIT_DIR}/../.." && pwd)
  else
    PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
  fi
else
  PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
fi
RUN_SCRIPTS_DIR="${PROJECT_ROOT}/eval/run_scripts"
if [[ -f "${RUN_SCRIPTS_DIR}/.local_paths.sh" ]]; then
  # shellcheck disable=SC1091
  source "${RUN_SCRIPTS_DIR}/.local_paths.sh"
fi
GROOT_ROOT="${PROJECT_ROOT}/eval/models/Isaac-GR00T"
ROBOCASA_ROOT="${ROBOCASA_ROOT:-${PROJECT_ROOT}/eval/simulators/robocasa}"
SINGLE_TASK_DIR="${PROJECT_ROOT}/eval/single_task"
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

# GROOT_ROOT first (so `import gr00t` resolves to the pristine submodule,
# not any editable-installed fork elsewhere on this env's site-packages),
# then ROBOCASA_ROOT (pristine robocasa), then SINGLE_TASK_DIR (so
# run_single_task_groot.py can `import replay_capture`).
export PYTHONPATH="${GROOT_ROOT}:${ROBOCASA_ROOT}:${SINGLE_TASK_DIR}:${PYTHONPATH:-}"
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
N_EPISODES="${N_EPISODES:-10}"
N_ACTION_STEPS="${N_ACTION_STEPS:-16}"
SAVE_REPLAY="${SAVE_REPLAY:-1}"

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
echo "N_EPISODES=${N_EPISODES}"
echo "N_ACTION_STEPS=${N_ACTION_STEPS}"
echo "SAVE_REPLAY=${SAVE_REPLAY}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L || true
fi

EXTRA_ARGS=()
if [[ "${SAVE_REPLAY}" == "1" ]]; then
  EXTRA_ARGS+=(--save_replay)
fi

srun_status=0
srun python "${SINGLE_TASK_DIR}/run_single_task_groot.py" \
  --model_path "${MODEL_PATH}" \
  --task "${TASK}" \
  --split "${SPLIT}" \
  --video_dir "${VIDEO_DIR}" \
  --port "${PORT}" \
  --n_episodes "${N_EPISODES}" \
  --n_action_steps "${N_ACTION_STEPS}" \
  "${EXTRA_ARGS[@]}" || srun_status="$?"

exit "${srun_status}"
