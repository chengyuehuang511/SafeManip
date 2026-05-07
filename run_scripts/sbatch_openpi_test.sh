#!/usr/bin/bash

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUN_SCRIPTS_DIR="${PROJECT_ROOT}/run_scripts"
if [[ -f "${RUN_SCRIPTS_DIR}/.local_paths.sh" ]]; then
  # shellcheck disable=SC1091
  source "${RUN_SCRIPTS_DIR}/.local_paths.sh"
fi
cd "${PROJECT_ROOT}"

job_name="eval_openpi_single_task"
output_dir="logs/eval/${job_name}"
base_video_dir="${VIDEO_DIR:-${BASE_VIDEO_DIR:-${PROJECT_ROOT}/results/openpi}}"

task_names=(
  "CloseBlenderLid"
  "CloseFridge"
  "CloseToasterOvenDoor"
  "CoffeeSetupMug"
  "NavigateKitchen"
  "OpenCabinet"
  "OpenDrawer"
  "OpenStandMixerHead"
  "PickPlaceCounterToCabinet"
  "PickPlaceCounterToStove"
  "PickPlaceDrawerToCounter"
  "PickPlaceSinkToCounter"
  "PickPlaceToasterToCounter"
  "SlideDishwasherRack"
  "TurnOffStove"
  "TurnOnElectricKettle"
  "TurnOnMicrowave"
  "TurnOnSinkFaucet"
  "DeliverStraw"
  "GetToastedBread"
  "KettleBoiling"
  "LoadDishwasher"
  "PackIdenticalLunches"
  "PreSoakPan"
  "PrepareCoffee"
  "RinseSinkBasin"
  "ScrubCuttingBoard"
  "SearingMeat"
  "SetUpCuttingStation"
  "StackBowlsCabinet"
  "SteamInMicrowave"
  "StirVegetables"
  "StoreLeftoversInBowl"
  "WashLettuce"
  "ArrangeBreadBasket"
  "ArrangeTea"
  "BreadSelection"
  "CategorizeCondiments"
  "CuttingToolSelection"
  "GarnishPancake"
  "GatherTableware"
  "HeatKebabSandwich"
  "MakeIceLemonade"
  "PanTransfer"
  "PortionHotDogs"
  "RecycleBottlesByType"
  "SeparateFreezerRack"
  "WaffleReheat"
  "WashFruitColander"
  "WeighIngredients"
)

seed="${SEED:-42}"
model_variant="${OPENPI_MODEL_VARIANT:-pi0}"
model_family="${OPENPI_MODEL_FAMILY:-pretraining}"
video_dir="${VIDEO_OUTPUT_DIR:-${base_video_dir}/${model_variant}/${model_family}}"
log_dir="${LOG_DIR:-${video_dir}}"
save_privileged_info="${SAVE_PRIVILEGED_INFO:-1}"
run_monitor="${RUN_MONITOR:-1}"
privileged_trajectory_horizon="${PRIVILEGED_TRAJECTORY_HORIZON:-128}"
sceneflow_root="${SCENEFLOW_ROOT:-${PROJECT_ROOT}}"
replan_steps="${REPLAN_STEPS:-5}"

model_env="OPENPI_MODEL_VARIANT=${model_variant},OPENPI_MODEL_FAMILY=${model_family},OPENPI_HF_SNAPSHOT=${OPENPI_HF_SNAPSHOT},LOG_DIR=${log_dir},VIDEO_DIR=${video_dir}"
if [[ -n "${OPENPI_CONFIG:-}" ]]; then
  model_env="${model_env},OPENPI_CONFIG=${OPENPI_CONFIG}"
fi
if [[ -n "${OPENPI_CHECKPOINT_DIR:-}" ]]; then
  model_env="${model_env},OPENPI_CHECKPOINT_DIR=${OPENPI_CHECKPOINT_DIR}"
fi
if [[ -n "${OPENPI_CHECKPOINT_ROOT:-}" ]]; then
  model_env="${model_env},OPENPI_CHECKPOINT_ROOT=${OPENPI_CHECKPOINT_ROOT}"
fi
if [[ -n "${OPENPI_EXP_NAME:-}" ]]; then
  model_env="${model_env},OPENPI_EXP_NAME=${OPENPI_EXP_NAME}"
fi
if [[ -n "${OPENPI_TARGET_CONFIG:-}" ]]; then
  model_env="${model_env},OPENPI_TARGET_CONFIG=${OPENPI_TARGET_CONFIG}"
fi

mkdir -p "${output_dir}"
mkdir -p "${log_dir}"
mkdir -p "${video_dir}"

task_list=$(IFS=:; echo "${task_names[*]}")
array_end=$(( ${#task_names[@]} - 1 ))

sbatch_args=(
  --array="0-${array_end}" \
  --export="ALL,TASK_LIST=${task_list},SPLIT=target,SEED=${seed},N_EPISODES=${N_EPISODES:-50},SAVE_PRIVILEGED_INFO=${save_privileged_info},RUN_MONITOR=${run_monitor},PRIVILEGED_TRAJECTORY_HORIZON=${privileged_trajectory_horizon},SCENEFLOW_ROOT=${sceneflow_root},REPLAN_STEPS=${replan_steps},${model_env}" \
  --job-name="${job_name}" \
  --output="${output_dir}/${job_name}-slurm-%A_%a.out" \
  --error="${output_dir}/${job_name}-slurm-%A_%a.err"
)
if [[ -n "${SLURM_PARTITION:-}" ]]; then
  sbatch_args+=(--partition="${SLURM_PARTITION}")
fi
if [[ -n "${SLURM_EXCLUDE_NODES:-}" ]]; then
  sbatch_args+=(--exclude="${SLURM_EXCLUDE_NODES}")
fi

sbatch "${sbatch_args[@]}" "run_scripts/eval_openpi_single_task.sh"

echo "Submitted ${job_name} for ${#task_names[@]} tasks"
echo "OpenPI selection: ${model_env}"
echo "Video output dir: ${video_dir}"
