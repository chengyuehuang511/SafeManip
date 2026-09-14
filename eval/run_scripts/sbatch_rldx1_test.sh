#!/usr/bin/bash

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RUN_SCRIPTS_DIR="${PROJECT_ROOT}/eval/run_scripts"
if [[ -f "${RUN_SCRIPTS_DIR}/.local_paths.sh" ]]; then
  # shellcheck disable=SC1091
  source "${RUN_SCRIPTS_DIR}/.local_paths.sh"
fi
cd "${PROJECT_ROOT}"

job_name="eval_rldx1_single_task"
output_dir="logs/eval/${job_name}"
base_video_dir="${VIDEO_DIR:-${BASE_VIDEO_DIR:-${PROJECT_ROOT}/results/rldx1}}"

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
save_replay="${SAVE_REPLAY:-1}"
n_episodes="${N_EPISODES:-50}"
model_path="${MODEL_PATH:-${HOME}/flash/checkpoints/RLDX-1-FT-RC365}"
video_dir="${VIDEO_OUTPUT_DIR:-${base_video_dir}}"

run_env="SEED=${seed},SAVE_REPLAY=${save_replay},N_EPISODES=${n_episodes},MODEL_PATH=${model_path}"

mkdir -p "${output_dir}"
mkdir -p "${video_dir}"
task_list=$(IFS=:; echo "${task_names[*]}")
array_end=$(( ${#task_names[@]} - 1 ))

gpu_type="${GPU_TYPE:-a40}"

sbatch_args=(
  --array="0-${array_end}" \
  --export="ALL,SPLIT=${SPLIT:-target},TASK_LIST=${task_list},VIDEO_DIR=${video_dir},${run_env}" \
  --job-name="${job_name}" \
  --gpus-per-node="${gpu_type}:1" \
  --output="${output_dir}/${job_name}-slurm-%A_%a.out" \
  --error="${output_dir}/${job_name}-slurm-%A_%a.err"
)
if [[ -n "${SLURM_PARTITION:-}" ]]; then
  sbatch_args+=(--partition="${SLURM_PARTITION}")
fi
if [[ -n "${SLURM_EXCLUDE_NODES:-}" ]]; then
  sbatch_args+=(--exclude="${SLURM_EXCLUDE_NODES}")
fi

sbatch "${sbatch_args[@]}" "eval/run_scripts/${job_name}.sh"
echo "Submitted job array ${job_name} for ${#task_names[@]} tasks with ${run_env} on ${gpu_type} GPUs"
echo "Video output dir: ${video_dir}"
