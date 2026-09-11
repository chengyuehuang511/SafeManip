#!/usr/bin/bash

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
GROOT_ROOT="${PROJECT_ROOT}/eval/models/Isaac-GR00T"
RUN_SCRIPTS_DIR="${PROJECT_ROOT}/eval/run_scripts"
if [[ -f "${RUN_SCRIPTS_DIR}/.local_paths.sh" ]]; then
  # shellcheck disable=SC1091
  source "${RUN_SCRIPTS_DIR}/.local_paths.sh"
fi
cd "${PROJECT_ROOT}"

# job_name="eval_target_posttraining"
job_name="eval_groot_single_task"
output_dir="logs/eval/${job_name}"
base_video_dir="${VIDEO_DIR:-${BASE_VIDEO_DIR:-${PROJECT_ROOT}/results/groot}}"

task_names=(
  "CloseBlenderLid"
  # "CloseFridge"
  # "CloseToasterOvenDoor"
  # "CoffeeSetupMug"
  # "NavigateKitchen"
  # "OpenCabinet"
  # "OpenDrawer"
  # "OpenStandMixerHead"
  # "PickPlaceCounterToCabinet"
  # "PickPlaceCounterToStove"
  # "PickPlaceDrawerToCounter"
  # "PickPlaceSinkToCounter"
  # "PickPlaceToasterToCounter"
  # "SlideDishwasherRack"
  # "TurnOffStove"
  # "TurnOnElectricKettle"
  # "TurnOnMicrowave"
  # "TurnOnSinkFaucet"
  # "DeliverStraw"
  # "GetToastedBread"
  # "KettleBoiling"
  # "LoadDishwasher"
  # "PackIdenticalLunches"
  # "PreSoakPan"
  # "PrepareCoffee"
  # "RinseSinkBasin"
  # "ScrubCuttingBoard"
  # "SearingMeat"
  # "SetUpCuttingStation"
  # "StackBowlsCabinet"
  # "SteamInMicrowave"
  # "StirVegetables"
  # "StoreLeftoversInBowl"
  # "WashLettuce"
  # "ArrangeBreadBasket"
  # "ArrangeTea"
  # "BreadSelection"
  # "CategorizeCondiments"
  # "CuttingToolSelection"
  # "GarnishPancake"
  # "GatherTableware"
  # "HeatKebabSandwich"
  # "MakeIceLemonade"
  # "PanTransfer"
  # "PortionHotDogs"
  # "RecycleBottlesByType"
  # "SeparateFreezerRack"
  # "WaffleReheat"
  # "WashFruitColander"
  # "WeighIngredients"
)
seed="${SEED:-42}"
save_replay="${SAVE_REPLAY:-1}"
foundation_model_root="${FOUNDATION_MODEL_ROOT:-${GROOT_CHECKPOINT_ROOT}/foundation_model_learning}"
# MODEL_FAMILY choices:
#   target_posttraining: task-set-specific checkpoints under target_posttraining/{atomic_seen,composite_seen,composite_unseen}
#   target_only: task-set-specific checkpoints under target_only/{atomic_seen,composite_seen,composite_unseen}
#   pretraining: shared checkpoint under pretraining/checkpoint-80000
#   multitask_learning: shared checkpoint under gr00t_n1-5/multitask_learning/checkpoint-120000
# You can also bypass this with MODEL_PATH=/path/to/checkpoint when submitting.
model_family="${MODEL_FAMILY:-target_posttraining}"
video_dir="${VIDEO_OUTPUT_DIR:-${base_video_dir}/${model_family}}"
model_path="${MODEL_PATH:-}"
model_env="FOUNDATION_MODEL_ROOT=${foundation_model_root},MODEL_FAMILY=${model_family}"
if [[ -n "${model_path}" ]]; then
  model_env="${model_env},MODEL_PATH=${model_path}"
fi

# Pristine gr00t.model.policy/gr00t.eval.simulation reads none of the
# GR00T_* determinism/autocast env vars the *_safemanip fork used to
# understand (see git history for that version of this file), so this is
# just SEED + whether to save a replayable rollout dataset alongside the
# eval videos.
run_env="SEED=${seed},SAVE_REPLAY=${save_replay}"

mkdir -p "${output_dir}"
mkdir -p "${video_dir}"
task_list=$(IFS=:; echo "${task_names[*]}")
array_end=$(( ${#task_names[@]} - 1 ))

sbatch_args=(
  --array="0-${array_end}" \
  --export="ALL,SPLIT=target,TASK_LIST=${task_list},VIDEO_DIR=${video_dir},${run_env},${model_env}" \
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

sbatch "${sbatch_args[@]}" "eval/run_scripts/${job_name}.sh"
echo "Submitted job array ${job_name} for ${#task_names[@]} tasks with ${run_env}"
echo "Checkpoint selection: ${model_env}"
echo "Video output dir: ${video_dir}"
