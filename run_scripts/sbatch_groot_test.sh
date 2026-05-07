#!/usr/bin/bash

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GROOT_ROOT="${PROJECT_ROOT}/Isaac-GR00T"
RUN_SCRIPTS_DIR="${PROJECT_ROOT}/run_scripts"
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
isolate_privileged_info="${ISOLATE_PRIVILEGED_INFO:-1}"
rollout_video_source="${ROLLOUT_VIDEO_SOURCE:-state_render}"
rollout_sim_render_warmup_passes="${ROLLOUT_SIM_RENDER_WARMUP_PASSES:-0}"
stabilize_rollout_video="${STABILIZE_ROLLOUT_VIDEO:-1}"
sceneflow_root="${SCENEFLOW_ROOT:-${PROJECT_ROOT}}"
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

# Preset 1: seed only, no per-action reseeding, no deterministic backend constraints.
determinism_env="SEED=${seed},ISOLATE_PRIVILEGED_INFO=${isolate_privileged_info},ROLLOUT_VIDEO_SOURCE=${rollout_video_source},ROLLOUT_SIM_RENDER_WARMUP_PASSES=${rollout_sim_render_warmup_passes},STABILIZE_ROLLOUT_VIDEO=${stabilize_rollout_video},SCENEFLOW_ROOT=${sceneflow_root},GR00T_POLICY_COMPUTE_DTYPE=bfloat16,GR00T_POLICY_DISABLE_AUTOCAST=0,GR00T_DISABLE_FLASH_ATTN=0,GR00T_TORCH_DETERMINISTIC=0,GR00T_CUDNN_BENCHMARK=1,GR00T_CUDNN_DETERMINISTIC=0,GR00T_MATMUL_ALLOW_TF32=1,GR00T_CUDNN_ALLOW_TF32=1,GR00T_PER_ACTION_SEED=0"

# Preset 2: seed + per-action reseeding, no deterministic backend constraints. Uncomment to use.
# determinism_env="SEED=${seed},GR00T_POLICY_COMPUTE_DTYPE=bfloat16,GR00T_POLICY_DISABLE_AUTOCAST=0,GR00T_DISABLE_FLASH_ATTN=0,GR00T_TORCH_DETERMINISTIC=0,GR00T_CUDNN_BENCHMARK=1,GR00T_CUDNN_DETERMINISTIC=0,GR00T_MATMUL_ALLOW_TF32=1,GR00T_CUDNN_ALLOW_TF32=1,GR00T_PER_ACTION_SEED=1"

# Preset 3: non-strict determinism. Uncomment to use.
# determinism_env="SEED=${seed},GR00T_POLICY_COMPUTE_DTYPE=bfloat16,GR00T_POLICY_DISABLE_AUTOCAST=0,GR00T_DISABLE_FLASH_ATTN=0,GR00T_TORCH_DETERMINISTIC=1,GR00T_TORCH_DETERMINISTIC_WARN_ONLY=1,GR00T_CUDNN_BENCHMARK=0,GR00T_CUDNN_DETERMINISTIC=1,GR00T_MATMUL_ALLOW_TF32=0,GR00T_CUDNN_ALLOW_TF32=0,GR00T_PER_ACTION_SEED=1,CUBLAS_WORKSPACE_CONFIG=:4096:8,NVIDIA_TF32_OVERRIDE=0"

# Preset 4: strict determinism. Uncomment to use.
# determinism_env="SEED=${seed},GR00T_POLICY_COMPUTE_DTYPE=float32,GR00T_POLICY_DISABLE_AUTOCAST=1,GR00T_DISABLE_FLASH_ATTN=1,GR00T_TORCH_DETERMINISTIC=1,GR00T_TORCH_DETERMINISTIC_WARN_ONLY=0,GR00T_CUDNN_BENCHMARK=0,GR00T_CUDNN_DETERMINISTIC=1,GR00T_MATMUL_ALLOW_TF32=0,GR00T_CUDNN_ALLOW_TF32=0,GR00T_PER_ACTION_SEED=1,CUBLAS_WORKSPACE_CONFIG=:4096:8,NVIDIA_TF32_OVERRIDE=0"

mkdir -p "${output_dir}"
mkdir -p "${video_dir}"
task_list=$(IFS=:; echo "${task_names[*]}")
array_end=$(( ${#task_names[@]} - 1 ))

sbatch_args=(
  --array="0-${array_end}" \
  --export="ALL,SPLIT=target,TASK_LIST=${task_list},VIDEO_DIR=${video_dir},${determinism_env},${model_env}" \
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

sbatch "${sbatch_args[@]}" "run_scripts/${job_name}.sh"
echo "Submitted job array ${job_name} for ${#task_names[@]} tasks with ${determinism_env}"
echo "Checkpoint selection: ${model_env}"
echo "Video output dir: ${video_dir}"
