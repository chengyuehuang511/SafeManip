#!/usr/bin/bash

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RUN_SCRIPTS_DIR="${PROJECT_ROOT}/eval/run_scripts"
if [[ -f "${RUN_SCRIPTS_DIR}/.local_paths.sh" ]]; then
  # shellcheck disable=SC1091
  source "${RUN_SCRIPTS_DIR}/.local_paths.sh"
fi
cd "${PROJECT_ROOT}"

job_name="eval_cosmos_policy_libero_suite"
output_dir="logs/eval/${job_name}"
base_video_dir="${VIDEO_DIR:-${BASE_VIDEO_DIR:-${PROJECT_ROOT}/eval/saved_eval_rollouts/libero/cosmos_policy}}"

# 4 suites only (single combined checkpoint trained jointly across all 4,
# no libero_90) -- same scope as the other LIBERO sweep launchers.
task_suites=(
  "libero_spatial"
  "libero_object"
  "libero_goal"
  "libero_10"
)

num_trials_per_task="${NUM_TRIALS_PER_TASK:-50}"

run_env="NUM_TRIALS_PER_TASK=${num_trials_per_task}"
if [[ -n "${SAVE_REPLAY:-}" ]]; then
  run_env="${run_env},SAVE_REPLAY=${SAVE_REPLAY}"
fi

mkdir -p "${output_dir}"
mkdir -p "${base_video_dir}"
task_suite_list=$(IFS=:; echo "${task_suites[*]}")
array_end=$(( ${#task_suites[@]} - 1 ))
array_range="${ARRAY_RANGE:-0-${array_end}}"

gpu_type="${GPU_TYPE:-a40}"

sbatch_args=(
  --array="${array_range}" \
  --export="ALL,TASK_SUITE_LIST=${task_suite_list},VIDEO_DIR=${base_video_dir},${run_env}" \
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
echo "Submitted job array ${job_name} for ${#task_suites[@]} suites with ${run_env}"
echo "Video output dir: ${base_video_dir}"
