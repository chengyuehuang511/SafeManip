#!/usr/bin/bash

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RUN_SCRIPTS_DIR="${PROJECT_ROOT}/eval/run_scripts"
if [[ -f "${RUN_SCRIPTS_DIR}/.local_paths.sh" ]]; then
  # shellcheck disable=SC1091
  source "${RUN_SCRIPTS_DIR}/.local_paths.sh"
fi
cd "${PROJECT_ROOT}"

job_name="eval_openpi_libero_suite"
output_dir="logs/eval/${job_name}"
# Matches RoboCasa's own eval/saved_eval_rollouts/<split>/<model> layout
# (see eval/launch_pretrain_split.sh) rather than a separate results/ tree.
base_video_dir="${VIDEO_DIR:-${BASE_VIDEO_DIR:-${PROJECT_ROOT}/eval/saved_eval_rollouts/libero/openpi}}"

# Plain LIBERO only (no libero_90/libero_plus/libero_pro) -- matches the 4
# suites both openpi's own README and the GR00T-family LIBERO checkpoints
# report against.
task_suites=(
  "libero_spatial"
  "libero_object"
  "libero_goal"
  "libero_10"
)

model_variant="${OPENPI_MODEL_VARIANT:-pi0.5}"
video_dir="${VIDEO_OUTPUT_DIR:-${base_video_dir}/${model_variant}}"
num_trials_per_task="${NUM_TRIALS_PER_TASK:-50}"

run_env="OPENPI_MODEL_VARIANT=${model_variant},NUM_TRIALS_PER_TASK=${num_trials_per_task}"
if [[ -n "${OPENPI_CONFIG:-}" ]]; then
  run_env="${run_env},OPENPI_CONFIG=${OPENPI_CONFIG}"
fi
if [[ -n "${OPENPI_CHECKPOINT_DIR:-}" ]]; then
  run_env="${run_env},OPENPI_CHECKPOINT_DIR=${OPENPI_CHECKPOINT_DIR}"
fi

mkdir -p "${output_dir}"
mkdir -p "${video_dir}"
task_suite_list=$(IFS=:; echo "${task_suites[*]}")
array_end=$(( ${#task_suites[@]} - 1 ))
# Overridable so a subset of suites (e.g. one that failed) can be
# resubmitted without redoing the whole sweep -- e.g. ARRAY_RANGE="3"
# bash sbatch_openpi_libero_test.sh. Indices still index into the SAME
# full task_suites array above, unaffected by this override.
array_range="${ARRAY_RANGE:-0-${array_end}}"

gpu_type="${GPU_TYPE:-a40}"

sbatch_args=(
  --array="${array_range}" \
  --export="ALL,TASK_SUITE_LIST=${task_suite_list},VIDEO_DIR=${video_dir},${run_env}" \
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
echo "Submitted job array ${job_name} for ${#task_suites[@]} suites (${model_variant}) with ${run_env}"
echo "Video output dir: ${video_dir}"
