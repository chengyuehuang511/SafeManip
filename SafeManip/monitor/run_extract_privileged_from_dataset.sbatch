#!/usr/bin/bash
# One SLURM array task = one task's episodes, privileged-info-extracted +
# monitor-evaluated from the official RoboCasa lerobot training dataset
# (extract_privileged_from_dataset.py) instead of a live rollout. Submitted
# as a job array (one array index per task) by submit_extract_privileged.sh.
# Mirrors replay/official_playback/run_reconstruct_training_data.sbatch.
#SBATCH --job-name=extract_privileged_from_dataset
#SBATCH --nodes=1
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=45G
#SBATCH --qos=debug
#SBATCH --time=04:00:00
#SBATCH --output=/nethome/chuang475/testnvme/projects/SafeManip/SafeManip/monitor/logs/%x-%A_%a.out

set -uo pipefail  # not -e: one failed episode inside the script shouldn't kill the whole job

# NOT $(dirname "${BASH_SOURCE[0]}") -- sbatch copies this script into a spool
# dir before running it (see replay/privileged_info_reconstruction/run_reconstruct_task.sbatch
# for the confirmed failure mode).
SCRIPT_DIR="/nethome/chuang475/testnvme/projects/SafeManip/SafeManip/monitor"
TASKS_FILE="${TASKS_FILE:?set TASKS_FILE=<path to newline-separated task list> (submit_extract_privileged.sh does this)}"
TASK=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${TASKS_FILE}")
if [[ -z "${TASK}" ]]; then
  echo "no task at array index ${SLURM_ARRAY_TASK_ID} in ${TASKS_FILE}" >&2
  exit 1
fi

CONDA_SH="${CONDA_SH:-${HOME}/testnvme/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-robocasa}"
# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${CONDA_ENV_NAME}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"

mkdir -p "${SCRIPT_DIR}/logs"

N_EPISODES="${N_EPISODES:-10}"
DATASET_ROOT="${DATASET_ROOT:-$HOME/flash/datasets/robocasa/v1.0/target}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/output}"
CALL_STRIDE="${CALL_STRIDE:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"  # set to 0 to force-overwrite (e.g. re-running with a corrected call_stride)

EXTRA_ARGS=()
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  EXTRA_ARGS+=(--skip_existing)
fi

# cd somewhere that doesn't shadow `robocasa`/`gr00t` (see reconstruct_video.py docstring)
cd /tmp

echo "=== starting ${TASK} (array index ${SLURM_ARRAY_TASK_ID}) at $(date) ==="
python3 "${SCRIPT_DIR}/extract_privileged_from_dataset.py" \
  --task "${TASK}" \
  --dataset_root "${DATASET_ROOT}" \
  --output_root "${OUTPUT_ROOT}" \
  --n_episodes "${N_EPISODES}" \
  --call_stride "${CALL_STRIDE}" \
  --run_monitor \
  "${EXTRA_ARGS[@]}"
echo "=== finished ${TASK} at $(date) ==="
