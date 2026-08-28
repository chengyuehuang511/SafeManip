#!/usr/bin/bash
# One SLURM array task = one task's episodes, privileged-info-extracted +
# monitor-evaluated via extract_privileged_from_dataset_sampled.py (sample
# every N raw frames, unscaled thresholds -- see that script's module
# docstring). Submitted as a job array (one array index per task) by
# submit_extract_privileged_sampled.py. Mirrors
# run_extract_privileged_from_dataset.sbatch (the "scaled" method's sbatch)
# exactly except for which script it calls and the default output dir --
# kept as a fully separate file so neither script/sbatch pair risks the
# other's already-validated behavior.
#SBATCH --job-name=extract_privileged_sampled
#SBATCH --nodes=1
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=45G
#SBATCH --qos=debug
#SBATCH --time=02:00:00
#SBATCH --output=/nethome/chuang475/testnvme/projects/SafeManip/SafeManip/monitor/logs/%x-%A_%a.out

set -uo pipefail  # not -e: one failed episode inside the script shouldn't kill the whole job

SCRIPT_DIR="/nethome/chuang475/testnvme/projects/SafeManip/SafeManip/monitor"
TASKS_FILE="${TASKS_FILE:?set TASKS_FILE=<path to newline-separated task list> (submit_extract_privileged_sampled.sh does this)}"
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
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/output_sampled}"
SAMPLE_STRIDE="${SAMPLE_STRIDE:-16}"

cd /tmp

echo "=== starting ${TASK} (array index ${SLURM_ARRAY_TASK_ID}) at $(date) ==="
python3 "${SCRIPT_DIR}/extract_privileged_from_dataset_sampled.py" \
  --task "${TASK}" \
  --dataset_root "${DATASET_ROOT}" \
  --output_root "${OUTPUT_ROOT}" \
  --n_episodes "${N_EPISODES}" \
  --sample_stride "${SAMPLE_STRIDE}" \
  --run_monitor \
  --skip_existing
echo "=== finished ${TASK} at $(date) ==="
