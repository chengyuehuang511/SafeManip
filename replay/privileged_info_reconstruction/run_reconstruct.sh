#!/usr/bin/bash
# Reconstruct + compare one episode's video on a GPU compute node.
#
# Usage:
#   TASK=ArrangeBreadBasket EPISODE=0 sbatch run_reconstruct.sh
# or run synchronously on an interactively-allocated node:
#   TASK=ArrangeBreadBasket EPISODE=0 bash run_reconstruct.sh
#
# Env vars:
#   TASK              (required) task name, e.g. ArrangeBreadBasket
#   EPISODE           (required) episode index, e.g. 0
#   RESULTS_ROOT       default: the target_posttraining/evals/target dir this
#                       repo's viewer defaults to
#   CONDA_ENV_NAME     default: robocasa
#   CONDA_SH           default: ~/testnvme/miniconda3/etc/profile.d/conda.sh
#SBATCH --job-name=reconstruct_episode
#SBATCH --nodes=1
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=45G
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --output=/tmp/%u/reconstruct-%j.log

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

TASK="${TASK:?set TASK=<TaskName>}"
EPISODE="${EPISODE:?set EPISODE=<index>}"
RESULTS_ROOT="${RESULTS_ROOT:-/nethome/chuang475/testnvme/projects/SafeManip/results/evals/all_tasks_3_ckpt_50_rollouts/target_posttraining/evals/target}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-robocasa}"
CONDA_SH="${CONDA_SH:-${HOME}/testnvme/miniconda3/etc/profile.d/conda.sh}"

# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${CONDA_ENV_NAME}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"

# Find the latest rollout_data/<task>--<timestamp> dir (same rule the viewer uses).
ROLLOUT_DIR=$(find "${RESULTS_ROOT}/${TASK}/rollout_data" -maxdepth 1 -mindepth 1 -type d \
  -name "${TASK}--*" | sort | tail -n1)
if [[ -z "${ROLLOUT_DIR}" ]]; then
  echo "No rollout_data dir found for task ${TASK} under ${RESULTS_ROOT}" >&2
  exit 1
fi

PRIVILEGED_INFO="${ROLLOUT_DIR}/privileged_information_${EPISODE}.json"
ORIGINAL_VIDEO=$(find "${ROLLOUT_DIR}" -maxdepth 1 -name "*--episode=${EPISODE}--success=*--task=task.mp4")
OUT_DIR="${SCRIPT_DIR}/output/${TASK}"
mkdir -p "${OUT_DIR}"
RECON_VIDEO="${OUT_DIR}/episode_${EPISODE}_reconstructed.mp4"
COMPARISON_JSON="${OUT_DIR}/episode_${EPISODE}_comparison.json"

echo "task=${TASK} episode=${EPISODE}"
echo "privileged_info=${PRIVILEGED_INFO}"
echo "original_video=${ORIGINAL_VIDEO}"
echo "-> ${RECON_VIDEO}"

# cd somewhere that doesn't shadow `robocasa`/`gr00t` (see reconstruct_video.py docstring)
cd /tmp

python3 "${SCRIPT_DIR}/reconstruct_video.py" \
  --privileged_info "${PRIVILEGED_INFO}" \
  --output "${RECON_VIDEO}"

if [[ -n "${ORIGINAL_VIDEO}" ]]; then
  python3 "${SCRIPT_DIR}/compare_frames.py" \
    --original "${ORIGINAL_VIDEO}" \
    --reconstructed "${RECON_VIDEO}" \
    --privileged_info "${PRIVILEGED_INFO}" \
    --output "${COMPARISON_JSON}"
else
  echo "WARNING: no original video found, skipping comparison" >&2
fi
