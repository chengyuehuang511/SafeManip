#!/usr/bin/env bash
# Submit one SLURM array job that reconstructs N episodes (default 10) of
# every task (default: all under DATASET_ROOT's composite/ + atomic/ dirs,
# 50 tasks total for the current target dataset) from the official RoboCasa
# lerobot training data, ground-truth-exact (see README.md).
#
# Usage:
#   bash submit_training_data.sh                 # all tasks found under DATASET_ROOT
#   bash submit_training_data.sh TaskA TaskB      # just these tasks
#   N_EPISODES=5 bash submit_training_data.sh     # override episodes/task
#   VIDEO_SKIP=1 bash submit_training_data.sh     # render every raw frame (native 20fps,
#                                                  # matches the original dataset video's
#                                                  # frame rate exactly) instead of the
#                                                  # default video_skip=2 (10fps)
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DATASET_ROOT="${DATASET_ROOT:-$HOME/flash/datasets/robocasa/v1.0/target}"

if [[ $# -gt 0 ]]; then
  TASKS=("$@")
else
  mapfile -t TASKS < <(
    { find "${DATASET_ROOT}/composite" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' 2>/dev/null
      find "${DATASET_ROOT}/atomic" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' 2>/dev/null
    } | sort
  )
fi

# NOT /tmp: that's node-local, invisible to whichever compute node the array
# job actually lands on (confirmed: "sed: can't read /tmp/..." on the worker).
mkdir -p "${SCRIPT_DIR}/logs"
TASKS_FILE=$(mktemp "${SCRIPT_DIR}/logs/training_data_tasks.XXXXXX")
printf '%s\n' "${TASKS[@]}" > "${TASKS_FILE}"
N_TASKS=${#TASKS[@]}

echo "submitting array job for ${N_TASKS} task(s) (task list: ${TASKS_FILE})"
job_id=$(sbatch --parsable --array="0-$((N_TASKS - 1))" \
  --export=ALL,TASKS_FILE="${TASKS_FILE}",N_EPISODES="${N_EPISODES:-10}",DATASET_ROOT="${DATASET_ROOT}",VIDEO_SKIP="${VIDEO_SKIP:-2}" \
  "${SCRIPT_DIR}/run_reconstruct_training_data.sbatch")
echo "job ${job_id} (array 0-$((N_TASKS - 1))) submitted"
echo "task list kept at ${TASKS_FILE} (needed for the whole job's lifetime -- don't delete until it finishes)"
