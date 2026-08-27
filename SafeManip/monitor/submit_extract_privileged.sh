#!/usr/bin/env bash
# Submit one SLURM array job that extracts privileged-info + runs the
# symbolic monitor for N episodes (default 10) of every task (default: all
# under DATASET_ROOT's composite/ + atomic/ dirs) from the official RoboCasa
# lerobot training data, ground-truth-exact (see README.md /
# extract_privileged_from_dataset.py's module docstring). Mirrors
# replay/official_playback/submit_training_data.sh.
#
# Usage:
#   bash submit_extract_privileged.sh                 # all tasks found under DATASET_ROOT
#   bash submit_extract_privileged.sh TaskA TaskB      # just these tasks
#   N_EPISODES=5 bash submit_extract_privileged.sh     # override episodes/task
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
# job actually lands on.
mkdir -p "${SCRIPT_DIR}/logs"
TASKS_FILE=$(mktemp "${SCRIPT_DIR}/logs/extract_privileged_tasks.XXXXXX")
printf '%s\n' "${TASKS[@]}" > "${TASKS_FILE}"
N_TASKS=${#TASKS[@]}

echo "submitting array job for ${N_TASKS} task(s) (task list: ${TASKS_FILE})"
job_id=$(sbatch --parsable --array="0-$((N_TASKS - 1))" \
  --export=ALL,TASKS_FILE="${TASKS_FILE}",N_EPISODES="${N_EPISODES:-10}",DATASET_ROOT="${DATASET_ROOT}" \
  "${SCRIPT_DIR}/run_extract_privileged_from_dataset.sbatch")
echo "job ${job_id} (array 0-$((N_TASKS - 1))) submitted"
echo "task list kept at ${TASKS_FILE} (needed for the whole job's lifetime -- don't delete until it finishes)"
