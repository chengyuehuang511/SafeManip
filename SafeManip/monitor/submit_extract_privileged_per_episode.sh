#!/usr/bin/env bash
# Submit one SLURM array job with one array index per (task, episode) pair --
# finer-grained than submit_extract_privileged.sh's one-index-per-task
# (which serializes N_EPISODES episodes inside each array task). For a full
# 50-task x 10-episode sweep this submits 500 array tasks instead of 50, so
# the whole batch's wall-clock time is bounded by the single slowest
# episode, not the slowest task's sum of episodes.
#
# Usage:
#   bash submit_extract_privileged_per_episode.sh                 # all tasks, episodes 0..N_EPISODES-1
#   bash submit_extract_privileged_per_episode.sh TaskA TaskB      # just these tasks
#   N_EPISODES=5 bash submit_extract_privileged_per_episode.sh     # override episode count/task
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DATASET_ROOT="${DATASET_ROOT:-$HOME/flash/datasets/robocasa/v1.0/target}"
N_EPISODES="${N_EPISODES:-10}"

if [[ $# -gt 0 ]]; then
  TASKS=("$@")
else
  mapfile -t TASKS < <(
    { find "${DATASET_ROOT}/composite" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' 2>/dev/null
      find "${DATASET_ROOT}/atomic" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' 2>/dev/null
    } | sort
  )
fi

# NOT /tmp: node-local, invisible to whichever compute node an array task
# actually lands on (and to this session's own shell).
mkdir -p "${SCRIPT_DIR}/logs"
PAIRS_FILE=$(mktemp "${SCRIPT_DIR}/logs/extract_privileged_pairs.XXXXXX")
for task in "${TASKS[@]}"; do
  for ((ep = 0; ep < N_EPISODES; ep++)); do
    echo "${task} ${ep}" >> "${PAIRS_FILE}"
  done
done
N_PAIRS=$(wc -l < "${PAIRS_FILE}")

echo "submitting array job for ${N_PAIRS} (task, episode) pair(s) across ${#TASKS[@]} task(s) (pairs file: ${PAIRS_FILE})"
job_id=$(sbatch --parsable --array="0-$((N_PAIRS - 1))" \
  --export=ALL,PAIRS_FILE="${PAIRS_FILE}",DATASET_ROOT="${DATASET_ROOT}" \
  "${SCRIPT_DIR}/run_extract_privileged_from_dataset_per_episode.sbatch")
echo "job ${job_id} (array 0-$((N_PAIRS - 1))) submitted"
echo "pairs file kept at ${PAIRS_FILE} (needed for the whole job's lifetime -- don't delete until it finishes)"
