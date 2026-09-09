#!/usr/bin/env bash
# LIBERO analog of submit_extract_privileged_per_episode.sh: one SLURM array
# index per (task, episode) pair. Task list defaults to all 40 in-scope
# tasks (libero_10 + libero_goal + libero_object + libero_spatial, 10 tasks
# each), 10 episodes each = 400 array tasks -- LIBERO's task dirs are the 4
# suite directories directly (not robocasa's composite/atomic layout), so
# the task-name discovery glob differs from the robocasa version.
#
# Usage:
#   bash submit_extract_privileged_libero_per_episode.sh                 # all 40 tasks, episodes 0..N_EPISODES-1
#   bash submit_extract_privileged_libero_per_episode.sh TaskA TaskB      # just these tasks
#   N_EPISODES=5 bash submit_extract_privileged_libero_per_episode.sh    # override episode count
#   SUITES="libero_10 libero_goal" bash submit_extract_privileged_libero_per_episode.sh  # override suite scope
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# NOT /srv/datasets/libero -- login-node-only mount, invisible from compute
# nodes (confirmed via srun; see the sbatch script's own comment for the
# full story of how this caused a silent 100%-failure array job). Also used
# here on the login node itself for task-name discovery below, so it must be
# readable from wherever this submit script runs too -- ~/flash/datasets/
# is readable both places, confirmed.
DATASET_ROOT="${DATASET_ROOT:-${HOME}/flash/datasets/libero_raw}"
N_EPISODES="${N_EPISODES:-10}"
SUITES="${SUITES:-libero_10 libero_goal libero_object libero_spatial}"

if [[ $# -gt 0 ]]; then
  TASKS=("$@")
else
  mapfile -t TASKS < <(
    for suite in ${SUITES}; do
      find "${DATASET_ROOT}/${suite}" -maxdepth 1 -name '*_demo.hdf5' -printf '%f\n' 2>/dev/null \
        | sed 's/_demo\.hdf5$//'
    done | sort
  )
fi

# NOT /tmp: node-local, invisible to whichever compute node an array task
# actually lands on (and to this session's own shell).
mkdir -p "${SCRIPT_DIR}/logs"
PAIRS_FILE=$(mktemp "${SCRIPT_DIR}/logs/extract_privileged_libero_pairs.XXXXXX")
for task in "${TASKS[@]}"; do
  for ((ep = 0; ep < N_EPISODES; ep++)); do
    echo "${task} ${ep}" >> "${PAIRS_FILE}"
  done
done
N_PAIRS=$(wc -l < "${PAIRS_FILE}")

echo "submitting array job for ${N_PAIRS} (task, episode) pair(s) across ${#TASKS[@]} task(s) (pairs file: ${PAIRS_FILE})"
job_id=$(sbatch --parsable --array="0-$((N_PAIRS - 1))" \
  --export=ALL,PAIRS_FILE="${PAIRS_FILE}",DATASET_ROOT="${DATASET_ROOT}" \
  "${SCRIPT_DIR}/run_extract_privileged_from_dataset_libero_per_episode.sbatch")
echo "job ${job_id} (array 0-$((N_PAIRS - 1))) submitted"
echo "pairs file kept at ${PAIRS_FILE} (needed for the whole job's lifetime -- don't delete until it finishes)"
