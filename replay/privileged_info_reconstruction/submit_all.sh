#!/usr/bin/env bash
# Submit one sbatch job per task (each job reconstructs+compares every
# episode of that task's latest rollout). Usage:
#   bash submit_all.sh                      # all 50 tasks under RESULTS_ROOT
#   bash submit_all.sh TaskA TaskB           # just these tasks
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RESULTS_ROOT="${RESULTS_ROOT:-/nethome/chuang475/testnvme/projects/SafeManip/results/evals/all_tasks_3_ckpt_50_rollouts/target_posttraining/evals/target}"

if [[ $# -gt 0 ]]; then
  TASKS=("$@")
else
  mapfile -t TASKS < <(find "${RESULTS_ROOT}" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | sort)
fi

echo "submitting ${#TASKS[@]} task job(s)..."
for task in "${TASKS[@]}"; do
  job_id=$(sbatch --parsable --export=ALL,TASK="${task}" "${SCRIPT_DIR}/run_reconstruct_task.sbatch")
  echo "  ${task} -> job ${job_id}"
done
