#!/usr/bin/env bash
# Submit one SLURM array job with one array index per (task, episode) pair
# needing its monitor step re-run -- the monitor-only analog of
# submit_extract_privileged_per_episode.sh. Only enqueues pairs whose
# _monitor.json is missing or older than predicates.py/specs.py (i.e. was
# last written before the current code), so re-running this after a small
# follow-up fix doesn't redo everything.
#
# Usage:
#   OUTPUT_ROOT=<vN dir> bash submit_rerun_monitor_only_per_episode.sh
#   OUTPUT_ROOT=<vN dir> bash submit_rerun_monitor_only_per_episode.sh TaskA TaskB
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT=<vN dir already containing privileged_information_<N>.json files>}"

NEWEST_SOURCE=$(stat -c '%Y' "${SCRIPT_DIR}/sim/robocasa/predicates.py" "${SCRIPT_DIR}/specs.py" | sort -n | tail -1)

if [[ $# -gt 0 ]]; then
  TASKS=("$@")
else
  mapfile -t TASKS < <(find "${OUTPUT_ROOT}" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | sort)
fi

mkdir -p "${SCRIPT_DIR}/logs"
PAIRS_FILE=$(mktemp "${SCRIPT_DIR}/logs/rerun_monitor_pairs.XXXXXX")
for task in "${TASKS[@]}"; do
  for priv_path in "${OUTPUT_ROOT}/${task}"/privileged_information_*.json; do
    [[ "${priv_path}" == *_monitor.json ]] && continue
    [[ -f "${priv_path}" ]] || continue
    ep=$(basename "${priv_path}" .json)
    ep="${ep#privileged_information_}"
    monitor_path="${OUTPUT_ROOT}/${task}/privileged_information_${ep}_monitor.json"
    if [[ ! -f "${monitor_path}" ]] || [[ "$(stat -c '%Y' "${monitor_path}")" -lt "${NEWEST_SOURCE}" ]]; then
      echo "${task} ${ep}" >> "${PAIRS_FILE}"
    fi
  done
done
N_PAIRS=$(wc -l < "${PAIRS_FILE}")

if [[ "${N_PAIRS}" -eq 0 ]]; then
  echo "nothing to do -- every _monitor.json under ${OUTPUT_ROOT} is already newer than predicates.py/specs.py"
  rm -f "${PAIRS_FILE}"
  exit 0
fi

echo "submitting array job for ${N_PAIRS} stale/missing (task, episode) pair(s) (pairs file: ${PAIRS_FILE})"
job_id=$(sbatch --parsable --array="0-$((N_PAIRS - 1))" \
  --export=ALL,PAIRS_FILE="${PAIRS_FILE}",OUTPUT_ROOT="${OUTPUT_ROOT}" \
  "${SCRIPT_DIR}/run_rerun_monitor_only_per_episode.sbatch")
echo "job ${job_id} (array 0-$((N_PAIRS - 1))) submitted"
echo "pairs file kept at ${PAIRS_FILE} (needed for the whole job's lifetime -- don't delete until it finishes)"
