#!/usr/bin/bash
# One SLURM array task = exactly one (task, episode) pair, re-running JUST the
# monitor step (rerun_monitor_only.py's --task/--episode mode) against an
# already-extracted privileged_information_<N>.json, overwriting its
# _monitor.json in place. Same per-episode-parallelism pattern as
# run_extract_privileged_from_dataset_per_episode.sbatch, applied to the
# monitor-only rerun instead of full extraction -- confirmed 2026-09-03 that
# running this serially in one process (loading each 100-170MB privileged_
# information file one at a time) took 2.5+ hours for only 320/500 episodes.
# No simulation here (pure JSON + LTLfDFA evaluation), so this needs far less
# per-task compute than extraction -- kept the same conda env for safety but
# a much lighter resource request.
# Submitted by submit_rerun_monitor_only_per_episode.sh.
#SBATCH --job-name=rerun_monitor_per_episode
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --output=/nethome/chuang475/testnvme/projects/SafeManip/SafeManip/monitor/logs/%x-%A_%a.out

set -uo pipefail  # not -e: don't want one weird exit code tearing down output flush

# NOT $(dirname "${BASH_SOURCE[0]}") -- sbatch copies this script into a spool
# dir before running it.
SCRIPT_DIR="/nethome/chuang475/testnvme/projects/SafeManip/SafeManip/monitor"
PAIRS_FILE="${PAIRS_FILE:?set PAIRS_FILE=<path to newline-separated 'task episode' pairs> (submit_rerun_monitor_only_per_episode.sh does this)}"
PAIR_LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${PAIRS_FILE}")
if [[ -z "${PAIR_LINE}" ]]; then
  echo "no pair at array index ${SLURM_ARRAY_TASK_ID} in ${PAIRS_FILE}" >&2
  exit 1
fi
TASK="${PAIR_LINE%% *}"
EPISODE="${PAIR_LINE##* }"

CONDA_SH="${CONDA_SH:-${HOME}/testnvme/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-robocasa}"
# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${CONDA_ENV_NAME}"

mkdir -p "${SCRIPT_DIR}/logs"

# NOT /tmp -- see extraction sbatch's own comment for why (SLURM per-job /tmp
# isolation silently loses output written there).
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT=<vN dir already containing privileged_information_<N>.json files>}"

cd /tmp

echo "=== starting ${TASK} ep${EPISODE} (array index ${SLURM_ARRAY_TASK_ID}) at $(date) ==="
python3 "${SCRIPT_DIR}/rerun_monitor_only.py" \
  --output_root "${OUTPUT_ROOT}" \
  --task "${TASK}" \
  --episode "${EPISODE}"
echo "=== finished ${TASK} ep${EPISODE} at $(date) ==="
