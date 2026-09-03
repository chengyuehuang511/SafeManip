#!/usr/bin/bash
# One SLURM array task = exactly one (task, episode) pair, privileged-info-
# extracted + monitor-evaluated from the official RoboCasa lerobot training
# dataset. Finer-grained parallelization than run_extract_privileged_from_
# dataset.sbatch's one-array-task-per-task (which runs N_EPISODES episodes
# sequentially inside each array task, bottlenecked by whichever task's
# episodes happen to be slowest) -- here every episode of every task gets
# its own array index, so the whole batch's wall-clock time is bounded by
# the single slowest *episode*, not the slowest *task's sum of episodes*.
# Submitted by submit_extract_privileged_per_episode.sh.
#SBATCH --job-name=extract_privileged_per_episode
#SBATCH --nodes=1
# No GPU requested -- see run_extract_privileged_from_dataset.sbatch's own
# comment for why (pure physics-state replay, no rendering, ever).
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --qos=debug
#SBATCH --time=04:00:00
#SBATCH --output=/nethome/chuang475/testnvme/projects/SafeManip/SafeManip/monitor/logs/%x-%A_%a.out

set -uo pipefail  # not -e: don't want one weird exit code tearing down output flush

# NOT $(dirname "${BASH_SOURCE[0]}") -- sbatch copies this script into a spool
# dir before running it.
SCRIPT_DIR="/nethome/chuang475/testnvme/projects/SafeManip/SafeManip/monitor"
PAIRS_FILE="${PAIRS_FILE:?set PAIRS_FILE=<path to newline-separated 'task episode' pairs> (submit_extract_privileged_per_episode.sh does this)}"
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

if [[ -n "${MUJOCO_GL:-}" ]]; then
  export MUJOCO_GL
fi
if [[ -n "${MUJOCO_EGL_DEVICE_ID:-}" ]]; then
  export MUJOCO_EGL_DEVICE_ID
fi

mkdir -p "${SCRIPT_DIR}/logs"

DATASET_ROOT="${DATASET_ROOT:-$HOME/flash/datasets/robocasa/v1.0/target}"
# NOT /tmp -- SLURM jobs get their own private /tmp, invisible to any other
# process including this same session's own shell (confirmed the hard way:
# 3 episodes' worth of real compute silently lost to this exact mistake in
# an earlier round). Always a genuinely shared, NFS-backed path.
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/output}"
CALL_STRIDE="${CALL_STRIDE:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

EXTRA_ARGS=()
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  EXTRA_ARGS+=(--skip_existing)
fi

cd /tmp

echo "=== starting ${TASK} ep${EPISODE} (array index ${SLURM_ARRAY_TASK_ID}) at $(date) ==="
python3 "${SCRIPT_DIR}/extract_privileged_from_dataset.py" \
  --task "${TASK}" \
  --episode "${EPISODE}" \
  --dataset_root "${DATASET_ROOT}" \
  --output_root "${OUTPUT_ROOT}" \
  --call_stride "${CALL_STRIDE}" \
  --run_monitor \
  "${EXTRA_ARGS[@]}"
echo "=== finished ${TASK} ep${EPISODE} at $(date) ==="
