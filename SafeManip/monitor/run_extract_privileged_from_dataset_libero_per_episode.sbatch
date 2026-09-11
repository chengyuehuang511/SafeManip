#!/usr/bin/bash
# LIBERO analog of run_extract_privileged_from_dataset_per_episode.sbatch:
# one SLURM array task = exactly one (task, episode) pair, privileged-info
# extracted from LIBERO's own official demo hdf5s (states-based replay) then
# monitor-evaluated, same finer-grained-than-per-task parallelization
# rationale as the robocasa version. Submitted by
# submit_extract_privileged_libero_per_episode.sh.
#
# Differs from the robocasa sbatch in two ways beyond the obvious
# extract_privileged_from_dataset_libero.py script swap:
#   - conda env for extraction is `safemanip_libero` (robosuite==1.4.0,
#     matching LIBERO's own pin), NOT `robocasa` (robosuite 1.5.2 -- API
#     incompatible with LIBERO's env classes, confirmed during this
#     integration's setup).
#   - LIBERO_CONFIG_PATH must point at this repo's own fresh LIBERO checkout
#     config (not the default ~/.libero/config.yaml, which may point at a
#     different, possibly user-modified LIBERO checkout).
# The monitor phase still runs via a `robocasa`-env subprocess bridge
# (extract_privileged_from_dataset_libero.py's own run_monitor_subprocess) --
# unrelated to this sbatch's own conda env, no change needed here for that.
#SBATCH --job-name=extract_privileged_libero_per_episode
#SBATCH --nodes=1
# No GPU requested -- pure physics-state replay, no rendering, ever (same as
# robocasa's own per-episode sbatch).
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --output=/nethome/chuang475/testnvme/projects/SafeManip/SafeManip/monitor/logs/%x-%A_%a.out

set -uo pipefail  # not -e: don't want one weird exit code tearing down output flush

# NOT $(dirname "${BASH_SOURCE[0]}") -- sbatch copies this script into a spool
# dir before running it.
SCRIPT_DIR="/nethome/chuang475/testnvme/projects/SafeManip/SafeManip/monitor"
PAIRS_FILE="${PAIRS_FILE:?set PAIRS_FILE=<path to newline-separated 'task episode' pairs> (submit_extract_privileged_libero_per_episode.sh does this)}"
PAIR_LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${PAIRS_FILE}")
if [[ -z "${PAIR_LINE}" ]]; then
  echo "no pair at array index ${SLURM_ARRAY_TASK_ID} in ${PAIRS_FILE}" >&2
  exit 1
fi
TASK="${PAIR_LINE%% *}"
EPISODE="${PAIR_LINE##* }"

CONDA_SH="${CONDA_SH:-${HOME}/testnvme/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-safemanip_libero}"
# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${CONDA_ENV_NAME}"

export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-/nethome/chuang475/testnvme/projects/SafeManip/eval/simulators/libero/.libero_config}"
export PYTHONPATH="${SCRIPT_DIR}:$(dirname "${SCRIPT_DIR}")${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${SCRIPT_DIR}/logs"

# NOT /srv/datasets/libero -- confirmed via `srun ... ls /srv/datasets/libero`
# on the actual compute nodes this array job lands on (t1000/vicki/neo):
# that mount is login-node-only (sky2.cc.gatech.edu), invisible from every
# compute node, which silently 404'd every single array task the first time
# this job ran (all "COMPLETED" per sacct despite 100% real failure -- see
# the exit-code fix below). ~/flash/datasets/ is the established
# cluster-wide-visible convention this repo already uses for RoboCasa's own
# dataset (~/flash/datasets/robocasa/v1.0/target) -- confirmed THIS specific
# mirror is visible from compute nodes too, the same way, before trusting it
# (`srun --nodelist=<a compute node> ls ~/flash/datasets/libero_raw/libero_10`).
DATASET_ROOT="${DATASET_ROOT:-${HOME}/flash/datasets/libero_raw}"
# NOT /tmp -- SLURM jobs get their own private /tmp, invisible to any other
# process including this same session's own shell (same lesson robocasa's
# own sbatch already learned the hard way).
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/output/v22_2026-09-08_libero_baseline}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

EXTRA_ARGS=()
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  EXTRA_ARGS+=(--skip_existing)
fi

cd /tmp

echo "=== starting ${TASK} ep${EPISODE} (array index ${SLURM_ARRAY_TASK_ID}) at $(date) ==="
python3 "${SCRIPT_DIR}/extract_privileged_from_dataset_libero.py" \
  --task "${TASK}" \
  --episode "${EPISODE}" \
  --dataset_root "${DATASET_ROOT}" \
  --output_root "${OUTPUT_ROOT}" \
  --run_monitor \
  "${EXTRA_ARGS[@]}"
# Explicit exit-code capture -- confirmed the hard way this job's first run
# silently masked a 100%-failure rate as "COMPLETED 0:0" in sacct: without
# this, bash's own exit status is that of the LAST command in the script
# (the trailing echo below), which always succeeds regardless of whether the
# python3 call above actually did anything. `set -uo pipefail` (this
# script's own shebang line) does NOT cover "last command in the script
# happens to be a successful echo" -- that's not a pipefail case, it's just
# normal sequential exit-status overwriting. Capture and propagate explicitly
# so a real failure shows up as SLURM/sacct FAILED, not a false COMPLETED.
PY_EXIT=$?
echo "=== finished ${TASK} ep${EPISODE} at $(date) (exit=${PY_EXIT}) ==="
exit "${PY_EXIT}"
