#!/usr/bin/bash
# One SLURM array task = one hyperparameter sweep cell (hp_sweep_grid.cells()),
# re-monitoring the whole episode subset under that cell's SAFEMANIP_HP_*
# overrides into its own output tree.
#
# WHY SLURM AND NOT JUST LOCALLY: an interactive login session on this cluster
# lives in a cgroup capped at 15 GB
# (/sys/fs/cgroup/memory/user.slice/user-<uid>.slice/memory.limit_in_bytes).
# RoboCasa privileged_information files in this subset are 16-556 MB of JSON and
# peak at roughly 12x that in RSS, so anything past ~8 concurrent episodes gets
# individual episodes SIGKILLed -- which shows up as "exit-9, no traceback" and
# looks exactly like a monitor crash while `free` still reports 375 GB available.
# Under SLURM the cgroup is whatever --mem asks for, so the sweep gets real room.
#
# No simulation happens here (pure JSON replay + LTLf DFA evaluation), so this is
# CPU/memory-bound, not GPU work.
#
# Submitted by submit_hp_sweep.sh.
#SBATCH --job-name=hp_sweep_cell
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --qos=debug
#SBATCH --time=04:00:00
#SBATCH --output=/path/to/SafeManip/SafeManip/monitor/logs/%x-%A_%a.out

set -uo pipefail  # not -e: a single failed episode must not tear down the cell

# NOT $(dirname "${BASH_SOURCE[0]}") -- sbatch copies this script into a spool dir.
SCRIPT_DIR="/path/to/SafeManip/SafeManip/monitor"
CELLS_FILE="${CELLS_FILE:?set CELLS_FILE=<newline-separated cell names> (submit_hp_sweep.sh does this)}"
CELL=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${CELLS_FILE}")
if [[ -z "${CELL}" ]]; then
  echo "no cell at array index ${SLURM_ARRAY_TASK_ID} in ${CELLS_FILE}" >&2
  exit 1
fi

CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
# robocasa, for BOTH suites: it is the only env here with ltlf2dfa + pydot
# installed, and the monitor step never touches a simulator, so the LIBERO
# episodes replay correctly under it too (verified 2026-09-22 against the
# committed v40 monitor output).
CONDA_ENV_NAME="${CONDA_ENV_NAME:-robocasa}"
# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${CONDA_ENV_NAME}"

mkdir -p "${SCRIPT_DIR}/logs"

# Keep the in-flight estimate comfortably under --mem: the gate's 12x RSS
# multiplier is an estimate, and being SIGKILLed costs a whole episode.
export HP_SWEEP_MEM_BUDGET_GB="${HP_SWEEP_MEM_BUDGET_GB:-44}"

# NOT /tmp -- SLURM gives each job a private /tmp, so anything written there is
# silently lost when the job ends (see the extraction sbatch's own comment).
cd /tmp

echo "=== cell ${CELL} (array index ${SLURM_ARRAY_TASK_ID}) starting at $(date) ==="
python3 "${SCRIPT_DIR}/hp_sweep_run.py" \
  --cell "${CELL}" \
  --workers "${SLURM_CPUS_PER_TASK:-8}" \
  --timeout 2400
echo "=== cell ${CELL} finished at $(date) ==="
