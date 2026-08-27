#!/usr/bin/bash
# One SLURM job = one task's full rollout (all episodes), reusing the built
# env across episodes (see reconstruct_task.py). Submitted once per task by
# submit_all.sh.
#SBATCH --job-name=reconstruct_task
#SBATCH --nodes=1
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=45G
#SBATCH --qos=debug
#SBATCH --time=02:00:00
#SBATCH --exclude=qt-1,starrysky,chappie,hk47,deebot,conroy,clippy,chitti
#SBATCH --output=/nethome/chuang475/testnvme/projects/SafeManip/replay/privileged_info_reconstruction/logs/%x-%A-%a.out

set -uo pipefail  # not -e: one failed episode inside reconstruct_task.py shouldn't kill the whole job

# NOT $(dirname "${BASH_SOURCE[0]}") -- sbatch copies this script into a spool
# dir (/var/lib/slurm/slurmd/job<N>/...) before running it, so BASH_SOURCE
# resolves to that temp copy's location, not this file's real directory
# (confirmed: caused "can't open file '.../job<N>/reconstruct_task.py'").
SCRIPT_DIR="/nethome/chuang475/testnvme/projects/SafeManip/replay/privileged_info_reconstruction"
TASK="${TASK:?set TASK=<TaskName> (submit_all.sh does this via --export)}"

CONDA_SH="${CONDA_SH:-${HOME}/testnvme/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-robocasa}"
# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${CONDA_ENV_NAME}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"

mkdir -p "${SCRIPT_DIR}/logs"

# cd somewhere that doesn't shadow `robocasa`/`gr00t` (see reconstruct_video.py docstring)
cd /tmp

# NOT --skip_existing: some tasks (ArrangeBreadBasket ep0-2, ArrangeTea
# ep40-49) had reconstructed.mp4/comparison.json on disk from earlier
# interactive development, predating later fixes (fixture-position
# override, sentinel-frame handling, final Newton calibration,
# data_integrity_suspect diagnostics). --skip_existing treats "file exists"
# as "file is current" and silently kept those stale, lower-quality
# artifacts during a full-batch rerun (confirmed: their task_summary.json
# entries are missing the data_integrity_suspect field entirely, i.e. they
# predate that field being added). Always regenerate everything so the
# whole batch is byte-for-byte the same code version.
echo "=== starting ${TASK} at $(date) ==="
python3 "${SCRIPT_DIR}/reconstruct_task.py" --task "${TASK}"
echo "=== finished ${TASK} at $(date) ==="
