#!/usr/bin/bash
#SBATCH --job-name=eval_rldx1_libero_single_task
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=16
#SBATCH --gpus-per-node="a40:1"
#SBATCH --qos="short"
#SBATCH --mem-per-gpu=45G
#SBATCH --time=24:00:00
#SBATCH --array=0

set -euo pipefail

# Runs eval/single_task/run_single_task_rldx1_libero.py against the
# PRISTINE eval/models/RLDX-1 submodule and eval/simulators/libero (the
# same shared LIBERO submodule reused for openpi's LIBERO eval too -- NOT
# RLDX-1's own external_dependencies/LIBERO, left uninitialized).
#
# Hyperparameters (n_action_steps=8, max_episode_steps=720, per-task
# n_episodes) match RLDX-1's own official run_scripts/eval/libero/
# eval_libero.sh exactly -- see eval/EVAL_PROTOCOL_NOTES.md.

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  if [[ "$(basename "${SLURM_SUBMIT_DIR}")" == "run_scripts" ]]; then
    PROJECT_ROOT=$(cd "${SLURM_SUBMIT_DIR}/../.." && pwd)
  else
    PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
  fi
else
  PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
fi
RUN_SCRIPTS_DIR="${PROJECT_ROOT}/eval/run_scripts"
if [[ -f "${RUN_SCRIPTS_DIR}/.local_paths.sh" ]]; then
  # shellcheck disable=SC1091
  source "${RUN_SCRIPTS_DIR}/.local_paths.sh"
fi
RLDX1_ROOT="${PROJECT_ROOT}/eval/models/RLDX-1"
LIBERO_ROOT="${LIBERO_ROOT:-${PROJECT_ROOT}/eval/simulators/libero}"
SINGLE_TASK_DIR="${PROJECT_ROOT}/eval/single_task"
cd "${RLDX1_ROOT}"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-rldx1}"

if [[ -f "${CONDA_SH}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
else
  echo "Missing conda initialization script: ${CONDA_SH}" >&2
  exit 1
fi

conda activate "${CONDA_ENV_NAME}"

export PYTHONPATH="${RLDX1_ROOT}:${LIBERO_ROOT}:${SINGLE_TASK_DIR}:${PYTHONPATH:-}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/${USER}/triton-${SLURM_JOB_ID:-local}}"
mkdir -p "${TRITON_CACHE_DIR}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
if [[ -z "${MUJOCO_EGL_DEVICE_ID:-}" ]]; then
  export MUJOCO_EGL_DEVICE_ID="${CUDA_VISIBLE_DEVICES:-0}"
  export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID%%,*}"
fi
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"

TASK="${TASK:-}"
TASK_LIST="${TASK_LIST:-}"
N_EPISODES="${N_EPISODES:-}"
N_EPISODES_LIST="${N_EPISODES_LIST:-}"
if [[ -z "${TASK}" && -n "${TASK_LIST}" ]]; then
  IFS=':' read -r -a TASK_LIST_ITEMS <<< "${TASK_LIST}"
  IFS=':' read -r -a N_EPISODES_LIST_ITEMS <<< "${N_EPISODES_LIST}"
  if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "TASK_LIST provided but SLURM_ARRAY_TASK_ID is missing." >&2
    exit 1
  fi
  if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#TASK_LIST_ITEMS[@]} )); then
    echo "Invalid SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
    exit 1
  fi
  TASK="${TASK_LIST_ITEMS[SLURM_ARRAY_TASK_ID]}"
  N_EPISODES="${N_EPISODES_LIST_ITEMS[SLURM_ARRAY_TASK_ID]}"
fi
TASK="${TASK:-libero_sim/pick_up_the_alphabet_soup_and_place_it_in_the_basket}"
# 50 default (not RLDX-1's own official per-suite 50/20/20/20 split) --
# deliberately overridden to a uniform 50 everywhere, see
# sbatch_rldx1_libero_test.sh's n_episodes_list.
N_EPISODES="${N_EPISODES:-50}"
MODEL_PATH="${MODEL_PATH:-RLWRLD/RLDX-1-FT-LIBERO}"
# Nested by TASK from the start -- see eval_grootn16_single_task.sh.
TASK_CLEAN="${TASK#libero_sim/}"
VIDEO_DIR="${VIDEO_DIR:-${RLDX1_ROOT}/videos_libero}/${TASK_CLEAN}"
N_ACTION_STEPS="${N_ACTION_STEPS:-8}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-720}"

echo "Hostname: $(hostname)"
echo "MODEL_PATH=${MODEL_PATH}"
echo "TASK=${TASK}"
echo "VIDEO_DIR=${VIDEO_DIR}"
echo "N_EPISODES=${N_EPISODES}"
echo "N_ACTION_STEPS=${N_ACTION_STEPS}"
echo "MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L || true
fi

srun_status=0
srun python "${SINGLE_TASK_DIR}/run_single_task_rldx1_libero.py" \
  --model_path "${MODEL_PATH}" \
  --task "${TASK}" \
  --video_dir "${VIDEO_DIR}" \
  --n_episodes "${N_EPISODES}" \
  --n_action_steps "${N_ACTION_STEPS}" \
  --max_episode_steps "${MAX_EPISODE_STEPS}" || srun_status="$?"

exit "${srun_status}"
