#!/usr/bin/bash
#SBATCH --job-name=eval_groot_n1d6_libero_single_task
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=16
#SBATCH --gpus-per-node="a40:1"
#SBATCH --qos="short"
#SBATCH --mem-per-gpu=45G
#SBATCH --time=24:00:00
#SBATCH --array=0

set -euo pipefail

# Runs eval/single_task/run_single_task_groot_n1d6_libero.py against the
# PRISTINE eval/models/Isaac-GR00T_official_n1d6 submodule (official
# NVIDIA/Isaac-GR00T, branch n1d6 -- N1.6), for the COMBINED (single
# checkpoint, all 4 suites) GR00T-N1.6 LIBERO checkpoint
# 0xAnkitSingh/GR00T-N1.6-LIBERO -- NOT the grootn16 fork, per "LIBERO
# always uses official" policy (see eval/EVAL_PROTOCOL_NOTES.md), even
# though the fork has its own working LIBERO support too.
#
# Uses a dedicated conda env (groot-libero, cloned from the shared
# "robocasa" env then robosuite downgraded to 1.4.0) -- same
# robosuite==1.5.2 vs 1.4.0 conflict as openpi/RLDX-1's LIBERO envs.

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
GROOT_N1D6_ROOT="${PROJECT_ROOT}/eval/models/Isaac-GR00T_official_n1d6"
LIBERO_ROOT="${LIBERO_ROOT:-${PROJECT_ROOT}/eval/simulators/libero}"
SINGLE_TASK_DIR="${PROJECT_ROOT}/eval/single_task"
cd "${GROOT_N1D6_ROOT}"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-groot-libero}"

if [[ -f "${CONDA_SH}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
else
  echo "Missing conda initialization script: ${CONDA_SH}" >&2
  exit 1
fi

conda activate "${CONDA_ENV_NAME}"

# See _setup_libero_config.sh's own docstring: LIBERO's get_libero_path()
# reads bddl_files/init_states/etc. from a machine-wide config file
# entirely independent of PYTHONPATH -- this generates one pointing at our
# actual LIBERO_ROOT instead of trusting whatever ~/.libero/config.yaml
# happens to already contain on this machine.
# shellcheck disable=SC1091
source "${RUN_SCRIPTS_DIR}/_setup_libero_config.sh"

# GROOT_N1D6_ROOT first (so `import gr00t` resolves to this pristine
# official submodule, not grootn16/eval/models/Isaac-GR00T or their
# editable-installed copies in the conda env's site-packages), then
# LIBERO_ROOT (shared eval/simulators/libero, reused the same way RLDX-1's
# LIBERO eval does -- confirmed content-identical to RLDX-1's own vendored
# copy; n1d6's own external_dependencies/LIBERO is left uninitialized),
# then SINGLE_TASK_DIR.
export PYTHONPATH="${GROOT_N1D6_ROOT}:${LIBERO_ROOT}:${SINGLE_TASK_DIR}:${PYTHONPATH:-}"
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
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-}"
MAX_EPISODE_STEPS_LIST="${MAX_EPISODE_STEPS_LIST:-}"
if [[ -z "${TASK}" && -n "${TASK_LIST}" ]]; then
  IFS=':' read -r -a TASK_LIST_ITEMS <<< "${TASK_LIST}"
  IFS=':' read -r -a N_EPISODES_LIST_ITEMS <<< "${N_EPISODES_LIST}"
  IFS=':' read -r -a MAX_EPISODE_STEPS_LIST_ITEMS <<< "${MAX_EPISODE_STEPS_LIST}"
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
  if [[ -n "${MAX_EPISODE_STEPS_LIST}" ]]; then
    MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS_LIST_ITEMS[SLURM_ARRAY_TASK_ID]}"
  fi
fi
TASK="${TASK:-pick_up_the_alphabet_soup_and_place_it_in_the_basket}"
N_EPISODES="${N_EPISODES:-50}"
MODEL_PATH="${MODEL_PATH:-0xAnkitSingh/GR00T-N1.6-LIBERO}"
TASK_CLEAN="${TASK#libero_sim/}"
VIDEO_DIR="${VIDEO_DIR:-${GROOT_N1D6_ROOT}/videos_libero}/${TASK_CLEAN}"
N_ACTION_STEPS="${N_ACTION_STEPS:-8}"
# 720 is only a fallback for single-task manual invocation -- the sweep
# launcher passes MAX_EPISODE_STEPS_LIST with openpi's per-suite max_steps
# values (220/280/300/520 for spatial/object/goal/10) instead, matching
# RLDX-1's LIBERO launcher -- see eval/EVAL_PROTOCOL_NOTES.md.
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-720}"
# Off by default, same convention as eval_rldx1_libero_single_task.sh --
# opt in via SAVE_REPLAY=1. Produces LIBERO's own native demo.hdf5 format.
SAVE_REPLAY="${SAVE_REPLAY:-0}"
REPLAY_DIR="${REPLAY_DIR:-${VIDEO_DIR}/replay}"

echo "Hostname: $(hostname)"
echo "MODEL_PATH=${MODEL_PATH}"
echo "TASK=${TASK}"
echo "VIDEO_DIR=${VIDEO_DIR}"
echo "N_EPISODES=${N_EPISODES}"
echo "N_ACTION_STEPS=${N_ACTION_STEPS}"
echo "MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS}"
echo "SAVE_REPLAY=${SAVE_REPLAY}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L || true
fi

EXTRA_ARGS=()
if [[ "${SAVE_REPLAY}" == "1" ]]; then
  EXTRA_ARGS+=(--save_replay --replay_dir "${REPLAY_DIR}")
fi

srun_status=0
srun python "${SINGLE_TASK_DIR}/run_single_task_groot_n1d6_libero.py" \
  --model_path "${MODEL_PATH}" \
  --task "${TASK_CLEAN}" \
  --video_dir "${VIDEO_DIR}" \
  --n_episodes "${N_EPISODES}" \
  --n_action_steps "${N_ACTION_STEPS}" \
  --max_episode_steps "${MAX_EPISODE_STEPS}" \
  "${EXTRA_ARGS[@]}" || srun_status="$?"

exit "${srun_status}"
