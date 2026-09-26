#!/usr/bin/bash
#SBATCH --job-name=eval_rldx1_single_task
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=16
#SBATCH --gpus-per-node="a40:1"
#SBATCH --qos="short"
#SBATCH --mem-per-gpu=45G
#SBATCH --time=24:00:00
#SBATCH --array=0

set -euo pipefail

# Runs eval/single_task/run_single_task_rldx1.py against the PRISTINE
# eval/models/RLDX-1 submodule (RLWRLD/RLDX-1, robocasa-benchmark
# leaderboard's "RLDX-1" submission, commit
# ef05cd4ae634ff97d672d42275febbc0b92cc192) and eval/simulators/robocasa
# (pristine robocasa/robocasa) -- NOT RLDX-1's own external_dependencies/
# robocasa submodule, which is deliberately left uninitialized so this
# reuses the exact same robocasa checkout every other model in this repo
# uses. RLDX-1 needs a newer transformers (Qwen3-VL backbone,
# transformers.masking_utils) than the existing "robocasa" conda env has,
# so it gets its own dedicated "rldx1" env (torch 2.7.0, transformers
# 4.57.0, diffusers 0.35.1, peft 0.17.1, flash-attn 2.7.4.post1, matching
# eval/models/RLDX-1/pyproject.toml's pins) plus robosuite/robocasa/
# gymnasium/zmq/msgpack installed the same way as the "robocasa" env.

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
ROBOCASA_ROOT="${ROBOCASA_ROOT:-${PROJECT_ROOT}/eval/simulators/robocasa}"
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

# RLDX1_ROOT first (so `import rldx` resolves to this pristine submodule),
# then ROBOCASA_ROOT (pristine robocasa), then SINGLE_TASK_DIR (so
# run_single_task_rldx1.py can `import replay_capture`).
export PYTHONPATH="${RLDX1_ROOT}:${ROBOCASA_ROOT}:${SINGLE_TASK_DIR}:${PYTHONPATH:-}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/${USER}/triton-${SLURM_JOB_ID:-local}}"
mkdir -p "${TRITON_CACHE_DIR}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
if [[ -z "${MUJOCO_EGL_DEVICE_ID:-}" ]]; then
  # See eval_groot_single_task.sh / eval_openpi_single_task.sh for why this
  # defaults rather than reading CUDA_VISIBLE_DEVICES directly under set -u.
  export MUJOCO_EGL_DEVICE_ID="${CUDA_VISIBLE_DEVICES:-0}"
  export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID%%,*}"
fi
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"

TASK="${TASK:-}"
TASK_LIST="${TASK_LIST:-}"
if [[ -z "${TASK}" && -n "${TASK_LIST}" ]]; then
  IFS=':' read -r -a TASK_LIST_ITEMS <<< "${TASK_LIST}"
  if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "TASK_LIST provided but SLURM_ARRAY_TASK_ID is missing." >&2
    exit 1
  fi
  if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#TASK_LIST_ITEMS[@]} )); then
    echo "Invalid SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}; expected 0-$(( ${#TASK_LIST_ITEMS[@]} - 1 ))" >&2
    exit 1
  fi
  TASK="${TASK_LIST_ITEMS[SLURM_ARRAY_TASK_ID]}"
fi
TASK="${TASK:-PrepareCoffee}"
SPLIT="${SPLIT:-target}"
MODEL_PATH="${MODEL_PATH:-${HOME}/checkpoints/RLDX-1-FT-RC365}"
# Nested by TASK from the start -- see eval_grootn16_single_task.sh.
VIDEO_DIR="${VIDEO_DIR:-${RLDX1_ROOT}/videos_single_task}/${TASK}"
SEED="${SEED:-42}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-${SEED}}"
N_EPISODES="${N_EPISODES:-50}"
N_ACTION_STEPS="${N_ACTION_STEPS:-16}"
# Left empty by default -- run_single_task_rldx1.py falls back to
# robocasa's own per-task get_task_horizon(TASK) unless MAX_EPISODE_STEPS
# is explicitly set here.
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-}"
SAVE_REPLAY="${SAVE_REPLAY:-1}"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH does not exist or is not a directory: ${MODEL_PATH}" >&2
  exit 1
fi

echo "Hostname: $(hostname)"
echo "Working directory: ${RLDX1_ROOT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "MUJOCO_GL=${MUJOCO_GL}"
echo "MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "TASK=${TASK}"
echo "SPLIT=${SPLIT}"
echo "VIDEO_DIR=${VIDEO_DIR}"
echo "SEED=${SEED}"
echo "N_EPISODES=${N_EPISODES}"
echo "N_ACTION_STEPS=${N_ACTION_STEPS}"
echo "MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS}"
echo "SAVE_REPLAY=${SAVE_REPLAY}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L || true
fi

EXTRA_ARGS=()
if [[ "${SAVE_REPLAY}" == "1" ]]; then
  EXTRA_ARGS+=(--save_replay)
fi
if [[ -n "${MAX_EPISODE_STEPS}" ]]; then
  EXTRA_ARGS+=(--max_episode_steps "${MAX_EPISODE_STEPS}")
fi

srun_status=0
srun python "${SINGLE_TASK_DIR}/run_single_task_rldx1.py" \
  --model_path "${MODEL_PATH}" \
  --task "${TASK}" \
  --split "${SPLIT}" \
  --video_dir "${VIDEO_DIR}" \
  --seed "${SEED}" \
  --n_episodes "${N_EPISODES}" \
  --n_action_steps "${N_ACTION_STEPS}" \
  "${EXTRA_ARGS[@]}" || srun_status="$?"

exit "${srun_status}"
