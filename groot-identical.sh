#!/usr/bin/bash
#SBATCH --job-name=groot_identical
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=16
#SBATCH --gpus-per-node="l40s:1"
#SBATCH --mem-per-gpu=45G
#SBATCH --time=36:00:00

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Fixed-scene ("identical episode") GR00T evaluation launcher.
#
# Runs N_EPISODES rollouts on the SAME RoboCasa scene and the SAME initial state
# (seed-locked). Only the policy's own stochasticity varies between episodes.
# It is the standard single-task GR00T eval except it calls
# scripts/run_single_task_identical.py (which forces a fixed reset seed on every
# episode) and writes to its own results/groot_identical directory.
#
# The scene is fully determined by SEED, so changing SEED selects a different
# (but still single, repeated) scene. The varied-scene pipeline
# (run_scripts/sbatch_groot_test.sh) is untouched.
#
# Validate that the produced episodes really share one scene with:
#   python SafeManip/validate_identical_scene.py \
#     results/groot_identical/evals/<split>/<TASK>/rollout_data/<TASK>--*/
# ─────────────────────────────────────────────────────────────────────────────

# Resolve the repository root whether run directly or via sbatch.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
fi
RUN_SCRIPTS_DIR="${PROJECT_ROOT}/run_scripts"
# Machine-specific paths (CONDA_SH, GROOT_CHECKPOINT_ROOT, ...) live here. See README.
if [[ -f "${RUN_SCRIPTS_DIR}/.local_paths.sh" ]]; then
  # shellcheck disable=SC1091
  source "${RUN_SCRIPTS_DIR}/.local_paths.sh"
fi
GROOT_ROOT="${PROJECT_ROOT}/Isaac-GR00T"

# ── USER CONFIG (override via environment) ──────────────────────────────────
TASK="${TASK:-PackIdenticalLunches}"
SPLIT="${SPLIT:-target}"
N_EPISODES="${N_EPISODES:-50}"
SEED="${SEED:-42}"
# Provide MODEL_PATH directly, or let it resolve under a foundation-model root.
MODEL_FAMILY="${MODEL_FAMILY:-target_posttraining}"
FOUNDATION_MODEL_ROOT="${FOUNDATION_MODEL_ROOT:-${GROOT_CHECKPOINT_ROOT:-}/foundation_model_learning}"
MODEL_PATH="${MODEL_PATH:-${FOUNDATION_MODEL_ROOT}/${MODEL_FAMILY}/composite_seen/checkpoint-60000}"
VIDEO_DIR="${VIDEO_DIR:-${PROJECT_ROOT}/results/groot_identical/${MODEL_FAMILY}}"
# ────────────────────────────────────────────────────────────────────────────

CONDA_ENV_NAME="${CONDA_ENV_NAME:-robocasa}"
if [[ -z "${CONDA_SH:-}" || ! -f "${CONDA_SH}" ]]; then
  echo "CONDA_SH is not set or does not exist. Set it in run_scripts/.local_paths.sh" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${CONDA_ENV_NAME}"

cd "${GROOT_ROOT}"

export SCENEFLOW_ROOT="${SCENEFLOW_ROOT:-${PROJECT_ROOT}}"
export PYTHONPATH="${GROOT_ROOT}:${PYTHONPATH:-}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/${USER}/triton-${SLURM_JOB_ID:-local}}"
mkdir -p "${TRITON_CACHE_DIR}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/${USER}/matplotlib-${SLURM_JOB_ID:-local}}"
mkdir -p "${MPLCONFIGDIR}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-${CUDA_VISIBLE_DEVICES%%,*}}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export TOKENIZERS_PARALLELISM=false

export GR00T_POLICY_COMPUTE_DTYPE="${GR00T_POLICY_COMPUTE_DTYPE:-bfloat16}"
export GR00T_POLICY_DISABLE_AUTOCAST="${GR00T_POLICY_DISABLE_AUTOCAST:-0}"
export GR00T_DISABLE_FLASH_ATTN="${GR00T_DISABLE_FLASH_ATTN:-0}"
export GR00T_PER_ACTION_SEED="${GR00T_PER_ACTION_SEED:-0}"
export PYTHONHASHSEED="${SEED}"
export PYTHONNOUSERSITE=1

PORT="${PORT:-$((5555 + (${SLURM_JOB_ID:-0} % 20000)))}"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH does not exist or is not a directory: ${MODEL_PATH}" >&2
  echo "Set MODEL_PATH directly, or GROOT_CHECKPOINT_ROOT in run_scripts/.local_paths.sh" >&2
  exit 1
fi

echo "MODE=fixed-scene (all episodes share one scene + initial state)"
echo "TASK=${TASK}  SPLIT=${SPLIT}  N_EPISODES=${N_EPISODES}  SEED=${SEED}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "VIDEO_DIR=${VIDEO_DIR}  PORT=${PORT}"

mkdir -p "${VIDEO_DIR}"

python scripts/run_single_task_identical.py \
  --model_path "${MODEL_PATH}" \
  --task "${TASK}" \
  --split "${SPLIT}" \
  --video_dir "${VIDEO_DIR}" \
  --port "${PORT}" \
  --seed "${SEED}" \
  --n_episodes "${N_EPISODES}" \
  --save_privileged_info
