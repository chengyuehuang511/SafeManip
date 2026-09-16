#!/usr/bin/bash
#SBATCH --job-name=eval_cosmos_policy_robocasa_single_task
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=16
#SBATCH --gpus-per-node="a40:1"
#SBATCH --qos="short"
#SBATCH --mem-per-gpu=45G
#SBATCH --time=24:00:00
#SBATCH --array=0

set -euo pipefail

# Runs eval/single_task/run_single_task_cosmos_policy_robocasa.py against
# the PRISTINE eval/models/cosmos-policy submodule's own
# cosmos_policy/experiments/robot/robocasa/run_robocasa_eval.py, unmodified
# -- one task per invocation (`--task_name`), same pattern as
# eval_rldx1_single_task.sh/eval_grootn16_single_task.sh, but calling
# `eval_robocasa(cfg)` directly with a manually-constructed
# PolicyEvalConfig rather than going through a CLI, matching
# eval_cosmos_policy_libero_suite.sh's own approach.
#
# ORIGINAL (not RoboCasa365) 24-task RoboCasa benchmark -- this is not a
# switchable mode, it's the only one run_robocasa_eval.py supports (see
# run_single_task_cosmos_policy_robocasa.py's own docstring).
#
# Requires the eval/models/cosmos-policy-robocasa submodule
# (moojink/robocasa-cosmos-policy, cloned per cosmos-policy's own
# ROBOCASA.md instructions -- a SEPARATE fork from the shared
# eval/simulators/robocasa submodule other models use, since
# cosmos-policy's own ROBOCASA.md explicitly instructs cloning this one
# alongside it). Its own asset dirs (textures/objects/objaverse, and
# fixtures' binary mesh/texture files) are gitignored and NOT
# downloaded by default -- see ROBOCASA.md's own
# download_kitchen_assets.py step; this project instead symlinks them
# from the already-downloaded eval/simulators/robocasa copy (same public
# asset URLs, avoiding a redundant multi-GB re-download) and merges any
# missing fixture files via rsync --ignore-existing (preserving this
# fork's own tracked registry/config files).
#
# No max_steps override needed (matches this project's existing RoboCasa
# per-task TASK_MAX_STEPS convention already). Uses the CHECKPOINT's own
# native chunk_size=32/num_open_loop_steps=16 defaults (asserted to match
# the loaded checkpoint's own training config inside eval_robocasa()
# itself) -- notably DIFFERENT from LIBERO's chunk_size=16, a real
# per-benchmark hyperparameter difference for this model (not a bug).

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
COSMOS_POLICY_ROOT="${COSMOS_POLICY_ROOT:-${PROJECT_ROOT}/eval/models/cosmos-policy}"
COSMOS_POLICY_ROBOCASA_ROOT="${COSMOS_POLICY_ROBOCASA_ROOT:-${PROJECT_ROOT}/eval/models/cosmos-policy-robocasa}"
SINGLE_TASK_DIR="${PROJECT_ROOT}/eval/single_task"
cd "${COSMOS_POLICY_ROOT}"

# Dedicated env, SEPARATE from cosmos-policy's own LIBERO env (robosuite/
# mujoco version conflict -- see run_single_task_cosmos_policy_robocasa.py's
# own docstring). Cloned from the working `cosmos-policy` env (keeps its
# already-fixed torch/transformer_engine/flash_attn/protobuf combo) with
# robosuite==1.5.1/mujoco==3.2.6 swapped in for robocasa instead of
# libero's robosuite==1.4.1/mujoco==3.3.2.
CONDA_ENV_NAME="${CONDA_ENV_NAME:-cosmos-policy-robocasa}"

if [[ -f "${CONDA_SH}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
else
  echo "Missing conda initialization script: ${CONDA_SH}" >&2
  exit 1
fi

# See eval_cosmos_policy_libero_suite.sh's own comment: the
# gxx_linux-64/binutils_linux-64 conda-forge packages installed to build
# transformer_engine ship an activate.d hook incompatible with `set -u`.
set +u
conda activate "${CONDA_ENV_NAME}"
set -u

export PYTHONPATH="${COSMOS_POLICY_ROOT}:${COSMOS_POLICY_ROBOCASA_ROOT}:${SINGLE_TASK_DIR}:${PYTHONPATH:-}"

# transformer_engine's raw dlopen of pip-bundled cuda libs needs these on
# LD_LIBRARY_PATH explicitly -- see eval_cosmos_policy_libero_suite.sh's
# own comment for the full explanation.
LD_LIBRARY_PATH_ADD=$(python3 -c "
import glob, os
base = os.path.join(os.environ['CONDA_PREFIX'], 'lib', 'python3.10', 'site-packages', 'nvidia')
dirs = sorted(glob.glob(os.path.join(base, '*', 'lib')))
print(':'.join(dirs))
")
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH_ADD}:${LD_LIBRARY_PATH:-}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
if [[ -z "${MUJOCO_EGL_DEVICE_ID:-}" ]]; then
  export MUJOCO_EGL_DEVICE_ID="${CUDA_VISIBLE_DEVICES:-0}"
  export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID%%,*}"
fi
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"

TASK_NAME="${TASK_NAME:-}"
TASK_LIST="${TASK_LIST:-}"
if [[ -z "${TASK_NAME}" && -n "${TASK_LIST}" ]]; then
  IFS=':' read -r -a TASK_LIST_ITEMS <<< "${TASK_LIST}"
  if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "TASK_LIST provided but SLURM_ARRAY_TASK_ID is missing." >&2
    exit 1
  fi
  if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#TASK_LIST_ITEMS[@]} )); then
    echo "Invalid SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
    exit 1
  fi
  TASK_NAME="${TASK_LIST_ITEMS[SLURM_ARRAY_TASK_ID]}"
fi
TASK_NAME="${TASK_NAME:-PnPCounterToCab}"

CKPT_PATH="${CKPT_PATH:-nvidia/Cosmos-Policy-RoboCasa-Predict2-2B}"
CONFIG="${CONFIG:-cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference}"
CONFIG_FILE="${CONFIG_FILE:-cosmos_policy/config/config.py}"
DATASET_STATS_PATH="${DATASET_STATS_PATH:-${CKPT_PATH}/robocasa_dataset_statistics.json}"
T5_TEXT_EMBEDDINGS_PATH="${T5_TEXT_EMBEDDINGS_PATH:-${CKPT_PATH}/robocasa_t5_embeddings.pkl}"
CHUNK_SIZE="${CHUNK_SIZE:-32}"
NUM_OPEN_LOOP_STEPS="${NUM_OPEN_LOOP_STEPS:-16}"
NUM_DENOISING_STEPS_ACTION="${NUM_DENOISING_STEPS_ACTION:-5}"

VIDEO_DIR="${VIDEO_DIR:-${COSMOS_POLICY_ROOT}/videos_robocasa}/${TASK_NAME}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
SEED="${SEED:-7}"
SAVE_REPLAY="${SAVE_REPLAY:-0}"
REPLAY_DIR="${REPLAY_DIR:-${VIDEO_DIR}/replay}"

echo "Hostname: $(hostname)"
echo "CKPT_PATH=${CKPT_PATH}"
echo "TASK_NAME=${TASK_NAME}"
echo "VIDEO_DIR=${VIDEO_DIR}"
echo "NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK}"
echo "CHUNK_SIZE=${CHUNK_SIZE}"
echo "NUM_OPEN_LOOP_STEPS=${NUM_OPEN_LOOP_STEPS}"
echo "SEED=${SEED}"
echo "SAVE_REPLAY=${SAVE_REPLAY}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L || true
fi

mkdir -p "${VIDEO_DIR}"

EXTRA_ARGS=()
if [[ "${SAVE_REPLAY}" == "1" ]]; then
  EXTRA_ARGS+=(--save_replay --replay_dir "${REPLAY_DIR}")
fi

srun_status=0
python "${SINGLE_TASK_DIR}/run_single_task_cosmos_policy_robocasa.py" \
  --task_name "${TASK_NAME}" \
  --ckpt_path "${CKPT_PATH}" \
  --config "${CONFIG}" \
  --config_file "${CONFIG_FILE}" \
  --dataset_stats_path "${DATASET_STATS_PATH}" \
  --t5_text_embeddings_path "${T5_TEXT_EMBEDDINGS_PATH}" \
  --chunk_size "${CHUNK_SIZE}" \
  --num_open_loop_steps "${NUM_OPEN_LOOP_STEPS}" \
  --num_denoising_steps_action "${NUM_DENOISING_STEPS_ACTION}" \
  --num_trials_per_task "${NUM_TRIALS_PER_TASK}" \
  --seed "${SEED}" \
  --video_out_path "${VIDEO_DIR}" \
  "${EXTRA_ARGS[@]}" || srun_status="$?"

exit "${srun_status}"
