#!/usr/bin/bash
#SBATCH --job-name=eval_groot_n1d5_libero_suite
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=16
#SBATCH --gpus-per-node="a40:1"
#SBATCH --qos="short"
#SBATCH --mem-per-gpu=45G
#SBATCH --time=24:00:00
#SBATCH --array=0

set -euo pipefail

# Runs eval/single_task/run_libero_suite_groot_n1d5.py against the PRISTINE
# eval/models/Isaac-GR00T_official_n1d5 submodule's own
# examples/Libero/eval/run_libero_eval.py, unmodified -- same
# "eval_libero(cfg)" call-directly pattern as eval_openpi_libero_suite.sh,
# but N1.5's own run_libero_eval.py is client/server-based (its
# `GR00TPolicy` connects to a running `scripts/inference_service.py
# --server` via `gr00t.eval.service.ExternalRobotInferenceClient`, the
# same server/client split openpi's serve_policy.py/main.py use), so this
# script also starts that server itself, mirroring
# eval_openpi_libero_suite.sh's own server-startup block.
#
# N1.5 trains one SEPARATE checkpoint per suite (unlike openpi/RLDX-1's
# single unified checkpoint, same as GR00T-N1.6's own convention) --
# per-suite checkpoint + --data_config selected below from NVIDIA's own
# examples/Libero/README.md table (libero_goal alone needs
# LiberoDataConfigMeanStd; every other suite uses LiberoDataConfig).
#
# Uses the shared eval/simulators/libero submodule (same as RLDX-1/
# GR00T-N1.6's LIBERO eval) via _setup_libero_config.sh's "shared" config
# -- N1.5 has no dedicated fork/pin of its own to match, unlike openpi's
# libero_openpi.

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
GROOT_N1D5_ROOT="${GROOT_N1D5_ROOT:-${PROJECT_ROOT}/eval/models/Isaac-GR00T_official_n1d5}"
LIBERO_ROOT="${LIBERO_ROOT:-${PROJECT_ROOT}/eval/simulators/libero}"
SINGLE_TASK_DIR="${PROJECT_ROOT}/eval/single_task"
cd "${GROOT_N1D5_ROOT}"

# Reuses the same groot-libero conda env as GR00T-N1.6's launcher (same
# torch/transformers/robosuite==1.4.0 stack) -- N1.5's own `gr00t` package
# resolves ahead of any pip-installed copy via PYTHONPATH ordering below,
# same mechanism eval_groot_n1d6_libero_single_task.sh already relies on.
CONDA_ENV_NAME="${CONDA_ENV_NAME:-groot-libero}"

if [[ -f "${CONDA_SH}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
else
  echo "Missing conda initialization script: ${CONDA_SH}" >&2
  exit 1
fi

conda activate "${CONDA_ENV_NAME}"

# shellcheck disable=SC1091
source "${RUN_SCRIPTS_DIR}/_setup_libero_config.sh"

export PYTHONPATH="${GROOT_N1D5_ROOT}:${LIBERO_ROOT}:${SINGLE_TASK_DIR}:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
if [[ -z "${MUJOCO_EGL_DEVICE_ID:-}" ]]; then
  export MUJOCO_EGL_DEVICE_ID="${CUDA_VISIBLE_DEVICES:-0}"
  export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID%%,*}"
fi
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"

TASK_SUITE_NAME="${TASK_SUITE_NAME:-}"
TASK_SUITE_LIST="${TASK_SUITE_LIST:-}"
if [[ -z "${TASK_SUITE_NAME}" && -n "${TASK_SUITE_LIST}" ]]; then
  IFS=':' read -r -a TASK_SUITE_LIST_ITEMS <<< "${TASK_SUITE_LIST}"
  if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "TASK_SUITE_LIST provided but SLURM_ARRAY_TASK_ID is missing." >&2
    exit 1
  fi
  if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#TASK_SUITE_LIST_ITEMS[@]} )); then
    echo "Invalid SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
    exit 1
  fi
  TASK_SUITE_NAME="${TASK_SUITE_LIST_ITEMS[SLURM_ARRAY_TASK_ID]}"
fi
TASK_SUITE_NAME="${TASK_SUITE_NAME:-libero_spatial}"

# Per-suite checkpoint + data_config, from NVIDIA's own
# examples/Libero/README.md table -- overridable via MODEL_PATH/DATA_CONFIG
# for manual runs (e.g. a locally fine-tuned checkpoint).
case "${TASK_SUITE_NAME}" in
  libero_spatial)
    DEFAULT_MODEL_PATH="youliangtan/gr00t-n1.5-libero-spatial-posttrain"
    DEFAULT_DATA_CONFIG="examples.Libero.custom_data_config:LiberoDataConfig"
    ;;
  libero_object)
    DEFAULT_MODEL_PATH="youliangtan/gr00t-n1.5-libero-object-posttrain"
    DEFAULT_DATA_CONFIG="examples.Libero.custom_data_config:LiberoDataConfig"
    ;;
  libero_goal)
    DEFAULT_MODEL_PATH="youliangtan/gr00t-n1.5-libero-goal-posttrain"
    # Only libero_goal's checkpoint was trained with the MeanStd data
    # config -- see README.md's own note right after the server command.
    DEFAULT_DATA_CONFIG="examples.Libero.custom_data_config:LiberoDataConfigMeanStd"
    ;;
  libero_90)
    DEFAULT_MODEL_PATH="youliangtan/gr00t-n1.5-libero-90-posttrain"
    DEFAULT_DATA_CONFIG="examples.Libero.custom_data_config:LiberoDataConfig"
    ;;
  libero_10)
    # README's table calls this suite "Long" (libero_10 == LIBERO-Long);
    # checkpoint name follows that "long" naming, not "10".
    DEFAULT_MODEL_PATH="youliangtan/gr00t-n1.5-libero-long-posttrain"
    DEFAULT_DATA_CONFIG="examples.Libero.custom_data_config:LiberoDataConfig"
    ;;
  *)
    echo "Unknown TASK_SUITE_NAME=${TASK_SUITE_NAME}" >&2
    exit 1
    ;;
esac
MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_PATH}}"
DATA_CONFIG="${DATA_CONFIG:-${DEFAULT_DATA_CONFIG}}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-new_embodiment}"
DENOISING_STEPS="${DENOISING_STEPS:-8}"

VIDEO_DIR="${VIDEO_DIR:-${GROOT_N1D5_ROOT}/videos_libero}/${TASK_SUITE_NAME}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-10}"
# Off by default, same convention as the other LIBERO launchers -- opt in
# via SAVE_REPLAY=1.
SAVE_REPLAY="${SAVE_REPLAY:-0}"
REPLAY_DIR="${REPLAY_DIR:-${VIDEO_DIR}/replay}"
# LIBERO-specific value, matching N1.6/RLDX-1's own official LIBERO
# convention (8) and the community LIBERO fine-tune's own documented eval
# convention (also 8) -- NOT the RoboCasa fork's generic default (16),
# which is a different benchmark's convention. See
# run_libero_suite_groot_n1d5.py's GR00T_N1D5_N_ACTION_STEPS comment for
# the full reasoning. N1.5's pristine run_libero_eval.py has no chunking
# at all (re-queries every env step); run_libero_suite_groot_n1d5.py
# patches this in via _patch_action_chunking.
N_ACTION_STEPS="${N_ACTION_STEPS:-8}"

if [[ -z "${PORT:-}" ]]; then
  PORT="$((8000 + (${SLURM_JOB_ID:-0} % 20000)))"
fi

echo "Hostname: $(hostname)"
echo "MODEL_PATH=${MODEL_PATH}"
echo "DATA_CONFIG=${DATA_CONFIG}"
echo "EMBODIMENT_TAG=${EMBODIMENT_TAG}"
echo "DENOISING_STEPS=${DENOISING_STEPS}"
echo "TASK_SUITE_NAME=${TASK_SUITE_NAME}"
echo "VIDEO_DIR=${VIDEO_DIR}"
echo "PORT=${PORT}"
echo "NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK}"
echo "NUM_STEPS_WAIT=${NUM_STEPS_WAIT}"
echo "N_ACTION_STEPS=${N_ACTION_STEPS}"
echo "SAVE_REPLAY=${SAVE_REPLAY}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L || true
fi

mkdir -p "${VIDEO_DIR}"

# Server startup, mirroring eval_openpi_libero_suite.sh's block exactly --
# ready-detection string taken from gr00t/eval/service.py's own
# `print(f"Server is ready and listening on {addr}")` (BaseInferenceServer,
# the same server class N1.6's create_gr00t_sim_policy also builds on, but
# N1.6 never runs it standalone like this since it calls the policy
# in-process instead of over ZMQ).
server_log="${VIDEO_DIR}/server-${SLURM_JOB_ID:-local}-${SLURM_ARRAY_TASK_ID:-0}.log"
python "${GROOT_N1D5_ROOT}/scripts/inference_service.py" \
  --server \
  --model_path "${MODEL_PATH}" \
  --data_config "${DATA_CONFIG}" \
  --embodiment_tag "${EMBODIMENT_TAG}" \
  --denoising_steps "${DENOISING_STEPS}" \
  --port "${PORT}" \
  > "${server_log}" 2>&1 &
server_pid=$!
trap 'kill "${server_pid}" 2>/dev/null || true' EXIT

echo "Waiting for policy server (pid=${server_pid}) to become ready..."
for i in $(seq 1 120); do
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    echo "Server process died before becoming ready; see ${server_log}" >&2
    cat "${server_log}" >&2 || true
    exit 1
  fi
  if grep -q "Server is ready and listening on" "${server_log}" 2>/dev/null; then
    echo "Server ready after ${i}s."
    break
  fi
  sleep 5
done
sleep 5  # give the socket a moment past the log line, matching other launchers' convention

EXTRA_ARGS=()
if [[ "${SAVE_REPLAY}" == "1" ]]; then
  EXTRA_ARGS+=(--save_replay --replay_dir "${REPLAY_DIR}")
fi

srun_status=0
python "${SINGLE_TASK_DIR}/run_libero_suite_groot_n1d5.py" \
  --task_suite_name "${TASK_SUITE_NAME}" \
  --host localhost \
  --port "${PORT}" \
  --num_trials_per_task "${NUM_TRIALS_PER_TASK}" \
  --num_steps_wait "${NUM_STEPS_WAIT}" \
  --n_action_steps "${N_ACTION_STEPS}" \
  --headless \
  --video_out_path "${VIDEO_DIR}" \
  "${EXTRA_ARGS[@]}" || srun_status="$?"

exit "${srun_status}"
