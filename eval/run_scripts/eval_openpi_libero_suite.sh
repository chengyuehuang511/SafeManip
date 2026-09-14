#!/usr/bin/bash
#SBATCH --job-name=eval_openpi_libero_suite
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=16
#SBATCH --gpus-per-node="a40:1"
#SBATCH --qos="short"
#SBATCH --mem-per-gpu=45G
#SBATCH --time=24:00:00
#SBATCH --array=0

set -euo pipefail

# Runs eval/single_task/run_libero_suite_openpi.py against
# eval/models/openpi_official's own examples/libero/main.py, unmodified.
# Uses openpi_official (official Physical-Intelligence/openpi) rather than
# eval/models/openpi (the robocasa-benchmark fork used for RoboCasa) --
# confirmed byte-identical examples/libero/main.py and scripts/
# serve_policy.py between the two, but official's src/openpi/training/
# config.py doesn't have the robocasa-fork-only broken norm-stats fallback
# (see serve_policy_wrapper.py's fix (c)) at all, so this is a strictly
# cleaner reproduction for LIBERO specifically.
#
# Uses eval/simulators/libero_openpi, a DEDICATED, standalone submodule
# pinned at f78abd68ee283de9f9be3c8f7e2a9ad60246e95c (Dec 2023) to exactly
# match openpi's own third_party/libero pin -- NOT the shared
# eval/simulators/libero submodule (currently pinned Mar 2025). Unlike
# robocasa, where all models happened to pin compatible versions, LIBERO
# has a documented history of eval numbers not reproducing across versions
# (e.g. openvla's LIBERO results are known to shift with newer LIBERO
# releases), and openpi's own pin is ~15 months older than our shared
# submodule's. Deliberately NOT initializing openpi's own nested
# third_party/libero submodule in place (that would leave files checked
# out inside eval/models/openpi's own working tree, which we don't touch,
# same as grootn16/RLDX-1's own nested robocasa forks are left
# uninitialized) -- this is a separate clone of the identical pinned
# commit, registered as its own top-level submodule instead.
# RLDX-1's LIBERO eval, by contrast, DOES reuse the shared eval/simulators/
# libero submodule -- confirmed via `diff -rq` to be content-identical to
# RLDX-1's own vendored external_dependencies/LIBERO copy, so no version
# drift there. See eval/EVAL_PROTOCOL_NOTES.md.
#
# Hyperparameters (resize_size=224, replan_steps=5, num_steps_wait=10,
# num_trials_per_task=50, seed=7) are openpi's own official defaults from
# examples/libero/main.py's Args dataclass -- see eval/EVAL_PROTOCOL_NOTES.md.

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
export OPENPI_ROOT="${OPENPI_ROOT:-${PROJECT_ROOT}/eval/models/openpi_official}"
LIBERO_ROOT="${LIBERO_ROOT:-${PROJECT_ROOT}/eval/simulators/libero_openpi}"
SINGLE_TASK_DIR="${PROJECT_ROOT}/eval/single_task"
cd "${OPENPI_ROOT}"

OPENPI_CONDA_ENV_NAME="${OPENPI_CONDA_ENV_NAME:-${CONDA_ENV_NAME:-openpi-robocasa}}"

if [[ -f "${CONDA_SH}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
else
  echo "Missing conda initialization script: ${CONDA_SH}" >&2
  exit 1
fi

conda activate "${OPENPI_CONDA_ENV_NAME}"

export PYTHONPATH="${OPENPI_ROOT}:${OPENPI_ROOT}/src:${OPENPI_ROOT}/packages/openpi-client/src:${LIBERO_ROOT}:${SINGLE_TASK_DIR}:${PYTHONPATH:-}"
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

OPENPI_MODEL_VARIANT="${OPENPI_MODEL_VARIANT:-pi0.5}"  # pi0 or pi0.5
if [[ "${OPENPI_MODEL_VARIANT}" == "pi0.5" ]]; then
  OPENPI_CONFIG="${OPENPI_CONFIG:-pi05_libero}"
  OPENPI_CHECKPOINT_DIR="${OPENPI_CHECKPOINT_DIR:-gs://openpi-assets/checkpoints/pi05_libero}"
elif [[ "${OPENPI_MODEL_VARIANT}" == "pi0" ]]; then
  OPENPI_CONFIG="${OPENPI_CONFIG:-pi0_libero}"
  OPENPI_CHECKPOINT_DIR="${OPENPI_CHECKPOINT_DIR:-gs://openpi-assets/checkpoints/pi0_libero}"
else
  echo "Unknown OPENPI_MODEL_VARIANT=${OPENPI_MODEL_VARIANT} (expected pi0 or pi0.5)" >&2
  exit 1
fi

VIDEO_DIR="${VIDEO_DIR:-${OPENPI_ROOT}/videos_libero}/${TASK_SUITE_NAME}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
REPLAN_STEPS="${REPLAN_STEPS:-5}"
RESIZE_SIZE="${RESIZE_SIZE:-224}"
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-10}"
SEED="${SEED:-7}"

if [[ -z "${PORT:-}" ]]; then
  PORT="$((8000 + (${SLURM_JOB_ID:-0} % 20000)))"
fi

echo "Hostname: $(hostname)"
echo "OPENPI_MODEL_VARIANT=${OPENPI_MODEL_VARIANT}"
echo "OPENPI_CONFIG=${OPENPI_CONFIG}"
echo "OPENPI_CHECKPOINT_DIR=${OPENPI_CHECKPOINT_DIR}"
echo "TASK_SUITE_NAME=${TASK_SUITE_NAME}"
echo "VIDEO_DIR=${VIDEO_DIR}"
echo "PORT=${PORT}"
echo "NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK}"
echo "REPLAN_STEPS=${REPLAN_STEPS}"
echo "RESIZE_SIZE=${RESIZE_SIZE}"
echo "NUM_STEPS_WAIT=${NUM_STEPS_WAIT}"
echo "SEED=${SEED}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L || true
fi

mkdir -p "${VIDEO_DIR}"

server_log="${VIDEO_DIR}/server-${SLURM_JOB_ID:-local}-${SLURM_ARRAY_TASK_ID:-0}.log"
python "${SINGLE_TASK_DIR}/serve_policy_wrapper.py" \
  --port "${PORT}" \
  policy:checkpoint \
  --policy.config "${OPENPI_CONFIG}" \
  --policy.dir "${OPENPI_CHECKPOINT_DIR}" \
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
  if grep -q "Creating server" "${server_log}" 2>/dev/null; then
    echo "Server ready after ${i}s."
    break
  fi
  sleep 5
done
sleep 5  # give the socket a moment past the log line, matching other launchers' convention

srun_status=0
python "${SINGLE_TASK_DIR}/run_libero_suite_openpi.py" \
  --task_suite_name "${TASK_SUITE_NAME}" \
  --host localhost \
  --port "${PORT}" \
  --num_trials_per_task "${NUM_TRIALS_PER_TASK}" \
  --replan_steps "${REPLAN_STEPS}" \
  --resize_size "${RESIZE_SIZE}" \
  --num_steps_wait "${NUM_STEPS_WAIT}" \
  --seed "${SEED}" \
  --video_out_path "${VIDEO_DIR}" || srun_status="$?"

exit "${srun_status}"
