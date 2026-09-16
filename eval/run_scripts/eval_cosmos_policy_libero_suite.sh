#!/usr/bin/bash
#SBATCH --job-name=eval_cosmos_policy_libero_suite
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=16
#SBATCH --gpus-per-node="a40:1"
#SBATCH --qos="short"
#SBATCH --mem-per-gpu=45G
#SBATCH --time=24:00:00
#SBATCH --array=0

set -euo pipefail

# Runs eval/single_task/run_libero_suite_cosmos_policy.py against the
# PRISTINE eval/models/cosmos-policy submodule's own
# cosmos_policy/experiments/robot/libero/run_libero_eval.py, unmodified --
# same pattern as eval_openvla_libero_suite.sh/
# eval_groot_n1d5_libero_suite.sh.
#
# Single combined checkpoint (nvidia/Cosmos-Policy-LIBERO-Predict2-2B),
# trained jointly on all 4 suites -- task_suite_name only selects which
# suite to evaluate. No server needed (in-process checkpoint loading).
#
# Uses the shared eval/simulators/libero submodule (same as RLDX-1/
# GR00T-N1.6/GR00T-N1.5/OpenVLA's LIBERO eval) via _setup_libero_config.sh's
# "shared" config.
#
# No max_steps override needed (Cosmos Policy's own hardcoded per-suite
# max_steps already matches openpi's convention). No action-chunking patch
# needed either (chunk_size/num_open_loop_steps are genuine config fields,
# both default 16 per LIBERO.md's own documented example).

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
LIBERO_ROOT="${LIBERO_ROOT:-${PROJECT_ROOT}/eval/simulators/libero}"
SINGLE_TASK_DIR="${PROJECT_ROOT}/eval/single_task"
cd "${COSMOS_POLICY_ROOT}"

# Dedicated conda env (torch==2.7.0, transformers==4.57.1 -- matches
# cosmos-policy's own pyproject.toml pins) with libero-group deps
# (bddl/cloudpickle/draccus/easydict/gym/robosuite==1.4.1/mujoco==3.3.2)
# installed via pip (this project's conda-env convention, not cosmos-
# policy's own uv-managed venv -- and deliberately NOT installing the
# generic pip "libero" package, using the shared eval/simulators/libero
# submodule via PYTHONPATH instead, same reasoning as every other LIBERO
# launcher in this project).
CONDA_ENV_NAME="${CONDA_ENV_NAME:-cosmos-policy}"

if [[ -f "${CONDA_SH}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
else
  echo "Missing conda initialization script: ${CONDA_SH}" >&2
  exit 1
fi

# The gxx_linux-64/binutils_linux-64 conda-forge packages (installed into
# this env to get a C++20-capable compiler for building transformer_engine)
# ship an activate.d hook that references $ADDR2LINE without a default,
# which crashes under `set -u`. Relax nounset just for the activation
# call, matching the common workaround for this class of conda-activation-
# script bug.
set +u
conda activate "${CONDA_ENV_NAME}"
set -u

# shellcheck disable=SC1091
source "${RUN_SCRIPTS_DIR}/_setup_libero_config.sh"

export PYTHONPATH="${COSMOS_POLICY_ROOT}:${LIBERO_ROOT}:${SINGLE_TASK_DIR}:${PYTHONPATH:-}"

# transformer_engine (a megatron-core dependency, imported transitively via
# cosmos_utils.py -> model_loader.py -> imaginaire/config.py) does a raw
# ctypes.CDLL() dlopen of libcudnn_adv.so.9 etc. without setting rpath, so
# it needs these pip-installed nvidia-*-cuXX wheels' lib/ dirs on
# LD_LIBRARY_PATH explicitly (torch itself doesn't need this -- it bundles
# its own RPATH -- but transformer_engine's raw dlopen does). Confirmed via
# a real crash: `OSError: libcudnn_adv.so.9: cannot open shared object
# file`. Built generically from whatever nvidia-* wheels are actually
# installed in this conda env, rather than hardcoding specific package
# names/versions.
_NVIDIA_LIB_DIRS=$(python3 -c "
import glob, os, sysconfig
site_packages = sysconfig.get_paths()['purelib']
dirs = sorted(glob.glob(os.path.join(site_packages, 'nvidia', '*', 'lib')))
print(':'.join(dirs))
" 2>/dev/null || true)
if [[ -n "${_NVIDIA_LIB_DIRS}" ]]; then
  export LD_LIBRARY_PATH="${_NVIDIA_LIB_DIRS}:${LD_LIBRARY_PATH:-}"
fi

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

CKPT_PATH="${CKPT_PATH:-nvidia/Cosmos-Policy-LIBERO-Predict2-2B}"
CONFIG="${CONFIG:-cosmos_predict2_2b_480p_libero__inference_only}"
CONFIG_FILE="${CONFIG_FILE:-cosmos_policy/config/config.py}"
DATASET_STATS_PATH="${DATASET_STATS_PATH:-${CKPT_PATH}/libero_dataset_statistics.json}"
T5_TEXT_EMBEDDINGS_PATH="${T5_TEXT_EMBEDDINGS_PATH:-${CKPT_PATH}/libero_t5_embeddings.pkl}"
CHUNK_SIZE="${CHUNK_SIZE:-16}"
NUM_OPEN_LOOP_STEPS="${NUM_OPEN_LOOP_STEPS:-16}"
NUM_DENOISING_STEPS_ACTION="${NUM_DENOISING_STEPS_ACTION:-5}"

VIDEO_DIR="${VIDEO_DIR:-${COSMOS_POLICY_ROOT}/videos_libero}/${TASK_SUITE_NAME}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
SEED="${SEED:-7}"
# Off by default, same convention as the other LIBERO launchers -- opt in
# via SAVE_REPLAY=1.
SAVE_REPLAY="${SAVE_REPLAY:-0}"
REPLAY_DIR="${REPLAY_DIR:-${VIDEO_DIR}/replay}"

echo "Hostname: $(hostname)"
echo "CKPT_PATH=${CKPT_PATH}"
echo "TASK_SUITE_NAME=${TASK_SUITE_NAME}"
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
python "${SINGLE_TASK_DIR}/run_libero_suite_cosmos_policy.py" \
  --task_suite_name "${TASK_SUITE_NAME}" \
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
