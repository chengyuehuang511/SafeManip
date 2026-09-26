#!/usr/bin/bash
#SBATCH --job-name=eval_openvla_libero_suite
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=16
#SBATCH --gpus-per-node="a40:1"
#SBATCH --qos="short"
#SBATCH --mem-per-gpu=45G
#SBATCH --time=24:00:00
#SBATCH --array=0

set -euo pipefail

# Runs eval/single_task/run_libero_suite_openvla.py against the PRISTINE
# eval/models/openvla submodule's own
# experiments/robot/libero/run_libero_eval.py, unmodified -- calls its
# `eval_libero(cfg)` directly (which loops over every task in the suite,
# `num_trials_per_task` episodes each), same pattern as
# eval_groot_n1d5_libero_suite.sh/eval_openpi_libero_suite.sh.
#
# Unlike openpi/GR00T-N1.5, OpenVLA needs NO running server -- it loads
# the HF checkpoint directly in-process, same as RLDX-1/GR00T-N1.6's
# in-process policy. No server-startup block here.
#
# Per-suite checkpoints (one official HF checkpoint per suite):
#   openvla/openvla-7b-finetuned-libero-{spatial,object,goal,10}
# (no libero_90 checkpoint released -- 4-suite scope, matching the other
# 3 LIBERO launchers' scope in this project).
#
# Uses the shared eval/simulators/libero submodule (same as RLDX-1/
# GR00T-N1.6/GR00T-N1.5's LIBERO eval) via _setup_libero_config.sh's
# "shared" config -- NOT the openvla conda env's own editable-installed
# `libero` package, which is confirmed to point at a stale/modified
# checkout (/path/to/LIBERO, the exact source of
# an earlier 4% RLDX-1 bug) -- PYTHONPATH
# prepending the shared submodule here is confirmed (via a real import
# test) to correctly override that stale egg-link.
#
# No max_steps override needed: OpenVLA's own hardcoded per-suite
# max_steps already exactly match openpi's official convention. No
# action-chunking patch needed either: OpenVLA is explicitly not trained
# with action chunking (queries the model every single env step by
# design, per its own README) -- n_action_steps=1 is this model's own
# genuine convention, not a gap.

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
OPENVLA_ROOT="${OPENVLA_ROOT:-${PROJECT_ROOT}/eval/models/openvla}"
LIBERO_ROOT="${LIBERO_ROOT:-${PROJECT_ROOT}/eval/simulators/libero}"
SINGLE_TASK_DIR="${PROJECT_ROOT}/eval/single_task"
cd "${OPENVLA_ROOT}"

# Pre-provisioned dedicated env matching OpenVLA's own hard pins
# (torch==2.5.1 [close to its own torch>=2.2.0/pyproject torch==2.2.0],
# transformers==4.40.1 exact, robosuite==1.4.1 exact) -- confirmed
# incompatible with openpi-libero/rldx1-libero/groot-libero's much newer
# transformers versions (OpenVLA registers custom AutoConfig/
# AutoModelForVision2Seq classes that are brittle across transformers
# versions).
CONDA_ENV_NAME="${CONDA_ENV_NAME:-openvla}"

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

export PYTHONPATH="${OPENVLA_ROOT}:${LIBERO_ROOT}:${SINGLE_TASK_DIR}:${PYTHONPATH:-}"
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

# Per-suite official checkpoint -- overridable via PRETRAINED_CHECKPOINT
# for manual runs (e.g. a locally fine-tuned checkpoint).
case "${TASK_SUITE_NAME}" in
  libero_spatial) DEFAULT_CHECKPOINT="openvla/openvla-7b-finetuned-libero-spatial" ;;
  libero_object)  DEFAULT_CHECKPOINT="openvla/openvla-7b-finetuned-libero-object" ;;
  libero_goal)    DEFAULT_CHECKPOINT="openvla/openvla-7b-finetuned-libero-goal" ;;
  libero_10)      DEFAULT_CHECKPOINT="openvla/openvla-7b-finetuned-libero-10" ;;
  *)
    echo "Unknown TASK_SUITE_NAME=${TASK_SUITE_NAME} (no official OpenVLA checkpoint for this suite)" >&2
    exit 1
    ;;
esac
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-${DEFAULT_CHECKPOINT}}"

VIDEO_DIR="${VIDEO_DIR:-${OPENVLA_ROOT}/videos_libero}/${TASK_SUITE_NAME}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-10}"
SEED="${SEED:-7}"
# Off by default, same convention as the other LIBERO launchers -- opt in
# via SAVE_REPLAY=1.
SAVE_REPLAY="${SAVE_REPLAY:-0}"
REPLAY_DIR="${REPLAY_DIR:-${VIDEO_DIR}/replay}"

echo "Hostname: $(hostname)"
echo "PRETRAINED_CHECKPOINT=${PRETRAINED_CHECKPOINT}"
echo "TASK_SUITE_NAME=${TASK_SUITE_NAME}"
echo "VIDEO_DIR=${VIDEO_DIR}"
echo "NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK}"
echo "NUM_STEPS_WAIT=${NUM_STEPS_WAIT}"
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
python "${SINGLE_TASK_DIR}/run_libero_suite_openvla.py" \
  --task_suite_name "${TASK_SUITE_NAME}" \
  --pretrained_checkpoint "${PRETRAINED_CHECKPOINT}" \
  --num_trials_per_task "${NUM_TRIALS_PER_TASK}" \
  --num_steps_wait "${NUM_STEPS_WAIT}" \
  --seed "${SEED}" \
  --video_out_path "${VIDEO_DIR}" \
  "${EXTRA_ARGS[@]}" || srun_status="$?"

exit "${srun_status}"
