#!/usr/bin/bash
# Replay episodes of the OFFICIAL RoboCasa training-data lerobot dataset,
# ground-truth-exact (no privileged-info reconstruction, no mobile-base
# calibration hack -- see README.md's discussion of reconstruct_video.py
# for why that path is lossy).
#
# Unlike reconstruct_video.py (which poses the sim from sparse per-monitor-
# frame object/robot poses recorded at *eval* time), this replays the
# *training* dataset's extras/episode_XXXXXX/{model.xml.gz,states.npz}:
# states.npz stores the full flattened MuJoCo sim state
# (`sim.get_state().flatten()`, shape (T, 256)) at every single timestep --
# literally the ground truth used to render the original training video.
# Loading it directly via env.sim.set_state_from_flattened + sim.forward()
# is exact, with no drift possible (see RoboCasa's own
# robocasa/scripts/dataset_scripts/playback_dataset.py, which this wraps).
#
# Usage:
#   DATASET=~/flash/datasets/robocasa/v1.0/target/composite/ArrangeBreadBasket/20250809/lerobot \
#     sbatch run_official_playback.sh
# or on an interactively-allocated GPU node:
#   DATASET=... bash run_official_playback.sh
#
# Env vars:
#   DATASET            (required) path to a lerobot-format dataset dir
#   N_EPISODES          default: 1 (number of episodes to play back; omit/empty = all)
#   USE_ACTIONS         default: 0 -- set to 1 to open-loop replay the
#                        recorded `action` column instead of loading states
#                        directly (prints per-step divergence vs. recorded
#                        states; useful to check action-replay determinism,
#                        but ground-truth state loading is the exact method)
#   VIDEO_PATH          default: <dataset_dir_parent>/<dataset_name>.mp4
#   CONDA_ENV_NAME      default: robocasa
#   CONDA_SH            default: ~/testnvme/miniconda3/etc/profile.d/conda.sh
#SBATCH --job-name=official_playback
#SBATCH --nodes=1
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=45G
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --output=/tmp/%u/official_playback-%j.log

set -euo pipefail

DATASET="${DATASET:?set DATASET=/path/to/lerobot/dataset}"
N_EPISODES="${N_EPISODES:-1}"
USE_ACTIONS="${USE_ACTIONS:-0}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-robocasa}"
CONDA_SH="${CONDA_SH:-${HOME}/testnvme/miniconda3/etc/profile.d/conda.sh}"
ROBOCASA_SCRIPTS_DIR="${ROBOCASA_SCRIPTS_DIR:-/coc/testnvme/chuang475/projects/robocasa/robocasa/scripts/dataset_scripts}"

# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${CONDA_ENV_NAME}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"

EXTRA_ARGS=()
if [[ -n "${N_EPISODES}" ]]; then
  EXTRA_ARGS+=(--n "${N_EPISODES}")
fi
if [[ "${USE_ACTIONS}" == "1" ]]; then
  EXTRA_ARGS+=(--use-actions)
fi
if [[ -n "${VIDEO_PATH:-}" ]]; then
  EXTRA_ARGS+=(--video_path "${VIDEO_PATH}")
fi

echo "dataset=${DATASET}"
echo "n_episodes=${N_EPISODES:-all} use_actions=${USE_ACTIONS}"

# cd somewhere that doesn't shadow `robocasa`/`gr00t` (see reconstruct_video.py docstring)
cd /tmp

python3 "${ROBOCASA_SCRIPTS_DIR}/playback_dataset.py" \
  --dataset "${DATASET}" \
  --render_image_names robot0_agentview_left robot0_agentview_right robot0_eye_in_hand \
  --verbose \
  "${EXTRA_ARGS[@]}"
