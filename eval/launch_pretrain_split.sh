#!/usr/bin/bash
# Launches the 5 models that belong to RoboCasa's "Multitask Learning"
# benchmarking setting (https://robocasa.ai/docs/build/html/benchmarking/
# multitask_learning.html: "We evaluate the model in pretrain scenes."),
# as opposed to the 3 Isaac-GR00T "Foundation Model Learning" families
# (target_posttraining/target_only/pretraining), which stay on split=target
# via launch_groot.sh -- see eval/EVAL_PROTOCOL_NOTES.md for the full
# reasoning behind this split.
#
# All 5 write under eval/saved_eval_rollouts/pretrain/<model>/... (as
# opposed to the target-split runs under .../target/<model>/...).
#
# Run from the repo root: bash eval/launch_pretrain_split.sh

set -euo pipefail
ROLLOUTS="${ROLLOUTS:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/saved_eval_rollouts}"

SPLIT=pretrain \
MODEL_FAMILY=multitask_learning \
N_EPISODES=50 \
VIDEO_DIR="${ROLLOUTS}/pretrain/groot" \
bash eval/run_scripts/sbatch_groot_test.sh

SPLIT=pretrain \
OPENPI_MODEL_VARIANT=pi0 \
OPENPI_MODEL_FAMILY=pretraining \
N_EPISODES=50 \
VIDEO_DIR="${ROLLOUTS}/pretrain/openpi" \
bash eval/run_scripts/sbatch_openpi_test.sh

SPLIT=pretrain \
OPENPI_MODEL_VARIANT=pi0.5 \
OPENPI_MODEL_FAMILY=pretraining \
N_EPISODES=50 \
VIDEO_DIR="${ROLLOUTS}/pretrain/openpi" \
bash eval/run_scripts/sbatch_openpi_test.sh

SPLIT=pretrain \
N_EPISODES=50 \
N_ACTION_STEPS=8 \
SAVE_REPLAY=1 \
VIDEO_DIR="${ROLLOUTS}/pretrain/grootn16" \
bash eval/run_scripts/sbatch_grootn16_test.sh

SPLIT=pretrain \
N_EPISODES=50 \
N_ACTION_STEPS=8 \
SAVE_REPLAY=1 \
VIDEO_DIR="${ROLLOUTS}/pretrain/rldx1" \
bash eval/run_scripts/sbatch_rldx1_test.sh
