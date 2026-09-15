#!/usr/bin/bash

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RUN_SCRIPTS_DIR="${PROJECT_ROOT}/eval/run_scripts"
if [[ -f "${RUN_SCRIPTS_DIR}/.local_paths.sh" ]]; then
  # shellcheck disable=SC1091
  source "${RUN_SCRIPTS_DIR}/.local_paths.sh"
fi
cd "${PROJECT_ROOT}"

job_name="eval_groot_n1d6_libero_single_task"
output_dir="logs/eval/${job_name}"
# Matches RoboCasa's own eval/saved_eval_rollouts/<split>/<model> layout
# (see eval/launch_pretrain_split.sh) rather than a separate results/ tree
# -- same convention as sbatch_openpi_libero_test.sh/
# sbatch_rldx1_libero_test.sh.
base_video_dir="${VIDEO_DIR:-${BASE_VIDEO_DIR:-${PROJECT_ROOT}/eval/saved_eval_rollouts/libero/grootn16}}"

# Same 40 individual LIBERO tasks, n_episodes, and per-suite max_episode_steps
# as sbatch_rldx1_libero_test.sh -- GR00T-N1.6's own checkpoint
# (0xAnkitSingh/GR00T-N1.6-LIBERO) is COMBINED across all 4 suites, same
# convention as RLDX-1's, and its env id format (`libero_sim/<task>`) is
# identical (confirmed identical gr00t.eval.sim.LIBERO.libero_env.
# register_libero_envs implementation in both codebases -- see
# run_single_task_groot_n1d6_libero.py's own docstring), so the task list
# is transcribed verbatim rather than duplicated with any changes. Uniform
# 50 episodes/task and openpi's own per-suite max_steps (spatial=220,
# object=280, goal=300, libero_10=520) -- same "all 3 LIBERO models share
# one consistent per-suite horizon convention" policy -- see
# eval/EVAL_PROTOCOL_NOTES.md.
task_names=(
  "libero_sim/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket"
  "libero_sim/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket"
  "libero_sim/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"
  "libero_sim/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it"
  "libero_sim/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate"
  "libero_sim/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy"
  "libero_sim/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate"
  "libero_sim/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket"
  "libero_sim/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"
  "libero_sim/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it"
  "libero_sim/open_the_middle_drawer_of_the_cabinet"
  "libero_sim/put_the_bowl_on_the_stove"
  "libero_sim/put_the_wine_bottle_on_top_of_the_cabinet"
  "libero_sim/open_the_top_drawer_and_put_the_bowl_inside"
  "libero_sim/put_the_bowl_on_top_of_the_cabinet"
  "libero_sim/push_the_plate_to_the_front_of_the_stove"
  "libero_sim/put_the_cream_cheese_in_the_bowl"
  "libero_sim/turn_on_the_stove"
  "libero_sim/put_the_bowl_on_the_plate"
  "libero_sim/put_the_wine_bottle_on_the_rack"
  "libero_sim/pick_up_the_alphabet_soup_and_place_it_in_the_basket"
  "libero_sim/pick_up_the_cream_cheese_and_place_it_in_the_basket"
  "libero_sim/pick_up_the_salad_dressing_and_place_it_in_the_basket"
  "libero_sim/pick_up_the_bbq_sauce_and_place_it_in_the_basket"
  "libero_sim/pick_up_the_ketchup_and_place_it_in_the_basket"
  "libero_sim/pick_up_the_tomato_sauce_and_place_it_in_the_basket"
  "libero_sim/pick_up_the_butter_and_place_it_in_the_basket"
  "libero_sim/pick_up_the_milk_and_place_it_in_the_basket"
  "libero_sim/pick_up_the_chocolate_pudding_and_place_it_in_the_basket"
  "libero_sim/pick_up_the_orange_juice_and_place_it_in_the_basket"
  "libero_sim/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"
  "libero_sim/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate"
  "libero_sim/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate"
  "libero_sim/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate"
  "libero_sim/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate"
  "libero_sim/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate"
  "libero_sim/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate"
  "libero_sim/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate"
  "libero_sim/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate"
  "libero_sim/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate"
)
n_episodes_list=(
  50 50 50 50 50 50 50 50 50 50
  50 50 50 50 50 50 50 50 50 50
  50 50 50 50 50 50 50 50 50 50
  50 50 50 50 50 50 50 50 50 50
)
# task_names above is grouped in blocks of 10 by suite, in this exact
# order (matching RLDX-1's own eval_libero.sh ALL_TASKS loop order):
# libero_10, libero_goal, libero_object, libero_spatial.
max_episode_steps_list=(
  520 520 520 520 520 520 520 520 520 520
  300 300 300 300 300 300 300 300 300 300
  280 280 280 280 280 280 280 280 280 280
  220 220 220 220 220 220 220 220 220 220
)

model_path="${MODEL_PATH:-0xAnkitSingh/GR00T-N1.6-LIBERO}"
video_dir="${VIDEO_OUTPUT_DIR:-${base_video_dir}}"
n_action_steps="${N_ACTION_STEPS:-8}"

run_env="MODEL_PATH=${model_path},N_ACTION_STEPS=${n_action_steps}"
if [[ -n "${SAVE_REPLAY:-}" ]]; then
  run_env="${run_env},SAVE_REPLAY=${SAVE_REPLAY}"
fi

mkdir -p "${output_dir}"
mkdir -p "${video_dir}"
task_list=$(IFS=:; echo "${task_names[*]}")
n_episodes_str=$(IFS=:; echo "${n_episodes_list[*]}")
max_episode_steps_str=$(IFS=:; echo "${max_episode_steps_list[*]}")
array_end=$(( ${#task_names[@]} - 1 ))
# Overridable so a subset of tasks (e.g. ones that failed on a shared-node
# issue) can be resubmitted without redoing the whole sweep -- e.g.
# ARRAY_RANGE="0,1,7,8" bash sbatch_groot_n1d6_libero_test.sh. Indices still
# index into the SAME full task_names/n_episodes_list/max_episode_steps_list
# arrays above, unaffected by this override.
array_range="${ARRAY_RANGE:-0-${array_end}}"

gpu_type="${GPU_TYPE:-a40}"

sbatch_args=(
  --array="${array_range}" \
  --export="ALL,TASK_LIST=${task_list},N_EPISODES_LIST=${n_episodes_str},MAX_EPISODE_STEPS_LIST=${max_episode_steps_str},VIDEO_DIR=${video_dir},${run_env}" \
  --job-name="${job_name}" \
  --gpus-per-node="${gpu_type}:1" \
  --output="${output_dir}/${job_name}-slurm-%A_%a.out" \
  --error="${output_dir}/${job_name}-slurm-%A_%a.err"
)
if [[ -n "${SLURM_PARTITION:-}" ]]; then
  sbatch_args+=(--partition="${SLURM_PARTITION}")
fi
if [[ -n "${SLURM_EXCLUDE_NODES:-}" ]]; then
  sbatch_args+=(--exclude="${SLURM_EXCLUDE_NODES}")
fi

sbatch "${sbatch_args[@]}" "eval/run_scripts/${job_name}.sh"
echo "Submitted job array ${job_name} for ${#task_names[@]} tasks with ${run_env}"
echo "Video output dir: ${video_dir}"
