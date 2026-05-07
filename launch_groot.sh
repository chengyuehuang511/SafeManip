MODEL_FAMILY=target_posttraining \
N_EPISODES=50 \
bash run_scripts/sbatch_groot_test.sh

MODEL_FAMILY=target_only \
N_EPISODES=50 \
bash run_scripts/sbatch_groot_test.sh

MODEL_FAMILY=pretraining \
N_EPISODES=50 \
bash run_scripts/sbatch_groot_test.sh

MODEL_FAMILY=multitask_learning \
N_EPISODES=50 \
bash run_scripts/sbatch_groot_test.sh
