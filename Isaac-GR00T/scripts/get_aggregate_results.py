import os

from termcolor import colored

from gr00t.eval.get_eval_stats import compute_stats

experiments = dict(
    atomic_10p="posttrain_atomic_seen_10p/try1/checkpoint-60000",
    atomic_10p_2stage="posttrain_atomic_seen_10p/2stage-try1/checkpoint-60000",
    atomic_30p="posttrain_atomic_seen_30p/try1/checkpoint-60000",
    atomic_30p_2stage="posttrain_atomic_seen_30p/2stage-try1/checkpoint-60000",
    atomic="posttrain_atomic_seen/freeze_vision_opencv_try2/checkpoint-60000",
    atomic_2stage="posttrain_atomic_seen/2stage-try1/checkpoint-60000",
    composite_seen_10p="posttrain_composite_seen_10p/try1/checkpoint-60000",
    composite_seen_10p_2stage="posttrain_composite_seen_10p/2stage-try1/checkpoint-60000",
    composite_seen_30p="posttrain_composite_seen_30p/try1/checkpoint-60000",
    composite_seen_30p_2stage="posttrain_composite_seen_30p/2stage-try1/checkpoint-60000",
    composite_seen="posttrain_composite_seen/freeze_vision_opencv_try1/checkpoint-60000",
    composite_seen_2stage="posttrain_composite_seen/2stage-try1/checkpoint-60000",
    composite_unseen_10p="posttrain_composite_unseen_10p/try1/checkpoint-60000",
    composite_unseen_10p_2stage="posttrain_composite_unseen_10p/2stage-try1/checkpoint-60000",
    composite_unseen_30p="posttrain_composite_unseen_30p/try1/checkpoint-60000",
    composite_unseen_30p_2stage="posttrain_composite_unseen_30p/2stage-try1/checkpoint-60000",
    composite_unseen="posttrain_composite_unseen/freeze_vision_opencv_try1/checkpoint-60000",
    composite_unseen_2stage="posttrain_composite_unseen/2stage-try1/checkpoint-60000",
)

BASE_EXP_DIR = "/scratch/10301/snasiriany/groot_expdata"

for exp_name, exp_path in experiments.items():
    print(colored(exp_name, "green"))
    compute_stats(os.path.join(BASE_EXP_DIR, exp_path))
