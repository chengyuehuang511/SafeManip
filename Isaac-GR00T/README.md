## NVIDIA Isaac GR00T
This is the NVIDIA Isaac GR00T fork repo for running RoboCasa benchmark experiments. This fork is based on the original [GR00T code](https://github.com/NVIDIA/Isaac-GR00T) from NVIDIA. Our fork supports training for **GR00T N1.5**.

### Recommended system specs
For training we recommend a GPU with at least 80 Gb of memory (H100, H200, etc).
For inference we recommend a GPU with at least 8 Gb of memory.


### Installation
```
git clone https://github.com/robocasa-benchmark/Isaac-GR00T
cd groot
pip install -e .
```

### Key files
- Training: [scripts/gr00t_finetune.py](https://github.com/robocasa-benchmark/Isaac-GR00T/blob/main/scripts/gr00t_finetune.py)
- Evaluation: [scripts/run_eval.py](https://github.com/robocasa-benchmark/Isaac-GR00T/blob/main/scripts/run_eval.py)

### Experiment workflow
```
# train model
python scripts/gr00t_finetune.py \
--output-dir <experiment-path> \
--dataset_soup <dataset-soup> \
--max_steps <num-training-steps>

# evaluate model
python scripts/run_eval.py \
--model_path <checkpoint-path> \
--task_set <task-set> \
--split <split>

# report evaluation results
python gr00t/eval/get_eval_stats.py \
--dir <checkpoint-path>
```
