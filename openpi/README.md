# openpi
This is the openpi fork repo for running RoboCasa benchmark experiments. This fork is based on the original [openpi code](https://github.com/Physical-Intelligence/openpi) from the Physical Intelligence team.

## Recommended system specs
For training we recommend a GPU with at least 80 Gb of memory (H100, H200, etc).
For inference we recommend a GPU with at least 8 Gb of memory.


## Installation
```
git clone https://github.com/robocasa-benchmark/openpi
cd openpi
pip install -e .
pip install -e packages/openpi-client/
```

## Key files
- Training: [scripts/train.py](https://github.com/robocasa-benchmark/openpi/blob/main/scripts/train.py)
- Evaluation: [scripts/serve_policy.py](https://github.com/robocasa-benchmark/openpi/blob/main/scripts/serve_policy.py) and [examples/robocasa/main.py](https://github.com/robocasa-benchmark/openpi/blob/main/examples/robocasa/main.py)
- Setting up configs: [src/openpi/training/config.py](https://github.com/robocasa-benchmark/openpi/blob/main/src/openpi/training/config.py)

## Experiment workflow
```
# train model
XLA_PYTHON_CLIENT_MEM_FRACTION=1.0 python scripts/train.py \
<dataset-soup> \
--exp-name=<exp-name>

# evaluate model
# part a: start inference server
python scripts/serve_policy.py \
--port=8000 policy:checkpoint \
--policy.config=<dataset-soup> \
--policy.dir=<checkpoint-path>

# part b: run evals on server
python examples/robocasa/main.py \
--args.port 8000 \
--args.task_set <task-set> \
--args.split <split> \
--args.log_dir <checkpoint-path>

# report evaluation results
python examples/robocasa/get_eval_stats.py \
--dir <checkpoint-path>
```
