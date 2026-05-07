# SafeManip

This anonymized repository accompanies **SafeManip: A Property-Driven Benchmark for Temporal Safety Evaluation in Robotic Manipulation**.

SafeManip evaluates robotic manipulation policies with privileged simulator state and temporal safety specifications. The codebase provides RoboCasa instrumentation, symbolic predicates, LTL/DFA monitors, policy evaluation launchers, and analysis utilities for summarizing safety outcomes.

## Contents

```text
SafeManip/
|-- Isaac-GR00T/              # GR00T policy and evaluation integration
|-- openpi/                   # OpenPI policy and evaluation integration
|-- robocasa/                 # RoboCasa fork with privileged-state export
|-- SafeManip/
|   |-- monitor/              # Symbolic monitor, predicates, LTL/DFA logic
|   |-- analysis/             # Post-evaluation analysis scripts
|   |-- docs/
|   |   |-- dfa_pseudo_code/
|   |   `-- predicate_ltl_design/
|   `-- install_mona.sh       # MONA installer for DFA construction
|-- run_scripts/              # Slurm launch and evaluation scripts
|-- examples/                 # Qualitative safety-category videos and monitor outputs
|-- launch_groot.sh           # GR00T evaluation launcher
|-- launch_openpi.sh          # OpenPI evaluation launcher
`-- install.sh                # Repository setup helper
```

## Main Components

Privileged information export:

- `robocasa/robocasa/environments/kitchen/kitchen.py`

Predicate and attribute computation:

- `robocasa/robocasa/environments/kitchen/attributes.py`
- `robocasa/robocasa/environments/kitchen/predicates.py`

Symbolic monitoring:

- `SafeManip/monitor/run_monitor_on_privileged.py`
- `SafeManip/monitor/monitor.py`
- `SafeManip/monitor/specs.py`
- `SafeManip/monitor/LTLfDFA.py`
- `SafeManip/monitor/monitor_metrics.py`

Evaluation entry points:

- `run_scripts/sbatch_groot_test.sh`
- `run_scripts/eval_groot_single_task.sh`
- `run_scripts/sbatch_openpi_test.sh`
- `run_scripts/eval_openpi_single_task.sh`

Specification and design notes:

- `SafeManip/docs/dfa_pseudo_code/`
- `SafeManip/docs/predicate_ltl_design/`

## Installation

Use a Python environment compatible with the policy stack being evaluated. The evaluation stack uses local editable installs so that the RoboCasa privileged-state changes and policy integration code are picked up directly from this repository.

### Shared RoboCasa and Monitor Setup

Install the simulator dependencies first. Clone RoboSuite into the repository root, then install it in editable mode:

```bash
git clone https://github.com/ARISE-Initiative/robosuite.git robosuite
pip install -e robosuite
```

Then install the local RoboCasa fork and monitor dependencies:

```bash
pip install -e robocasa
bash install.sh
```

The default `install.sh` setup installs:

- the local RoboCasa package in editable mode;
- monitor Python dependencies;
- MONA, which is used for DFA construction, unless it is already available.

For monitor-only setup:

```bash
bash install.sh --monitor-only
```

If MONA build dependencies are missing, install them in the active environment first. For conda environments, one option is:

```bash
conda install -c conda-forge flex bison make gcc gxx wget
```

### GR00T Policy Environment

In the environment used for GR00T evaluation, install the shared RoboCasa stack and GR00T package:

```bash
git clone https://github.com/ARISE-Initiative/robosuite.git robosuite
pip install -e robosuite
pip install -e robocasa
cd Isaac-GR00T
pip install -e .
cd ..
```

The equivalent helper command is:

```bash
bash install.sh --with-groot
```

### OpenPI Policy Environment

In the environment used for OpenPI evaluation, install the shared RoboCasa stack, OpenPI, and the OpenPI client package:

```bash
git clone https://github.com/ARISE-Initiative/robosuite.git robosuite
pip install -e robosuite
pip install -e robocasa
cd openpi
pip install -e .
pip install -e packages/openpi-client/
cd ..
```

The equivalent helper command is:

```bash
bash install.sh --with-openpi
```

## Local Configuration

Machine-specific paths, checkpoint locations, and scheduler settings should be placed in:

```bash
run_scripts/.local_paths.sh
```

This file is ignored by git. A minimal template is:

```bash
export CONDA_SH="${HOME}/miniconda3/etc/profile.d/conda.sh"
export ROBOCASA_CHECKPOINT_SNAPSHOT="${HOME}/.cache/huggingface/hub/models--robocasa--robocasa365_checkpoints/snapshots/<snapshot-id>"
export OPENPI_HF_SNAPSHOT="${ROBOCASA_CHECKPOINT_SNAPSHOT}/openpi"
export GROOT_CHECKPOINT_ROOT="${ROBOCASA_CHECKPOINT_SNAPSHOT}/gr00t_n1-5"
export FOUNDATION_MODEL_ROOT="${GROOT_CHECKPOINT_ROOT}/foundation_model_learning"
export MULTITASK_LEARNING_MODEL_PATH="${GROOT_CHECKPOINT_ROOT}/multitask_learning/checkpoint-120000"

# Optional scheduler settings.
export SLURM_PARTITION="<partition>"
export SLURM_EXCLUDE_NODES="<node1,node2>"
```

## Running Evaluations

GR00T evaluations:

```bash
bash launch_groot.sh
```

The launcher submits the configured task list for these model families:

- `target_posttraining`
- `target_only`
- `pretraining`
- `multitask_learning`

To submit a single GR00T model family:

```bash
MODEL_FAMILY=target_posttraining N_EPISODES=50 bash run_scripts/sbatch_groot_test.sh
```

OpenPI evaluations:

```bash
bash launch_openpi.sh
```

To submit a single OpenPI variant:

```bash
OPENPI_MODEL_VARIANT=pi0 OPENPI_MODEL_FAMILY=pretraining N_EPISODES=50 bash run_scripts/sbatch_openpi_test.sh
```

Task lists are defined in `run_scripts/sbatch_groot_test.sh` and `run_scripts/sbatch_openpi_test.sh`.

## Outputs

GR00T outputs are written to:

```text
results/groot/<model_family>/evals/<split>/<task>/rollout_data/<task>--<timestamp>/
```

OpenPI outputs are written to:

```text
results/openpi/<model_variant>/<model_family>/evals/<split>/<task>/rollout_data/<task>--<timestamp>/
```

Each rollout directory may contain:

- `*.mp4`: rollout videos;
- `privileged_information_<episode>.json`: exported privileged simulator state;
- `privileged_information_<episode>_monitor.json`: symbolic monitor output;
- `stats.json`: aggregate evaluation summary, available after successful run completion.

Scheduler logs are written under:

```text
logs/eval/
```

Qualitative safety-category examples are available in:

```text
/nethome/chuang475/testnvme/projects/SafeManip/examples
```

## Analysis and Paper Figures

To obtain the data needed to regenerate graphs reported in the paper, refer to the analysis instructions in:

```text
SafeManip/analysis/README.md
```

## Manual Monitor Invocation

The monitor can be run directly on an exported privileged trajectory:

```bash
cd SafeManip
python monitor/run_monitor_on_privileged.py \
  ../results/groot/target_posttraining/evals/target/CloseBlenderLid/rollout_data/<rollout>/privileged_information_0.json \
  --output ../results/groot/target_posttraining/evals/target/CloseBlenderLid/rollout_data/<rollout>/privileged_information_0_monitor.json
```

During GR00T and OpenPI evaluations, monitor files are generated automatically when privileged information export is enabled.

## Version-Control Hygiene

The following are intentionally ignored:

- `run_scripts/.local_paths.sh`
- `logs/`
- `results/`
- generated videos, checkpoints, caches, and Python build artifacts

If monitor JSON files are missing, inspect the corresponding scheduler `.out` file for `Failed to run symbolic monitor`.
