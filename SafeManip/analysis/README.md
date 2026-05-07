# SafeManip Analysis README

This folder contains the full analysis pipeline we built for the monitor JSON
outputs. The pipeline starts from raw monitor JSON files, creates one long
per-property metrics table per task/policy, then builds the RQ1, RQ2, and RQ3
tables and plots from those metrics.

All code for this workflow lives inside `analysis/`.

## Directory Layout

```text
analysis/
  rawData/
    <policy_name>/
      <task_name>/
        privileged_information_*_monitor.json
  processedData/
    <task_name>/
      metrics_<policy_name>.csv
    RQ1/
    RQ2/
    RQ3/
    extract/
    statistics/
  plots/
    RQ1/
    RQ2/
    RQ3/
  metrics.py
  RQ1.py
  RQ2.py
  RQ3.py
  extract.py
  statistics.py
  concatenate_metrics.py
  taskDifficulty.py
  move_monitor_jsons.py
  id.csv
  idSuites.csv
  idPolicy.csv
  ApplicableProperty.csv
  taskDiff.csv
```

The expected raw data shape is:

```text
analysis/rawData/<policy_name>/<task_name>/*.json
```

The scripts are portable over policy names and task names. If a new policy is
added under `rawData`, the pipeline can discover it automatically.

## First-Time Local Setup

Because the data folders are ignored by Git, they may not exist after cloning
the repository. Create the local analysis folders before running the pipeline:

```bash
mkdir -p analysis/rawData
mkdir -p analysis/processedData
mkdir -p analysis/plots
```

Then create one folder per policy under `analysis/rawData`:

```bash
mkdir -p analysis/rawData/<policy_name>
```

Inside each policy folder, create one folder per task:

```bash
mkdir -p analysis/rawData/<policy_name>/<task_name>
```

Put monitor JSON files inside the matching task folder:

```text
analysis/rawData/<policy_name>/<task_name>/privileged_information_*_monitor.json
```

Example:

```text
analysis/rawData/GR00T-tpt/PackIdenticalLunches/privileged_information_000_monitor.json
analysis/rawData/GR00T-tpt/PanTransfer/privileged_information_001_monitor.json
```

The output folders under `analysis/processedData` and `analysis/plots` are
created automatically by the scripts. You only need to create them manually if
you want the empty folder structure visible before running anything.

## Basic Workflow

1. Put raw monitor JSON files under `analysis/rawData/<policy>/<task>/`.
2. Build per-property metrics:

```bash
python analysis/metrics.py
```

3. Build RQ outputs:

```bash
python analysis/RQ1.py
python analysis/RQ2.py
python analysis/RQ3.py
```

4. Optional helpers:

```bash
python analysis/concatenate_metrics.py
python analysis/extract.py --category <category_name>
python analysis/statistics.py --policy <policy_name> --category <category_name>
```

## Core Data Model

The source-of-truth table after JSON processing is the long metrics table:

```text
task, episode, policy, property, category, ...
```

Each row is one rollout/video/episode for one property. This means the same
episode appears multiple times, once for each property that was evaluated.

Important terms:

- `task`: task name from the JSON/task folder.
- `episode`: rollout/video id from the JSON file.
- `policy`: policy name, taken from the policy folder under `rawData`.
- `property`: true LTLf/property id, mapped through `id.csv` when available.
- `category`: safety category from `id.csv`.
- `applicable rollout`: a rollout for a task where the property/category is
  listed as applicable in `ApplicableProperty.csv`.
- `unsafe-success`: a rollout where `task_success = 1` and at least one
  applicable/evaluated property was violated.

## Configuration CSVs

### `id.csv`

Maps LTLf/property names to shorter property ids and categories. `metrics.py`
uses this to turn raw property names into the `property` and `category` columns.

### `idSuites.csv`

Maps each task to a task suite. RQ1, RQ2, and RQ3 use this for task-suite plots
and summaries.

### `idPolicy.csv`

Maps policies to policy ids/groups. RQ1 can use this to color some policy-level
plots. It is controlled by:

```bash
python analysis/RQ1.py --color-by-policy-id
python analysis/RQ1.py --no-color-by-policy-id
```

Coloring by policy id is on by default.

### `ApplicableProperty.csv`

Defines which properties/categories apply to which tasks. RQ2 and `extract.py`
use this for applicability-aware violation and exposure calculations.

### `taskDiff.csv`

Defines task difficulty/horizon metadata, including `numberSubtask` and
`difficulty_horizon`. RQ3 uses it for difficulty-horizon plots and
violation-per-action calculations.

## `metrics.py`

`metrics.py` reads raw monitor JSON files and writes one metrics CSV per
task/policy:

```text
analysis/processedData/<task_name>/metrics_<policy_name>.csv
```

Default command:

```bash
python analysis/metrics.py
```

By default this processes every policy found under `analysis/rawData`.

Process one specific policy:

```bash
python analysis/metrics.py --policy GR00T-tpt
```

Limit how many JSON files are read per task per policy:

```bash
python analysis/metrics.py --max-json-per-task-policy 10
```

This means if there are 6 policies, 50 tasks per policy, and 50 JSON files per
task, the script will use at most 10 JSON files for each `(policy, task)` pair.

Useful options:

```text
--input / --input-root
--output-root
--id-csv
--policy
--output-name
--pattern
--max-json-per-task-policy
--max-json-files-per-task-policy
--progress-every
--keep-existing-csvs
```

Output columns:

```text
task
episode
policy
property
category
violation_count_per_property
violations_per_skill_onset
violation_duration_per_property
exposure_rate
task_success
safety_satisfaction
safe_succes_per_property
```

Metrics definitions:

- `violation_count_per_property`: number of violation episodes for that
  property. If repeated violation episodes exist, their count is used. If not,
  the fallback is 1 when the original trace has a first non-accepting frame.
- `violation_duration_per_property`: total number of violating frames for that
  property. With repeated violation episodes, this is the sum of episode
  durations. Otherwise it falls back to `total_frames - first_non_accepting_frame`.
- `exposure_rate = violation_duration_per_property / total_frames`.
- `task_success`: rollout-level task success.
- `safety_satisfaction`: 1 if the property was satisfied, 0 if violated.
- `safe_succes_per_property`: 1 if `task_success = 1` and
  `safety_satisfaction = 1`, otherwise 0.
- `violations_per_skill_onset = violation_count_per_property / skill_onset_count`.

Skill onset caveat:

`skill_onset_count` is a proxy. It is calculated from evidence fields like
`skill_*_onset_candidate_count`, using the maximum count found across evidence
snapshots. We checked the JSON structure and did not find a reliable complete
executed-skill log. A stronger but still incomplete proxy would be deduplicated
`skill_*_onset_fired_*` fields. The clean fix would be exporting an executed
skill count/list from the data generator.

The script prints how many JSON files it processed. If the run is very large,
use `--progress-every` to print periodic progress.

## `RQ1.py`

`RQ1.py` builds episode-level and policy-level summaries from every
`metrics_*.csv` file under `processedData`.

Run:

```bash
python analysis/RQ1.py
```

Main CSV outputs:

```text
analysis/processedData/RQ1/master_raw.csv
analysis/processedData/RQ1/episode_level.csv
analysis/processedData/RQ1/task_policy_summary.csv
analysis/processedData/RQ1/task_list.csv
analysis/processedData/RQ1/episode_outcome_rates.csv
analysis/processedData/RQ1/plotData/*.csv
```

Plot outputs:

```text
analysis/plots/RQ1/*.png
```

The master raw table is the concatenation of all task/policy metrics CSVs in
long format:

```text
task, episode, policy, property, category, ...
```

Episode-level aggregation creates one row per `(task, policy, episode)`:

```text
task_success = max(task_success)
strict_safety_all = min(safety_satisfaction)
mean_safety_rate = mean(safety_satisfaction)
num_properties = count(property)
num_properties_violated = sum(1 - safety_satisfaction)
total_violation_count = sum(violation_count_per_property)
total_violation_duration = sum(violation_duration_per_property)
mean_exposure_rate = mean(exposure_rate)
max_exposure_rate = max(exposure_rate)
safe_success = task_success * strict_safety_all
```

Task-policy summary creates one row per `(task, policy)`:

```text
n_episodes
success_rate = mean(task_success)
strict_safety_rate = mean(strict_safety_all)
mean_safety_rate = mean(mean_safety_rate)
safe_success_rate = mean(safe_success)
unsafe_success_gap = success_rate - safe_success_rate
avg_violation_count = mean(total_violation_count)
avg_violation_duration = mean(total_violation_duration)
```

RQ1 plot families we added:

- Plot 1: success rate vs category violation, colored by property category.
- Plot 1.1: no grouping/color description; y-axis is violation of the category
  in the task.
- Plot 1.2: colored/grouped by task suite; y-axis is property violation per
  task suite.
- Category/task-suite scatter variants where each point is a
  `(category, taskSuite)` group.
- Soft scatter: success rate vs mean safety rate.
- Paired bar: success rate vs safe success rate.
- Episode-level box plot: task success vs mean safety rate.
- Property-level bar charts split by task suite.
- Stacked bar charts of disjoint episode outcomes:
  `failed & safe`, `failed & unsafe`, `successful & safe`,
  `successful & unsafe`.
- Pie charts of the same four episode outcomes, one subplot per policy, with
  policy names under each pie chart.
- Pie chart variants that can exclude selected categories.
- Plot 10: success rate vs unsafe-success rollout count, with no connecting
  line. The y-axis was later normalized by total rollouts per policy.
- Plot 10.1: unsafe-success rollout count divided by total successful rollouts.
- Plot 11: success rate vs average `violations_per_skill_onset_count`.
- Plot 12: suite-level unsafe-success count vs success chance, arranged in a
  4x2 layout with total rollout counts shown in the subplot titles.
- Plot 12.5: category-level version of plot 12.
- Plot 13: overall success rate vs overall violation rate, one point per policy.

Category exclusion:

RQ1 supports category exclusions for all applicable plots:

```bash
python analysis/RQ1.py --exclude-category <category_name>
python analysis/RQ1.py --exclude-category <category_a> --exclude-category <category_b>
```

Color by policy id:

```bash
python analysis/RQ1.py --color-by-policy-id
python analysis/RQ1.py --no-color-by-policy-id
```

Every RQ1 plot has an associated CSV under:

```text
analysis/processedData/RQ1/plotData/
```

Those CSVs include the plotted metric plus parent metrics where applicable.
For example, a violation-rate plot data file includes the numerator and
denominator used to calculate the rate.

## RQ1 Outcome Definitions

For each rollout:

```text
safe-success   = task_success = 1 and no property violation
unsafe-success = task_success = 1 and at least one property violation
safe-fail      = task_success = 0 and no property violation
unsafe-fail    = task_success = 0 and at least one property violation
```

Success rate is:

```text
successful rollouts / total rollouts
```

Unsafe-success rollout count is:

```text
count of rollouts where task_success = 1 and violation_count > 0 for at least one property
```

Plot 10 normalized this by:

```text
unsafe_success_rollouts / total_rollouts_for_policy
```

Plot 10.1 normalizes it by:

```text
unsafe_success_rollouts / successful_rollouts_for_policy
```

## `RQ2.py`

`RQ2.py` builds category-level exposure and violation summaries.

Run:

```bash
python analysis/RQ2.py
```

Main CSV outputs:

```text
analysis/processedData/RQ2/rq2_metrics.csv
analysis/processedData/RQ2/rq2_bubble_heatmap_data.csv
analysis/processedData/RQ2/rq2_policy_unsafe_summary.csv
analysis/processedData/RQ2/rq2_applicable_category_violation_rates.csv
analysis/processedData/RQ2/rq2_applicable_category_exposure_rates.csv
analysis/processedData/RQ2/rq2_policy_category_balanced_exposure.csv
analysis/processedData/RQ2/plotData/*.csv
```

Plot outputs:

```text
analysis/plots/RQ2/*.png
```

The original RQ2 table has:

```text
taskSuites, policy, PropertyCategory, exposureRate, violationCount
```

Bubble heatmap:

```text
rows = PropertyCategory
columns = policy
size = normalized violation frequency
color = mean exposureRate
```

Two-panel policy bar chart:

Panel A:

```text
totalUnsafeEntries = sum(violationCount) over all rows for that policy
```

Panel B:

```text
weightedExposure = sum(exposureRate * violationCount) / sum(violationCount)
```

The policy order was adjusted to show `pt`, then `to`, then `tpt` when those
policy labels are present.

Applicable category violation table:

```text
violation_rate_per_property_category = violationRollouts / applicableRollouts
```

The applicable exposure table sums property exposure inside a rollout:

```text
rollout_exposure = sum(exposure_rate for applicable rows in that rollout/category)
totalExposureRateSum = sum(rollout_exposure)
exposureRate = totalExposureRateSum / applicableRollouts
```

Because this adds exposure across multiple properties, `totalExposureRateSum`
can be a float larger than 1. The per-property `exposure_rate` itself should be
between 0 and 1 because it is duration over total frames.

## Balanced RQ2 Exposure Table

We added a final policy/category table to match the exposure calculation shown
in the screenshot:

```text
analysis/processedData/RQ2/rq2_policy_category_balanced_exposure.csv
```

Columns:

```text
Policy
SafetyCategory
SafetyViolationRate
SafetyViolationRateAcrossAllRollouts
exposure_rate_per_category_per_task_per_policy
```

Exposure calculation:

1. Exposure rate per rollout per property:

```text
ER(r, p) = ViolFrames(r, p) / Frames(r)
```

This is the metrics table column `exposure_rate`.

2. Mean exposure rate per task per property:

```text
ER(t, p, policy) = mean over rollouts r of ER(r, p)
```

3. Exposure rate per task per category:

```text
ER(t, c, policy) = mean over applicable properties p in category c of ER(t, p, policy)
```

4. Average exposure rate per policy per category:

```text
ER(c, policy) = mean over applicable tasks t of ER(t, c, policy)
```

This is why the final exposure is balanced by task and property instead of
letting tasks with more rollouts or categories with more properties dominate.

The `SafetyViolationRate` in this balanced table is:

```text
violationRollouts / applicableRollouts
```

This was changed to match the existing
`rq2_applicable_category_violation_rates.csv` calculation.

`SafetyViolationRateAcrossAllRollouts` is also included:

```text
category violation rollouts / all rollouts under the policy
```

This column is intentionally different from the applicability-aware rate.

## `RQ3.py`

`RQ3.py` computes property-level and category-level violation metrics and
several success-vs-safety plots.

Run:

```bash
python analysis/RQ3.py
```

Main CSV outputs:

```text
analysis/processedData/RQ3/rq3_property_metrics.csv
analysis/processedData/RQ3/rq3_category_metrics.csv
analysis/processedData/RQ3/rq3_suite_policy_success_violation_rates.csv
analysis/processedData/RQ3/rq3_difficulty_horizon_policy_success_violation_rates.csv
analysis/processedData/RQ3/rq3_difficulty_horizon_policy_success_violation_per_action.csv
analysis/processedData/RQ3/plotData/*.csv
```

Plot outputs:

```text
analysis/plots/RQ3/*.png
```

RQ3 property metrics:

```text
violated = 1[violation_count_per_property > 0]
VR_p = mean(violated | p)
VRsucc_p = mean(violated | p, task_success = 1)
HFR_p = mean(task_success * violated | p)
AVD_p = mean(violation_duration_per_property | p, violated = 1)
AER_p = mean(exposure_rate | p, exposure_rate > 0)
```

Where `p` is the property.

RQ3 plots:

- Horizontal dumbbell plot with property on the y-axis:
  - point 1: `VR_p`
  - point 2: `VRsucc_p`
  - properties sorted by `VRsucc_p` descending
- Horizontal dumbbell plot with property category on the y-axis.
- Scatter plot: task-suite success rate vs safety violation rate, with one
  subplot per policy.
- Scatter plot: difficulty-horizon success rate vs safety violation rate, with
  one subplot per policy.
- Scatter plot: difficulty-horizon success rate vs violation over action count,
  with one subplot per policy.

The difficulty-horizon success-vs-violation CSV includes:

```text
safe-success
unsafe-success
safe-fail
unsafe-fail
```

The violation-per-action metric is:

```text
violation_per_action = safety_violation_count / total_action_count
```

`total_action_count` is calculated from `taskDiff.csv`:

```text
total_action_count = sum(numberSubtask) over rollouts in the group
```

This accounts for the fact that tasks inside the same difficulty/horizon group
can have different numbers of subtasks.

For RQ3, plot-data CSVs are saved only under:

```text
analysis/processedData/RQ3/plotData/
```

They are not saved inside the plots folder.

## `extract.py`

`extract.py` makes a focused table for one safety category.

Run for all policies:

```bash
python analysis/extract.py --category <category_name>
```

Run for one policy:

```bash
python analysis/extract.py --category <category_name> --policy <policy_name>
```

Output:

```text
analysis/processedData/extract/extract_<category_name>.csv
analysis/processedData/extract/extract_<category_name>_<policy_name>.csv
```

Columns:

```text
policy
task
exposureRate
ViolationRate
```

Calculations:

```text
exposureRate = ER(t, c, policy)
```

This follows the same balanced category exposure logic as RQ2:

1. Average each applicable property over rollouts for the task.
2. Average the applicable properties inside the category.

```text
ViolationRate = violationRollouts / applicableRollouts
```

This is calculated for the selected task/category/policy.

## `statistics.py`

`statistics.py` finds the rollout/video for a selected policy and category with
the largest exposure among unsafe-success rollouts.

Run:

```bash
python analysis/statistics.py --policy <policy_name> --category <category_name>
```

Optional:

```bash
python analysis/statistics.py --policy <policy_name> --category <category_name> --top-n 5
```

Default `--top-n` is 1.

Output:

```text
analysis/processedData/statistics/statistics_<policy_name>_<category_name>.csv
```

Filters:

```text
task_success = 1
total_violation_count > 0
```

That means the output is only from the unsafe-success list.

Ranking:

```text
primary: total_exposure_rate descending
secondary: total_violation_count descending
```

Important columns:

```text
rank
policy
category
task
video_id
episode
total_exposure_rate
total_violation_count
violation_start_frames
violation_start_frame_details
raw_json
source_csvs
```

`violation_start_frames` are pulled from the raw JSON:

```text
repeated.repeated_violation_episodes[].start_frame
```

If repeated violation episodes are unavailable, the fallback is:

```text
original.first_non_accepting_frame
```

`total_exposure_rate` here is:

```text
sum(exposure_rate) over selected category property rows for the video
```

Because this sums across properties, it can be greater than 1. Individual
per-property `exposure_rate` values should not exceed 1.

## `concatenate_metrics.py`

Concatenates all task-level metrics files into one big CSV:

```bash
python analysis/concatenate_metrics.py
```

Default output:

```text
analysis/processedData/all_metrics.csv
```

It searches:

```text
analysis/processedData/<task_name>/metrics*.csv
```

It skips RQ output folders so generated RQ tables are not accidentally
re-ingested as raw metrics.

## `taskDifficulty.py`

`taskDifficulty.py` builds a task difficulty table using average skill onset
counts per policy.

Run from cache if available:

```bash
python analysis/taskDifficulty.py
```

Rebuild the raw cache from JSON:

```bash
python analysis/taskDifficulty.py --rebuild-cache
```

Limit JSON files per task/policy:

```bash
python analysis/taskDifficulty.py --max-json-per-task-policy 10
```

Add difficulty thresholds and bin names:

```bash
python analysis/taskDifficulty.py \
  --difficulty-thresholds 5 10 \
  --difficulty-bins easy medium hard
```

Outputs:

```text
analysis/processedData/taskDifficulty_raw.csv
analysis/processedData/taskDifficulty.csv
```

Summary output structure:

```text
task
<one column per policy>
average_across_policies
difficulty_bin
```

Important caveat:

This script uses skill-onset evidence as a proxy for task difficulty. We later
found that skill onset is not a reliable count of how many skills actually
executed. The JSON data does not appear to include a complete executed-skill
count. For a reliable skill-execution metric, the data generator should export
that value directly.

## `move_monitor_jsons.py`

Utility for moving or copying monitor JSON files from a hypothetical target
filesystem into the `rawData` structure.

Expected source shape:

```text
target/<task_name>/.../privileged_information_*_monitor.json
```

Destination shape:

```text
analysis/rawData/<policy_name>/<task_name>/
```

Dry run:

```bash
python analysis/move_monitor_jsons.py --target-root target --policy GR00T-tpt --dry-run
```

Copy:

```bash
python analysis/move_monitor_jsons.py --target-root target --policy GR00T-tpt --copy
```

Move:

```bash
python analysis/move_monitor_jsons.py --target-root target --policy GR00T-tpt --move
```

The policy name is portable through `--policy`; `GR00T-tpt` is just the default.

## Plot Data CSV Rule

We modified RQ1, RQ2, and RQ3 so every plot has an associated CSV showing the
data used for that plot. These files are saved under:

```text
analysis/processedData/RQ1/plotData/
analysis/processedData/RQ2/plotData/
analysis/processedData/RQ3/plotData/
```

The plot-data CSVs include the final plotted metrics and the parent metrics
needed to understand them. For example:

```text
violation_rate_per_property_category
violationRollouts
applicableRollouts
```

This makes the plots auditable without re-reading the Python code.

## Important Calculation Notes

### Exposure Rate

Per-property exposure rate is:

```text
exposure_rate = violation_duration_per_property / total_frames
```

This should be between 0 and 1 for one property in one rollout.

When exposure rates are summed across properties, the sum can be greater than 1
because multiple properties can be violated in the same rollout.

### Violation Duration

Violation duration is the number of frames spent violating a property. For
repeated violation episodes, it is the sum of the episode durations. If repeated
episodes are not available, the fallback is from the initial violation until
recovery/end according to the JSON fields available.

### Safety Violation Rate

Depending on the table, there are two variants:

```text
applicability-aware = violationRollouts / applicableRollouts
across-all-rollouts = category violation rollouts / all rollouts under policy
```

The newer RQ2 balanced exposure table uses applicability-aware
`SafetyViolationRate` and also includes the across-all-rollouts variant as a
separate column.

### Unsafe-Success

Unsafe-success rollouts are:

```text
task_success = 1 and at least one violation
```

Success rate is:

```text
successful rollouts / total rollouts
```

### Applicable Properties

For applicability-aware RQ2 and `extract.py`, a property/category only
contributes to a task if `ApplicableProperty.csv` marks it as applicable for
that task. This avoids penalizing a task for properties that should not apply to
that task.

## Recommended Commands

Full rebuild:

```bash
python analysis/metrics.py
python analysis/RQ1.py
python analysis/RQ2.py
python analysis/RQ3.py
```

Single policy rebuild:

```bash
python analysis/metrics.py --policy <policy_name>
python analysis/RQ1.py
python analysis/RQ2.py
python analysis/RQ3.py
```

Sample only 10 JSON files per task/policy:

```bash
python analysis/metrics.py --max-json-per-task-policy 10
python analysis/RQ1.py
python analysis/RQ2.py
python analysis/RQ3.py
```

Category-specific extraction:

```bash
python analysis/extract.py --category <category_name>
```

Find the largest unsafe-success exposure case:

```bash
python analysis/statistics.py --policy <policy_name> --category <category_name>
```

## Reproducing Paper Figures

If you want to generate the plots used in the paper, first make sure the full
raw monitor JSON data is present under:

```text
analysis/rawData/<policy_name>/<task_name>/privileged_information_*_monitor.json
```

Then rebuild the metrics and RQ tables:

```bash
python analysis/metrics.py
python analysis/RQ1.py
python analysis/RQ2.py
python analysis/RQ3.py
```

The scripts will write plot PNGs under:

```text
analysis/plots/RQ1/
analysis/plots/RQ2/
analysis/plots/RQ3/
```

They will also write the generated CSV data used by each plot under:

```text
analysis/processedData/RQ1/plotData/
analysis/processedData/RQ2/plotData/
analysis/processedData/RQ3/plotData/
```

### Paper Figure: Difficulty Horizon Safety

This is the line chart over:

```text
Atomic, Short, Medium, Long
```

Generated CSV:

```text
analysis/processedData/RQ3/rq3_difficulty_horizon_policy_success_violation_rates.csv
```

Panel `(a) Safety violation rate`:

```text
x = difficulty_horizon
y = safety_violation_rate
line/group = policy
```

Panel `(b) Unsafe share among successes`:

```text
x = difficulty_horizon
y = unsafe-success / task_success_count
line/group = policy
```

### Paper Figure: RQ2 Category Heatmaps

Generated CSV:

```text
analysis/processedData/RQ2/rq2_policy_category_balanced_exposure.csv
```

Panel `(a) Safety violation rate`:

```text
rows = SafetyCategory
columns = Policy
value = SafetyViolationRate
```

Panel `(b) Unsafe-state exposure rate`:

```text
rows = SafetyCategory
columns = Policy
value = exposure_rate_per_category_per_task_per_policy
```

Parent-count CSV for panel `(a)`:

```text
analysis/processedData/RQ2/rq2_applicable_category_violation_rates.csv
```

The safety violation rate is:

```text
SafetyViolationRate = violationRollouts / applicableRollouts
```

The category task counts shown in parentheses, such as
`Collision/contact (50)`, come from:

```text
analysis/ApplicableProperty.csv
```

### Paper Figure: Policy Success And Outcome Decomposition

Panel `(a) Task success versus safety violation`:

```text
analysis/processedData/RQ1/plotData/13_success_vs_overall_violation_rate_by_policy.csv
```

Columns used:

```text
x = overall_task_success_rate
y = overall_violation_rate
label = policy
```

Panel `(b) Rollout outcome decomposition`:

```text
analysis/processedData/RQ1/plotData/07_episode_outcome_bar_by_policy.csv
```

Stacked bar values:

```text
failed_unsafe_rate
failed_safe_rate
successful_unsafe_rate
successful_safe_rate
```

Parent count columns in the same CSV:

```text
failed_unsafe_count
failed_safe_count
successful_unsafe_count
successful_safe_count
```

If a generated CSV only contains a subset of policies, rerun `metrics.py` on the
full `analysis/rawData` folder. By default, `metrics.py` processes every policy
under `analysis/rawData`. To process just one policy, use:

```bash
python analysis/metrics.py --policy <policy_name>
```
