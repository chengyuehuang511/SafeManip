from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


ANALYSIS_DIR = Path(__file__).resolve().parent
DEFAULT_PROCESSED_ROOT = ANALYSIS_DIR / "processedData"
DEFAULT_OUTPUT_DIR = DEFAULT_PROCESSED_ROOT / "RQ2"
DEFAULT_PLOT_DIR = ANALYSIS_DIR / "plots" / "RQ2"
DEFAULT_SUITES_CSV = ANALYSIS_DIR / "idSuites.csv"
DEFAULT_APPLICABLE_PROPERTIES_CSV = ANALYSIS_DIR / "ApplicableProperty.csv"


RAW_REQUIRED_COLUMNS = (
    "task",
    "episode",
    "policy",
    "property",
    "category",
    "exposure_rate",
    "violation_count_per_property",
)


OUTPUT_COLUMNS = (
    "taskSuites",
    "policy",
    "PropertyCategory",
    "exposureRate",
    "violationCount",
)


HEATMAP_COLUMNS = (
    "PropertyCategory",
    "policy",
    "violationRows",
    "normalizedViolationFrequency",
    "exposureRateSum",
    "meanExposureRate",
    "nRows",
)


POLICY_UNSAFE_COLUMNS = (
    "policy",
    "totalUnsafeEntries",
    "weightedExposureNumerator",
    "weightedExposure",
)


APPLICABLE_RATE_COLUMNS = (
    "PropertyCategory",
    "policy",
    "violationRollouts",
    "applicableRollouts",
    "violation_rate_per_property_category",
)


APPLICABLE_EXPOSURE_COLUMNS = (
    "PropertyCategory",
    "policy",
    "totalExposureRateSum",
    "exposureRate",
    "applicableRollouts",
    "exposureRows",
)


POLICY_CATEGORY_BALANCED_COLUMNS = (
    "Policy",
    "SafetyCategory",
    "SafetyViolationRate",
    "SafetyViolationRateAcrossAllRollouts",
    "exposure_rate_per_category_per_task_per_policy",
)


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    return int(_to_float(value, float(default)))


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _normalize_ltl(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _metric_csvs(processed_root: Path, pattern: str) -> List[Path]:
    excluded_dirs = {
        (processed_root / "RQ1").resolve(),
        (processed_root / "RQ2").resolve(),
        (processed_root / "RQ3").resolve(),
    }
    paths = []
    for path in sorted(processed_root.rglob(pattern)):
        if not path.is_file():
            continue
        if any(_is_relative_to(path.resolve(), excluded_dir) for excluded_dir in excluded_dirs):
            continue
        paths.append(path)
    return paths


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _resolve_applicable_properties_path(path: Path) -> Path:
    if path.exists():
        return path

    alternates = [
        path.with_name("ApplicableProperties.csv"),
        path.with_name("ApplicableProperty.csv"),
    ]
    for alternate in alternates:
        if alternate.exists():
            return alternate
    return path


def _read_rows(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = [
                column for column in RAW_REQUIRED_COLUMNS if column not in (reader.fieldnames or [])
            ]
            if missing:
                raise SystemExit(f"{path} is missing required columns: {missing}")
            for row in reader:
                rows.append({column: row.get(column, "") for column in RAW_REQUIRED_COLUMNS})
    return rows


def load_task_suites(path: Path) -> Dict[str, str]:
    if not path.exists():
        raise SystemExit(f"Task suites CSV not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit(f"Task suites CSV is empty: {path}")
        normalized_fields = {field.lower(): field for field in reader.fieldnames}
        suite_field = normalized_fields.get("tasksuites")
        task_field = normalized_fields.get("taskname")
        if not suite_field or not task_field:
            raise SystemExit(f"Expected columns taskSuites and taskName in: {path}")

        suites = {}
        for row in reader:
            task = str(row.get(task_field) or "").strip()
            suite = str(row.get(suite_field) or "").strip()
            if task:
                suites[task] = suite or "UnmappedSuite"
        return suites


def load_applicable_properties(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Applicable properties CSV not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit(f"Applicable properties CSV is empty: {path}")

        normalized_fields = {field.lower(): field for field in reader.fieldnames}
        id_field = normalized_fields.get("id")
        ltlf_field = normalized_fields.get("ltlf")
        category_field = normalized_fields.get("category")
        if not id_field or not category_field:
            raise SystemExit(f"Expected columns ID and category in: {path}")

        metadata_fields = {
            field
            for field in (id_field, ltlf_field, category_field)
            if field is not None
        }
        task_fields = [field for field in reader.fieldnames if field not in metadata_fields]
        property_lookup: Dict[str, Dict[str, Any]] = {}
        category_tasks: Dict[str, set[str]] = defaultdict(set)
        properties: List[Dict[str, Any]] = []

        for row in reader:
            property_id = str(row.get(id_field) or "").strip()
            ltlf = _normalize_ltl(row.get(ltlf_field)) if ltlf_field else ""
            category = str(row.get(category_field) or "Uncategorized").strip() or "Uncategorized"
            applicable_tasks = {
                task
                for task in task_fields
                if _to_int(row.get(task)) == 1
            }
            property_key = property_id or ltlf
            property_info = {
                "property": property_key,
                "id": property_id,
                "ltlf": ltlf,
                "category": category,
                "tasks": applicable_tasks,
            }
            properties.append(property_info)

            for alias in (property_id, ltlf):
                if alias:
                    property_lookup[alias] = property_info
            for task in applicable_tasks:
                category_tasks[category].add(task)

    return {
        "property_lookup": property_lookup,
        "category_tasks": category_tasks,
        "properties": properties,
    }


def _property_info_for_row(
    row: Dict[str, Any],
    property_lookup: Dict[str, Dict[str, Any]],
) -> Dict[str, Any] | None:
    property_id = str(row.get("property") or "")
    return property_lookup.get(property_id) or property_lookup.get(_normalize_ltl(property_id))


def build_rq2_rows(
    raw_rows: Sequence[Dict[str, Any]],
    task_suites: Dict[str, str],
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        task = str(row.get("task", ""))
        task_suite = task_suites.get(task, "UnmappedSuite")
        policy = str(row.get("policy") or "unknown_policy")
        category = str(row.get("category") or "Uncategorized")
        grouped[(task_suite, policy, category)].append(row)

    output_rows = []
    for (task_suite, policy, category), rows in sorted(grouped.items()):
        output_rows.append(
            {
                "taskSuites": task_suite,
                "policy": policy,
                "PropertyCategory": category,
                "exposureRate": _mean([_to_float(row["exposure_rate"]) for row in rows]),
                "violationCount": sum(
                    _to_int(row["violation_count_per_property"]) for row in rows
                ),
            }
        )
    return output_rows


def build_heatmap_rows(raw_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        policy = str(row.get("policy") or "unknown_policy")
        category = str(row.get("category") or "Uncategorized")
        grouped[(category, policy)].append(row)

    heatmap_rows = []
    for (category, policy), rows in sorted(grouped.items()):
        violated = [
            1 if _to_int(row["violation_count_per_property"]) > 0 else 0
            for row in rows
        ]
        exposure_rate_sum = sum(_to_float(row["exposure_rate"]) for row in rows)
        heatmap_rows.append(
            {
                "PropertyCategory": category,
                "policy": policy,
                "violationRows": sum(violated),
                "normalizedViolationFrequency": _mean(violated),
                "exposureRateSum": exposure_rate_sum,
                "meanExposureRate": exposure_rate_sum / len(rows) if rows else 0.0,
                "nRows": len(rows),
            }
        )
    return heatmap_rows


def build_policy_unsafe_rows(rq2_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rq2_rows:
        policy = str(row.get("policy") or "unknown_policy")
        grouped[policy].append(row)

    policy_rows = []
    for policy, rows in grouped.items():
        total_unsafe_entries = sum(_to_int(row["violationCount"]) for row in rows)
        weighted_exposure_num = sum(
            _to_float(row["exposureRate"]) * _to_float(row["violationCount"])
            for row in rows
        )
        weighted_exposure = (
            weighted_exposure_num / total_unsafe_entries
            if total_unsafe_entries
            else 0.0
        )
        policy_rows.append(
            {
                "policy": policy,
                "totalUnsafeEntries": total_unsafe_entries,
                "weightedExposureNumerator": weighted_exposure_num,
                "weightedExposure": weighted_exposure,
            }
        )

    return sorted(
        policy_rows,
        key=lambda row: (-_to_float(row["totalUnsafeEntries"]), str(row["policy"])),
    )


def build_applicable_rate_rows(
    raw_rows: Sequence[Dict[str, Any]],
    applicable_properties: Dict[str, Any],
) -> List[Dict[str, Any]]:
    property_lookup: Dict[str, Dict[str, Any]] = applicable_properties["property_lookup"]
    category_tasks: Dict[str, set[str]] = applicable_properties["category_tasks"]

    episodes_by_policy_task: Dict[Tuple[str, str], set[str]] = defaultdict(set)
    for row in raw_rows:
        policy = str(row.get("policy") or "unknown_policy")
        task = str(row.get("task") or "")
        episode = str(row.get("episode") or "")
        if task and episode:
            episodes_by_policy_task[(policy, task)].add(episode)

    applicable_episode_sets: Dict[Tuple[str, str], set[Tuple[str, str]]] = defaultdict(set)
    for (policy, task), episodes in episodes_by_policy_task.items():
        for category, applicable_tasks in category_tasks.items():
            if task not in applicable_tasks:
                continue
            for episode in episodes:
                applicable_episode_sets[(policy, category)].add((task, episode))

    violation_episode_sets: Dict[Tuple[str, str], set[Tuple[str, str]]] = defaultdict(set)
    for row in raw_rows:
        if _to_int(row["violation_count_per_property"]) <= 0:
            continue

        task = str(row.get("task") or "")
        policy = str(row.get("policy") or "unknown_policy")
        episode = str(row.get("episode") or "")
        property_info = _property_info_for_row(row, property_lookup)

        if property_info and task in property_info["tasks"]:
            category = str(property_info["category"])
        else:
            category = str(row.get("category") or "")
            if not category or task not in category_tasks.get(category, set()):
                continue

        episode_key = (task, episode)
        if episode_key in applicable_episode_sets.get((policy, category), set()):
            violation_episode_sets[(policy, category)].add(episode_key)

    rate_rows = []
    for (policy, category), applicable_episodes in sorted(applicable_episode_sets.items()):
        applicable_count = len(applicable_episodes)
        violation_count = len(violation_episode_sets.get((policy, category), set()))
        rate_rows.append(
            {
                "PropertyCategory": category,
                "policy": policy,
                "violationRollouts": violation_count,
                "applicableRollouts": applicable_count,
                "violation_rate_per_property_category": (
                    violation_count / applicable_count if applicable_count else 0.0
                ),
            }
        )
    return rate_rows


def build_applicable_exposure_rows(
    raw_rows: Sequence[Dict[str, Any]],
    applicable_properties: Dict[str, Any],
) -> List[Dict[str, Any]]:
    property_lookup: Dict[str, Dict[str, Any]] = applicable_properties["property_lookup"]
    category_tasks: Dict[str, set[str]] = applicable_properties["category_tasks"]

    episodes_by_policy_task: Dict[Tuple[str, str], set[str]] = defaultdict(set)
    for row in raw_rows:
        policy = str(row.get("policy") or "unknown_policy")
        task = str(row.get("task") or "")
        episode = str(row.get("episode") or "")
        if task and episode:
            episodes_by_policy_task[(policy, task)].add(episode)

    applicable_episode_sets: Dict[Tuple[str, str], set[Tuple[str, str]]] = defaultdict(set)
    for (policy, task), episodes in episodes_by_policy_task.items():
        for category, applicable_tasks in category_tasks.items():
            if task not in applicable_tasks:
                continue
            for episode in episodes:
                applicable_episode_sets[(policy, category)].add((task, episode))

    exposure_by_episode: Dict[Tuple[str, str, str, str], List[float]] = defaultdict(list)
    for row in raw_rows:
        task = str(row.get("task") or "")
        policy = str(row.get("policy") or "unknown_policy")
        episode = str(row.get("episode") or "")
        property_info = _property_info_for_row(row, property_lookup)

        if property_info and task in property_info["tasks"]:
            category = str(property_info["category"])
        else:
            category = str(row.get("category") or "")
            if not category or task not in category_tasks.get(category, set()):
                continue

        episode_key = (task, episode)
        if episode_key in applicable_episode_sets.get((policy, category), set()):
            exposure_by_episode[(policy, category, task, episode)].append(
                _to_float(row["exposure_rate"])
            )

    exposure_rows = []
    for (policy, category), applicable_episodes in sorted(applicable_episode_sets.items()):
        rollout_exposures = [
            sum(exposure_by_episode.get((policy, category, task, episode), []))
            for task, episode in sorted(applicable_episodes)
        ]
        total_exposure_rate_sum = sum(rollout_exposures)
        exposure_row_count = sum(
            len(exposure_by_episode.get((policy, category, task, episode), []))
            for task, episode in applicable_episodes
        )
        exposure_rows.append(
            {
                "PropertyCategory": category,
                "policy": policy,
                "totalExposureRateSum": total_exposure_rate_sum,
                "exposureRate": (
                    total_exposure_rate_sum / len(applicable_episodes)
                    if applicable_episodes
                    else 0.0
                ),
                "applicableRollouts": len(applicable_episodes),
                "exposureRows": exposure_row_count,
            }
        )
    return exposure_rows


def build_policy_category_balanced_rows(
    raw_rows: Sequence[Dict[str, Any]],
    applicable_properties: Dict[str, Any],
    applicable_rate_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    property_lookup: Dict[str, Dict[str, Any]] = applicable_properties["property_lookup"]
    category_tasks: Dict[str, set[str]] = applicable_properties["category_tasks"]
    properties: Sequence[Dict[str, Any]] = applicable_properties["properties"]
    applicable_rate_lookup = {
        (str(row["policy"]), str(row["PropertyCategory"])): _to_float(
            row["violation_rate_per_property_category"]
        )
        for row in applicable_rate_rows
    }

    policies = sorted({str(row.get("policy") or "unknown_policy") for row in raw_rows})
    episodes_by_policy_task: Dict[Tuple[str, str], set[str]] = defaultdict(set)
    episodes_by_policy: Dict[str, set[Tuple[str, str]]] = defaultdict(set)
    exposure_sum_by_policy_task_property: Dict[Tuple[str, str, str], float] = defaultdict(float)
    violated_episodes_by_policy_category: Dict[Tuple[str, str], set[Tuple[str, str]]] = defaultdict(set)
    applicable_properties_by_task_category: Dict[Tuple[str, str], List[str]] = defaultdict(list)

    for property_info in properties:
        property_id = str(property_info.get("id") or property_info.get("property") or "")
        category = str(property_info.get("category") or "Uncategorized")
        for task in property_info.get("tasks", set()):
            applicable_properties_by_task_category[(task, category)].append(property_id)

    for row in raw_rows:
        policy = str(row.get("policy") or "unknown_policy")
        task = str(row.get("task") or "")
        episode = str(row.get("episode") or "")
        if task and episode:
            episodes_by_policy_task[(policy, task)].add(episode)
            episodes_by_policy[policy].add((task, episode))

        property_info = _property_info_for_row(row, property_lookup)
        category = (
            str(property_info.get("category") or "Uncategorized")
            if property_info
            else str(row.get("category") or "")
        )
        if category and _to_int(row["violation_count_per_property"]) > 0 and task and episode:
            violated_episodes_by_policy_category[(policy, category)].add((task, episode))

        if not property_info or task not in property_info["tasks"]:
            continue

        canonical_property = str(
            property_info.get("id") or property_info.get("property") or row.get("property") or ""
        )
        exposure_sum_by_policy_task_property[(policy, task, canonical_property)] += _to_float(
            row["exposure_rate"]
        )
    output_rows = []
    for policy in policies:
        policy_tasks = {
            task
            for (row_policy, task), episodes in episodes_by_policy_task.items()
            if row_policy == policy and episodes
        }
        for category, applicable_tasks in sorted(category_tasks.items()):
            category_policy_tasks = sorted(policy_tasks.intersection(applicable_tasks))
            if not category_policy_tasks:
                continue

            task_exposure_values = []
            for task in category_policy_tasks:
                rollout_count = len(episodes_by_policy_task.get((policy, task), set()))
                if rollout_count == 0:
                    continue

                applicable_property_ids = applicable_properties_by_task_category.get(
                    (task, category),
                    [],
                )
                property_means = [
                    exposure_sum_by_policy_task_property.get(
                        (policy, task, property_id),
                        0.0,
                    )
                    / rollout_count
                    for property_id in applicable_property_ids
                ]
                task_exposure_values.append(_mean(property_means))

            if not task_exposure_values:
                continue

            output_rows.append(
                {
                    "Policy": policy,
                    "SafetyCategory": category,
                    "SafetyViolationRate": applicable_rate_lookup.get((policy, category), 0.0),
                    "SafetyViolationRateAcrossAllRollouts": (
                        len(violated_episodes_by_policy_category.get((policy, category), set()))
                        / len(episodes_by_policy.get(policy, set()))
                        if episodes_by_policy.get(policy)
                        else 0.0
                    ),
                    "exposure_rate_per_category_per_task_per_policy": _mean(
                        task_exposure_values
                    ),
                }
            )
    return output_rows


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _plot_data_path(plot_data_dir: Path, plot_path: Path) -> Path:
    return plot_data_dir / f"{plot_path.stem}.csv"


def write_plot_data_csv(
    plot_data_dir: Path,
    plot_path: Path,
    rows: Sequence[Dict[str, Any]],
    columns: Sequence[str],
) -> Path:
    csv_path = _plot_data_path(plot_data_dir, plot_path)
    write_csv(csv_path, rows, columns)
    sibling_csv_path = plot_path.with_suffix(".csv")
    if sibling_csv_path.resolve() != csv_path.resolve():
        write_csv(sibling_csv_path, rows, columns)
    return csv_path


def _setup_matplotlib():
    cache_dir = Path("/p/safevla/pip_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_bubble_heatmap(
    *,
    heatmap_rows: Sequence[Dict[str, Any]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt = _setup_matplotlib()

    categories = sorted({str(row["PropertyCategory"]) for row in heatmap_rows})
    policies = sorted({str(row["policy"]) for row in heatmap_rows})
    category_to_y = {category: index for index, category in enumerate(categories)}
    policy_to_x = {policy: index for index, policy in enumerate(policies)}

    x_values = [policy_to_x[str(row["policy"])] for row in heatmap_rows]
    y_values = [category_to_y[str(row["PropertyCategory"])] for row in heatmap_rows]
    violation_values = [
        _to_float(row["normalizedViolationFrequency"]) for row in heatmap_rows
    ]
    exposure_values = [_to_float(row["meanExposureRate"]) for row in heatmap_rows]
    sizes = [35 + 1200 * value for value in violation_values]

    fig_width = max(8.5, 1.6 * len(policies) + 3.0)
    fig_height = max(6.8, 0.55 * len(categories) + 2.6)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    scatter = ax.scatter(
        x_values,
        y_values,
        s=sizes,
        c=exposure_values,
        cmap="viridis",
        edgecolor="white",
        linewidth=0.8,
        alpha=0.9,
    )

    ax.set_title("RQ2 Bubble Heat Map: Violation Frequency and Exposure")
    ax.set_xlabel("policy")
    ax.set_ylabel("PropertyCategory")
    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels(policies, rotation=20, ha="right")
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories)
    ax.set_xlim(-0.75, max(0.75, len(policies) - 0.25))
    ax.set_ylim(-0.5, len(categories) - 0.5)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.22)

    colorbar = fig.colorbar(scatter, ax=ax, pad=0.035, fraction=0.045)
    colorbar.set_label("mean exposureRate")

    legend_values = [0.25, 0.5, 0.75, 1.0]
    handles = [
        ax.scatter(
            [],
            [],
            s=35 + 1200 * value,
            color="#777777",
            alpha=0.55,
            edgecolor="white",
            linewidth=0.8,
        )
        for value in legend_values
    ]
    ax.legend(
        handles,
        [f"{value:.2f}" for value in legend_values],
        title="normalized violation frequency",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=len(legend_values),
        frameon=False,
        borderaxespad=0.0,
        columnspacing=2.4,
        handletextpad=1.2,
    )

    fig.subplots_adjust(left=0.24, right=0.86, bottom=0.28, top=0.9)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _annotate_bars(ax: Any, values: Sequence[float], *, float_values: bool = False) -> None:
    y_max = max(values) if values else 0.0
    offset = (y_max * 0.025) if y_max else 0.025
    for patch, value in zip(ax.patches, values):
        label = f"{value:.3f}" if float_values else f"{int(round(value))}"
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            patch.get_height() + offset,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _policy_bar_order_key(row: Dict[str, Any]) -> Tuple[int, str]:
    policy = str(row.get("policy") or "").lower()
    policy_suffix = policy.replace("_", "-").rsplit("-", 1)[-1]
    order = {
        "pt": 0,
        "to": 1,
        "tpt": 2,
    }.get(policy_suffix, 99)
    return order, policy


def save_policy_unsafe_bar_panels(
    *,
    policy_rows: Sequence[Dict[str, Any]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt = _setup_matplotlib()
    policy_rows = sorted(policy_rows, key=_policy_bar_order_key)

    policies = [str(row["policy"]) for row in policy_rows]
    total_unsafe_entries = [
        _to_float(row["totalUnsafeEntries"]) for row in policy_rows
    ]
    weighted_exposures = [_to_float(row["weightedExposure"]) for row in policy_rows]
    x_positions = list(range(len(policies)))

    fig_width = max(11.0, 1.35 * len(policies) + 7.0)
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, 5.4))
    panel_a, panel_b = axes

    panel_a.bar(x_positions, total_unsafe_entries, color="#c96f53")
    panel_a.set_title("Unsafe entry frequency by policy")
    panel_a.set_ylabel("Total unsafe entries")
    panel_a.set_xticks(x_positions)
    panel_a.set_xticklabels(policies, rotation=25, ha="right")
    panel_a.grid(axis="y", alpha=0.25)
    if total_unsafe_entries:
        panel_a.set_ylim(0, max(total_unsafe_entries) * 1.18 or 1)
    _annotate_bars(panel_a, total_unsafe_entries)

    panel_b.bar(x_positions, weighted_exposures, color="#5b8db8")
    panel_b.set_title("Average unsafe dwell by policy")
    panel_b.set_ylabel("Weighted exposure rate")
    panel_b.set_xticks(x_positions)
    panel_b.set_xticklabels(policies, rotation=25, ha="right")
    panel_b.grid(axis="y", alpha=0.25)
    if weighted_exposures:
        panel_b.set_ylim(0, max(weighted_exposures) * 1.2 or 0.05)
    _annotate_bars(panel_b, weighted_exposures, float_values=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_applicable_rate_bar_subplots(
    *,
    rate_rows: Sequence[Dict[str, Any]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt = _setup_matplotlib()

    policies = sorted(
        {str(row["policy"]) for row in rate_rows},
        key=lambda policy: _policy_bar_order_key({"policy": policy}),
    )
    categories = sorted({str(row["PropertyCategory"]) for row in rate_rows})
    if not policies or not categories:
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        ax.text(0.5, 0.5, "No applicable-rate rows", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        return

    row_lookup = {
        (str(row["policy"]), str(row["PropertyCategory"])): row
        for row in rate_rows
    }
    n_cols = min(3, len(policies))
    n_rows = (len(policies) + n_cols - 1) // n_cols
    fig_width = max(8.5, 5.2 * n_cols)
    fig_height = max(5.2, 4.8 * n_rows)
    fig, axes_grid = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height),
        sharey=True,
        squeeze=False,
    )
    axes = list(axes_grid.flat)
    x_positions = list(range(len(categories)))

    for ax, policy in zip(axes, policies):
        rates = [
            _to_float(
                row_lookup.get((policy, category), {}).get(
                    "violation_rate_per_property_category",
                    0.0,
                )
            )
            for category in categories
        ]
        applicable_counts = [
            _to_int(row_lookup.get((policy, category), {}).get("applicableRollouts", 0))
            for category in categories
        ]
        bars = ax.bar(x_positions, rates, color="#7b5ea7")

        for bar, rate, applicable_count in zip(bars, rates, applicable_counts):
            if applicable_count == 0:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{rate:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

        ax.set_title(policy)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(categories, rotation=35, ha="right")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.25)

    for ax in axes[len(policies):]:
        ax.set_visible(False)

    for ax in axes[::n_cols]:
        ax.set_ylabel("violation rollouts / applicable rollouts")

    fig.suptitle("Applicable Violation Rate per Property Category by Policy", y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_applicable_exposure_bar_subplots(
    *,
    exposure_rows: Sequence[Dict[str, Any]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt = _setup_matplotlib()

    policies = sorted(
        {str(row["policy"]) for row in exposure_rows},
        key=lambda policy: _policy_bar_order_key({"policy": policy}),
    )
    categories = sorted({str(row["PropertyCategory"]) for row in exposure_rows})
    if not policies or not categories:
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        ax.text(0.5, 0.5, "No applicable-exposure rows", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        return

    row_lookup = {
        (str(row["policy"]), str(row["PropertyCategory"])): row
        for row in exposure_rows
    }
    n_cols = min(3, len(policies))
    n_rows = (len(policies) + n_cols - 1) // n_cols
    fig_width = max(8.5, 5.2 * n_cols)
    fig_height = max(5.2, 4.8 * n_rows)
    fig, axes_grid = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height),
        sharey=True,
        squeeze=False,
    )
    axes = list(axes_grid.flat)
    x_positions = list(range(len(categories)))

    for ax, policy in zip(axes, policies):
        exposures = [
            _to_float(row_lookup.get((policy, category), {}).get("exposureRate", 0.0))
            for category in categories
        ]
        applicable_counts = [
            _to_int(row_lookup.get((policy, category), {}).get("applicableRollouts", 0))
            for category in categories
        ]
        bars = ax.bar(x_positions, exposures, color="#5b8db8")

        for bar, exposure, applicable_count in zip(bars, exposures, applicable_counts):
            if applicable_count == 0:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{exposure:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

        ax.set_title(policy)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(categories, rotation=35, ha="right")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.25)

    for ax in axes[len(policies):]:
        ax.set_visible(False)

    for ax in axes[::n_cols]:
        ax.set_ylabel("mean total exposure rate per applicable rollout")

    fig.suptitle("Applicable Total Exposure Rate per Property Category by Policy", y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build RQ2 category exposure and violation-count table."
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=DEFAULT_PROCESSED_ROOT,
        help=f"Root containing metrics CSVs. Default: {DEFAULT_PROCESSED_ROOT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for RQ2 output. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=DEFAULT_PLOT_DIR,
        help=f"Directory for RQ2 plots. Default: {DEFAULT_PLOT_DIR}",
    )
    parser.add_argument(
        "--plot-data-dir",
        type=Path,
        default=None,
        help="Directory for per-plot RQ2 CSV data. Default: <output-dir>/plotData",
    )
    parser.add_argument(
        "--suites-csv",
        type=Path,
        default=DEFAULT_SUITES_CSV,
        help=f"CSV mapping taskName to taskSuites. Default: {DEFAULT_SUITES_CSV}",
    )
    parser.add_argument(
        "--applicable-properties-csv",
        type=Path,
        default=DEFAULT_APPLICABLE_PROPERTIES_CSV,
        help=(
            "CSV mapping property applicability to tasks. "
            f"Default: {DEFAULT_APPLICABLE_PROPERTIES_CSV}"
        ),
    )
    parser.add_argument(
        "--pattern",
        default="metrics_*.csv",
        help="CSV glob to aggregate. Default: metrics_*.csv",
    )
    parser.add_argument(
        "--output-name",
        default="rq2_metrics.csv",
        help="Output CSV filename. Default: rq2_metrics.csv",
    )
    parser.add_argument(
        "--heatmap-data-name",
        default="rq2_bubble_heatmap_data.csv",
        help="Aggregated heatmap data CSV filename. Default: rq2_bubble_heatmap_data.csv",
    )
    parser.add_argument(
        "--heatmap-name",
        default="rq2_bubble_heatmap.png",
        help="Bubble heatmap filename. Default: rq2_bubble_heatmap.png",
    )
    parser.add_argument(
        "--policy-summary-name",
        default="rq2_policy_unsafe_summary.csv",
        help="Policy unsafe summary CSV filename. Default: rq2_policy_unsafe_summary.csv",
    )
    parser.add_argument(
        "--policy-bars-name",
        default="rq2_policy_unsafe_bar_panels.png",
        help="Two-panel policy bar chart filename. Default: rq2_policy_unsafe_bar_panels.png",
    )
    parser.add_argument(
        "--applicable-rate-name",
        default="rq2_applicable_category_violation_rates.csv",
        help=(
            "Applicability-adjusted category violation-rate CSV filename. "
            "Default: rq2_applicable_category_violation_rates.csv"
        ),
    )
    parser.add_argument(
        "--applicable-rate-bars-name",
        default="rq2_applicable_category_violation_rate_bars.png",
        help=(
            "Applicability-adjusted category violation-rate bar plot filename. "
            "Default: rq2_applicable_category_violation_rate_bars.png"
        ),
    )
    parser.add_argument(
        "--applicable-exposure-name",
        default="rq2_applicable_category_exposure_rates.csv",
        help=(
            "Applicability-adjusted category exposure-rate CSV filename. "
            "Default: rq2_applicable_category_exposure_rates.csv"
        ),
    )
    parser.add_argument(
        "--policy-category-balanced-name",
        default="rq2_policy_category_balanced_exposure.csv",
        help=(
            "Policy/category balanced safety and exposure CSV filename. "
            "Default: rq2_policy_category_balanced_exposure.csv"
        ),
    )
    parser.add_argument(
        "--applicable-exposure-bars-name",
        default="rq2_applicable_category_exposure_rate_bars.png",
        help=(
            "Applicability-adjusted category exposure-rate bar plot filename. "
            "Default: rq2_applicable_category_exposure_rate_bars.png"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_root = args.processed_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    plot_dir = args.plot_dir.expanduser().resolve()
    plot_data_dir = (
        args.plot_data_dir.expanduser().resolve()
        if args.plot_data_dir is not None
        else output_dir / "plotData"
    )
    suites_csv = args.suites_csv.expanduser().resolve()
    applicable_properties_csv = _resolve_applicable_properties_path(
        args.applicable_properties_csv.expanduser().resolve()
    )

    metric_paths = _metric_csvs(processed_root, args.pattern)
    if not metric_paths:
        raise SystemExit(f"No metrics CSV files found under {processed_root} matching {args.pattern!r}.")

    raw_rows = _read_rows(metric_paths)
    task_suites = load_task_suites(suites_csv)
    applicable_properties = load_applicable_properties(applicable_properties_csv)
    rq2_rows = build_rq2_rows(raw_rows, task_suites)
    heatmap_rows = build_heatmap_rows(raw_rows)
    policy_unsafe_rows = build_policy_unsafe_rows(rq2_rows)
    applicable_rate_rows = build_applicable_rate_rows(raw_rows, applicable_properties)
    applicable_exposure_rows = build_applicable_exposure_rows(raw_rows, applicable_properties)
    policy_category_balanced_rows = build_policy_category_balanced_rows(
        raw_rows,
        applicable_properties,
        applicable_rate_rows,
    )
    output_path = output_dir / args.output_name
    heatmap_data_path = output_dir / args.heatmap_data_name
    heatmap_path = plot_dir / args.heatmap_name
    policy_summary_path = output_dir / args.policy_summary_name
    policy_bars_path = plot_dir / args.policy_bars_name
    applicable_rate_path = output_dir / args.applicable_rate_name
    applicable_rate_bars_path = plot_dir / args.applicable_rate_bars_name
    applicable_exposure_path = output_dir / args.applicable_exposure_name
    policy_category_balanced_path = output_dir / args.policy_category_balanced_name
    applicable_exposure_bars_path = plot_dir / args.applicable_exposure_bars_name
    plot_data_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_path, rq2_rows, OUTPUT_COLUMNS)
    write_csv(heatmap_data_path, heatmap_rows, HEATMAP_COLUMNS)
    write_csv(policy_summary_path, policy_unsafe_rows, POLICY_UNSAFE_COLUMNS)
    write_csv(applicable_rate_path, applicable_rate_rows, APPLICABLE_RATE_COLUMNS)
    write_csv(
        applicable_exposure_path,
        applicable_exposure_rows,
        APPLICABLE_EXPOSURE_COLUMNS,
    )
    write_csv(
        policy_category_balanced_path,
        policy_category_balanced_rows,
        POLICY_CATEGORY_BALANCED_COLUMNS,
    )
    write_plot_data_csv(plot_data_dir, heatmap_path, heatmap_rows, HEATMAP_COLUMNS)
    write_plot_data_csv(plot_data_dir, policy_bars_path, policy_unsafe_rows, POLICY_UNSAFE_COLUMNS)
    write_plot_data_csv(
        plot_data_dir,
        applicable_rate_bars_path,
        applicable_rate_rows,
        APPLICABLE_RATE_COLUMNS,
    )
    write_plot_data_csv(
        plot_data_dir,
        applicable_exposure_bars_path,
        applicable_exposure_rows,
        APPLICABLE_EXPOSURE_COLUMNS,
    )
    save_bubble_heatmap(heatmap_rows=heatmap_rows, output_path=heatmap_path)
    save_policy_unsafe_bar_panels(
        policy_rows=policy_unsafe_rows,
        output_path=policy_bars_path,
    )
    save_applicable_rate_bar_subplots(
        rate_rows=applicable_rate_rows,
        output_path=applicable_rate_bars_path,
    )
    save_applicable_exposure_bar_subplots(
        exposure_rows=applicable_exposure_rows,
        output_path=applicable_exposure_bars_path,
    )

    print(f"Read {len(metric_paths)} metrics CSV file(s).")
    print(f"Raw rows: {len(raw_rows)}")
    print(f"RQ2 rows: {len(rq2_rows)}")
    print(f"RQ2 heatmap rows: {len(heatmap_rows)}")
    print(f"RQ2 policy unsafe rows: {len(policy_unsafe_rows)}")
    print(f"RQ2 applicable-rate rows: {len(applicable_rate_rows)}")
    print(f"RQ2 applicable-exposure rows: {len(applicable_exposure_rows)}")
    print(f"RQ2 policy-category balanced rows: {len(policy_category_balanced_rows)}")
    print(f"Wrote: {output_path}")
    print(f"Wrote: {heatmap_data_path}")
    print(f"Wrote: {policy_summary_path}")
    print(f"Wrote: {applicable_rate_path}")
    print(f"Wrote: {applicable_exposure_path}")
    print(f"Wrote: {policy_category_balanced_path}")
    print(f"Wrote plot data CSVs under: {plot_data_dir}")
    print(f"Wrote: {heatmap_path}")
    print(f"Wrote: {policy_bars_path}")
    print(f"Wrote: {applicable_rate_bars_path}")
    print(f"Wrote: {applicable_exposure_bars_path}")


if __name__ == "__main__":
    main()
