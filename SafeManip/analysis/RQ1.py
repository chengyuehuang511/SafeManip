from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


ANALYSIS_DIR = Path(__file__).resolve().parent
DEFAULT_PROCESSED_ROOT = ANALYSIS_DIR / "processedData"
DEFAULT_RAW_ROOT = ANALYSIS_DIR / "rawData"
DEFAULT_OUTPUT_DIR = DEFAULT_PROCESSED_ROOT / "RQ1"
DEFAULT_PLOT_DIR = ANALYSIS_DIR / "plots" / "RQ1"
DEFAULT_SUITES_CSV = ANALYSIS_DIR / "idSuites.csv"
DEFAULT_POLICY_CSV = ANALYSIS_DIR / "idPolicy.csv"


POLICY_ID_COLORS = {
    "gen": "#2f6f9f",
    "ft": "#c96f53",
    "unmapped": "#777777",
}


RAW_COLUMNS = (
    "task",
    "episode",
    "policy",
    "property",
    "category",
    "violation_count_per_property",
    "violation_duration_per_property",
    "exposure_rate",
    "task_success",
    "safety_satisfaction",
    "safe_succes_per_property",
)


OPTIONAL_RAW_COLUMNS = (
    "violations_per_skill_onset",
)


MASTER_RAW_COLUMNS = RAW_COLUMNS + OPTIONAL_RAW_COLUMNS


TASK_LIST_COLUMNS = (
    "task",
)


EPISODE_COLUMNS = (
    "task",
    "policy",
    "episode",
    "task_success",
    "strict_safety_all",
    "mean_safety_rate",
    "num_properties",
    "num_properties_violated",
    "total_violation_count",
    "total_violation_duration",
    "mean_exposure_rate",
    "max_exposure_rate",
    "safe_success",
)


SUMMARY_COLUMNS = (
    "task",
    "policy",
    "n_episodes",
    "success_rate",
    "strict_safety_rate",
    "mean_safety_rate",
    "safe_success_rate",
    "unsafe_success_gap",
    "avg_violation_count",
    "avg_violation_duration",
)


CATEGORY_SUITE_COLUMNS = (
    "category",
    "taskSuites",
    "n_episodes",
    "success_rate",
    "strict_safety_rate",
)


OUTCOME_COLUMNS = (
    "taskSuites",
    "policy",
    "n_episodes",
    "overall_task_success_rate",
    "overall_safety_satisfaction_rate",
    "safe_success_rate",
    "unsafe_success_rate",
    "failed_safe_rate",
    "failed_unsafe_rate",
    "successful_safe_rate",
    "successful_unsafe_rate",
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


def _policy_from_metric_path(path: Path) -> str:
    prefix = "metrics_"
    stem = path.stem
    if stem.startswith(prefix) and len(stem) > len(prefix):
        return stem[len(prefix):]
    return stem or "unknown_policy"


def _metric_csvs(processed_root: Path, pattern: str) -> List[Path]:
    rq1_dir = (processed_root / "RQ1").resolve()
    paths = []
    for path in sorted(processed_root.rglob(pattern)):
        if not path.is_file():
            continue
        try:
            path.resolve().relative_to(rq1_dir)
            continue
        except ValueError:
            pass
        paths.append(path)
    return paths


def _read_rows(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        metric_policy = _policy_from_metric_path(path)
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = [column for column in RAW_COLUMNS if column not in (reader.fieldnames or [])]
            if missing:
                raise SystemExit(f"{path} is missing required columns: {missing}")
            for row in reader:
                metric_row = {column: row.get(column, "") for column in RAW_COLUMNS}
                for column in OPTIONAL_RAW_COLUMNS:
                    metric_row[column] = row.get(column, "")
                metric_row["_metric_policy"] = metric_policy
                rows.append(metric_row)
    return rows


def warn_mismatched_metric_paths(paths: Iterable[Path]) -> None:
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            tasks = sorted({row.get("task", "") for row in reader if row.get("task")})
        if len(tasks) == 1 and tasks[0] != path.parent.name:
            print(
                "Warning: metrics file task does not match parent folder: "
                f"{path} has task={tasks[0]!r}, parent={path.parent.name!r}"
            )


def discover_raw_tasks(raw_root: Path) -> List[str]:
    if not raw_root.exists():
        return []

    tasks = set()
    for policy_dir in raw_root.iterdir():
        if not policy_dir.is_dir():
            continue
        for task_dir in policy_dir.iterdir():
            if task_dir.is_dir():
                tasks.add(task_dir.name)
    return sorted(tasks)


def build_task_list_rows(
    raw_rows: Sequence[Dict[str, Any]],
    *,
    raw_root: Path,
) -> List[Dict[str, Any]]:
    tasks = {str(row.get("task") or "").strip() for row in raw_rows}
    tasks.update(discover_raw_tasks(raw_root))
    tasks.discard("")
    return [{"task": task} for task in sorted(tasks)]


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


def load_policy_ids(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit(f"Policy ID CSV is empty: {path}")
        normalized_fields = {field.lower(): field for field in reader.fieldnames}
        policy_field = normalized_fields.get("policy")
        id_field = normalized_fields.get("id")
        if not policy_field or not id_field:
            raise SystemExit(f"Expected columns policy and id in: {path}")

        policy_ids = {}
        for row in reader:
            policy = str(row.get(policy_field) or "").strip()
            policy_id = str(row.get(id_field) or "").strip()
            if policy:
                policy_ids[policy] = policy_id or "unmapped"
        return policy_ids


def _group_by(rows: Iterable[Dict[str, Any]], keys: Sequence[str]) -> Dict[Tuple[Any, ...], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in keys)].append(row)
    return dict(grouped)


def build_episode_rows(raw_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    episode_rows: List[Dict[str, Any]] = []
    for (task, policy, episode), rows in sorted(_group_by(raw_rows, ("task", "policy", "episode")).items()):
        task_success = max(_to_int(row["task_success"]) for row in rows)
        safety_values = [_to_float(row["safety_satisfaction"]) for row in rows]
        exposure_values = [_to_float(row["exposure_rate"]) for row in rows]
        strict_safety_all = min(_to_int(row["safety_satisfaction"]) for row in rows)

        episode_rows.append(
            {
                "task": task,
                "policy": policy,
                "episode": episode,
                "task_success": task_success,
                "strict_safety_all": strict_safety_all,
                "mean_safety_rate": _mean(safety_values),
                "num_properties": len(rows),
                "num_properties_violated": sum(1 - _to_int(row["safety_satisfaction"]) for row in rows),
                "total_violation_count": sum(_to_int(row["violation_count_per_property"]) for row in rows),
                "total_violation_duration": sum(_to_int(row["violation_duration_per_property"]) for row in rows),
                "mean_exposure_rate": _mean(exposure_values),
                "max_exposure_rate": max(exposure_values) if exposure_values else 0.0,
                "safe_success": task_success * strict_safety_all,
            }
        )
    return episode_rows


def build_summary_rows(episode_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summary_rows: List[Dict[str, Any]] = []
    for (task, policy), rows in sorted(_group_by(episode_rows, ("task", "policy")).items()):
        n_episodes = len(rows)
        success_count = sum(_to_float(row["task_success"]) for row in rows)
        strict_safety_count = sum(_to_float(row["strict_safety_all"]) for row in rows)
        safe_success_count = sum(_to_float(row["safe_success"]) for row in rows)
        success_rate = success_count / n_episodes if n_episodes else 0.0
        safe_success_rate = safe_success_count / n_episodes if n_episodes else 0.0
        summary_rows.append(
            {
                "task": task,
                "policy": policy,
                "n_episodes": n_episodes,
                "success_count": success_count,
                "success_rate": success_rate,
                "strict_safety_count": strict_safety_count,
                "strict_safety_rate": strict_safety_count / n_episodes if n_episodes else 0.0,
                "mean_safety_rate": _mean([_to_float(row["mean_safety_rate"]) for row in rows]),
                "safe_success_count": safe_success_count,
                "safe_success_rate": safe_success_rate,
                "unsafe_success_gap": success_rate - safe_success_rate,
                "avg_violation_count": _mean([_to_float(row["total_violation_count"]) for row in rows]),
                "avg_violation_duration": _mean([_to_float(row["total_violation_duration"]) for row in rows]),
            }
        )
    return summary_rows


def build_category_summary_rows(
    raw_rows: Sequence[Dict[str, Any]],
    task_suites: Dict[str, str],
) -> List[Dict[str, Any]]:
    category_episode_rows = []
    for (task, policy, episode, category), rows in sorted(
        _group_by(raw_rows, ("task", "policy", "episode", "category")).items()
    ):
        category = category or "Uncategorized"
        category_episode_rows.append(
            {
                "task": task,
                "taskSuites": task_suites.get(task, "UnmappedSuite"),
                "policy": policy,
                "episode": episode,
                "category": category,
                "task_success": max(_to_int(row["task_success"]) for row in rows),
                "strict_safety_category": min(
                    _to_int(row["safety_satisfaction"]) for row in rows
                ),
            }
        )

    category_summary_rows = []
    for (category, task_suite), rows in sorted(
        _group_by(category_episode_rows, ("category", "taskSuites")).items()
    ):
        n_episodes = len(rows)
        success_count = sum(_to_float(row["task_success"]) for row in rows)
        strict_safety_count = sum(_to_float(row["strict_safety_category"]) for row in rows)
        category_summary_rows.append(
            {
                "category": category,
                "taskSuites": task_suite,
                "n_episodes": n_episodes,
                "success_count": success_count,
                "success_rate": success_count / n_episodes if n_episodes else 0.0,
                "strict_safety_count": strict_safety_count,
                "strict_safety_rate": strict_safety_count / n_episodes if n_episodes else 0.0,
            }
        )
    return category_summary_rows


def build_plot1_category_rows(raw_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    category_episode_rows = []
    for (metric_policy, task, episode, category), rows in sorted(
        _group_by(raw_rows, ("_metric_policy", "task", "episode", "category")).items()
    ):
        category_episode_rows.append(
            {
                "policy": metric_policy or "unknown_policy",
                "task": task,
                "episode": episode,
                "category": category or "Uncategorized",
                "task_success": max(_to_int(row["task_success"]) for row in rows),
                "strict_safety_category": min(
                    _to_int(row["safety_satisfaction"]) for row in rows
                ),
            }
        )

    plot_rows = []
    for (policy, task, category), rows in sorted(
        _group_by(category_episode_rows, ("policy", "task", "category")).items()
    ):
        n_episodes = len(rows)
        success_count = sum(_to_float(row["task_success"]) for row in rows)
        category_violation_count = sum(
            1 - _to_int(row["strict_safety_category"]) for row in rows
        )
        plot_rows.append(
            {
                "policy": policy,
                "task": task,
                "category": category,
                "n_episodes": n_episodes,
                "success_count": success_count,
                "success_rate": success_count / n_episodes if n_episodes else 0.0,
                "category_violation_count": category_violation_count,
                "category_violation_rate": (
                    category_violation_count / n_episodes if n_episodes else 0.0
                ),
            }
        )
    return plot_rows


def build_plot1_overall_rows(raw_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return build_plot1_category_rows(raw_rows)


def build_plot1_suite_rows(
    raw_rows: Sequence[Dict[str, Any]],
    task_suites: Dict[str, str],
) -> List[Dict[str, Any]]:
    episode_rows = []
    for (metric_policy, task, episode), rows in sorted(
        _group_by(raw_rows, ("_metric_policy", "task", "episode")).items()
    ):
        episode_rows.append(
            {
                "policy": metric_policy or "unknown_policy",
                "taskSuites": task_suites.get(task, "UnmappedSuite"),
                "task": task,
                "episode": episode,
                "task_success": max(_to_int(row["task_success"]) for row in rows),
                "property_violation_rate": _mean(
                    [1 - _to_int(row["safety_satisfaction"]) for row in rows]
                ),
            }
        )

    plot_rows = []
    for (policy, task_suite), rows in sorted(
        _group_by(episode_rows, ("policy", "taskSuites")).items()
    ):
        n_episodes = len(rows)
        success_count = sum(_to_float(row["task_success"]) for row in rows)
        property_violation_rate_sum = sum(
            _to_float(row["property_violation_rate"]) for row in rows
        )
        plot_rows.append(
            {
                "policy": policy,
                "taskSuites": task_suite,
                "n_episodes": n_episodes,
                "success_count": success_count,
                "success_rate": success_count / n_episodes if n_episodes else 0.0,
                "property_violation_rate_sum": property_violation_rate_sum,
                "property_violation_rate": _mean(
                    [_to_float(row["property_violation_rate"]) for row in rows]
                ),
            }
        )
    return plot_rows


def _episode_outcome_flags(row: Dict[str, Any]) -> Dict[str, int]:
    task_success = _to_int(row["task_success"])
    strict_safety = _to_int(row["strict_safety_all"])
    return {
        "failed_safe": int(task_success == 0 and strict_safety == 1),
        "failed_unsafe": int(task_success == 0 and strict_safety == 0),
        "successful_safe": int(task_success == 1 and strict_safety == 1),
        "successful_unsafe": int(task_success == 1 and strict_safety == 0),
    }


def build_outcome_rows(
    episode_rows: Sequence[Dict[str, Any]],
    task_suites: Dict[str, str],
) -> List[Dict[str, Any]]:
    suite_episode_rows = []
    for row in episode_rows:
        suite_episode_rows.append(
            {
                **row,
                "taskSuites": task_suites.get(str(row.get("task", "")), "UnmappedSuite"),
            }
        )

    grouped_rows: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in suite_episode_rows:
        grouped_rows.setdefault((row["taskSuites"], row["policy"]), []).append(row)
        grouped_rows.setdefault(("ALL_SUITES", row["policy"]), []).append(row)

    outcome_rows = []
    for (suite, policy), rows in sorted(grouped_rows.items()):
        n_episodes = len(rows)
        flags = [_episode_outcome_flags(row) for row in rows]
        failed_safe_count = sum(flag["failed_safe"] for flag in flags)
        failed_unsafe_count = sum(flag["failed_unsafe"] for flag in flags)
        successful_safe_count = sum(flag["successful_safe"] for flag in flags)
        successful_unsafe_count = sum(flag["successful_unsafe"] for flag in flags)
        successful_safe_rate = successful_safe_count / n_episodes if n_episodes else 0.0
        successful_unsafe_rate = successful_unsafe_count / n_episodes if n_episodes else 0.0
        task_success_count = sum(_to_float(row["task_success"]) for row in rows)
        strict_safety_count = sum(_to_float(row["strict_safety_all"]) for row in rows)
        outcome_rows.append(
            {
                "taskSuites": suite,
                "policy": policy,
                "n_episodes": n_episodes,
                "overall_task_success_count": task_success_count,
                "overall_task_success_rate": task_success_count / n_episodes if n_episodes else 0.0,
                "overall_safety_satisfaction_count": strict_safety_count,
                "overall_safety_satisfaction_rate": (
                    strict_safety_count / n_episodes if n_episodes else 0.0
                ),
                "safe_success_count": successful_safe_count,
                "safe_success_rate": successful_safe_rate,
                "unsafe_success_count": successful_unsafe_count,
                "unsafe_success_rate": successful_unsafe_rate,
                "failed_safe_count": failed_safe_count,
                "failed_safe_rate": failed_safe_count / n_episodes if n_episodes else 0.0,
                "failed_unsafe_count": failed_unsafe_count,
                "failed_unsafe_rate": failed_unsafe_count / n_episodes if n_episodes else 0.0,
                "successful_safe_count": successful_safe_count,
                "successful_safe_rate": successful_safe_rate,
                "successful_unsafe_count": successful_unsafe_count,
                "successful_unsafe_rate": successful_unsafe_rate,
                "overall_violation_count": failed_unsafe_count + successful_unsafe_count,
                "overall_violation_rate": (
                    (failed_unsafe_count + successful_unsafe_count) / n_episodes
                    if n_episodes
                    else 0.0
                ),
            }
        )
    return outcome_rows


def build_category_outcome_rows(raw_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    category_episode_rows = []
    for (task, policy, episode, category), rows in sorted(
        _group_by(raw_rows, ("task", "policy", "episode", "category")).items()
    ):
        category_episode_rows.append(
            {
                "category": _normalize_category(category),
                "policy": policy,
                "task": task,
                "episode": episode,
                "task_success": max(_to_int(row["task_success"]) for row in rows),
                "strict_safety_category": min(
                    _to_int(row["safety_satisfaction"]) for row in rows
                ),
            }
        )

    outcome_rows = []
    for (category, policy), rows in sorted(
        _group_by(category_episode_rows, ("category", "policy")).items()
    ):
        flags = []
        for row in rows:
            task_success = _to_int(row["task_success"])
            strict_safety = _to_int(row["strict_safety_category"])
            flags.append(
                {
                    "successful_unsafe": int(task_success == 1 and strict_safety == 0),
                }
            )
        n_episodes = len(rows)
        successful_unsafe_count = sum(flag["successful_unsafe"] for flag in flags)
        task_success_count = sum(_to_float(row["task_success"]) for row in rows)
        outcome_rows.append(
            {
                "category": category,
                "policy": policy,
                "n_episodes": n_episodes,
                "overall_task_success_count": task_success_count,
                "overall_task_success_rate": task_success_count / n_episodes if n_episodes else 0.0,
                "successful_unsafe_count": successful_unsafe_count,
                "successful_unsafe_rate": (
                    successful_unsafe_count / n_episodes if n_episodes else 0.0
                ),
            }
        )
    return outcome_rows


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _columns_for_rows(rows: Sequence[Dict[str, Any]]) -> List[str]:
    columns: List[str] = []
    for row in rows:
        for column in row:
            if column.startswith("_") or column in columns:
                continue
            columns.append(column)
    return columns


def _plot_data_path(plot_data_dir: Path, plot_path: Path) -> Path:
    return plot_data_dir / f"{plot_path.stem}.csv"


def write_plot_data_csv(
    plot_data_dir: Path,
    plot_path: Path,
    rows: Sequence[Dict[str, Any]],
    columns: Sequence[str] | None = None,
) -> Path:
    csv_path = _plot_data_path(plot_data_dir, plot_path)
    if columns is None:
        columns = _columns_for_rows(rows)
    write_csv(csv_path, rows, columns)
    sibling_csv_path = plot_path.with_suffix(".csv")
    if sibling_csv_path.resolve() != csv_path.resolve():
        write_csv(sibling_csv_path, rows, columns)
    return csv_path


def warn_missing_plot_data(
    plot_paths: Sequence[Path],
    plot_data_dir: Path,
) -> None:
    missing = []
    for plot_path in plot_paths:
        csv_path = _plot_data_path(plot_data_dir, plot_path)
        if not csv_path.exists():
            missing.append((plot_path, csv_path))

    if not missing:
        return

    print("Warning: some RQ1 plots from this run do not have matching plotData CSVs:")
    for plot_path, csv_path in missing:
        print(f"  plot={plot_path}")
        print(f"  csv ={csv_path}")


def warn_existing_plot_data_mismatches(
    plot_dir: Path,
    plot_data_dir: Path,
) -> None:
    if not plot_dir.exists():
        return

    missing = []
    for plot_path in sorted(plot_dir.rglob("*.png")):
        relative_csv = plot_path.relative_to(plot_dir).with_suffix(".csv")
        csv_path = plot_data_dir / relative_csv
        if not csv_path.exists():
            missing.append((plot_path, csv_path))

    if not missing:
        return

    print("Warning: existing RQ1 PNG files without matching plotData CSVs:")
    for plot_path, csv_path in missing:
        print(f"  plot={plot_path}")
        print(f"  csv ={csv_path}")


def _policy_outcome_rows(outcome_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    policy_rows = [
        row for row in outcome_rows if str(row.get("taskSuites", "")) == "ALL_SUITES"
    ]
    if policy_rows:
        return sorted(policy_rows, key=lambda row: str(row.get("policy", "")))

    rows_by_policy = []
    for policy, rows in sorted(_group_by(outcome_rows, ("policy",)).items()):
        n_episodes = sum(_to_int(row["n_episodes"]) for row in rows)
        if not n_episodes:
            continue
        failed_safe_count = sum(_to_float(row.get("failed_safe_count", 0)) for row in rows)
        failed_unsafe_count = sum(_to_float(row.get("failed_unsafe_count", 0)) for row in rows)
        successful_safe_count = sum(_to_float(row.get("successful_safe_count", 0)) for row in rows)
        successful_unsafe_count = sum(_to_float(row.get("successful_unsafe_count", 0)) for row in rows)
        success_count = successful_safe_count + successful_unsafe_count
        strict_safety_count = failed_safe_count + successful_safe_count
        rows_by_policy.append(
            {
                "taskSuites": "ALL_SUITES",
                "policy": policy,
                "n_episodes": n_episodes,
                "overall_task_success_count": success_count,
                "overall_task_success_rate": success_count / n_episodes,
                "overall_safety_satisfaction_count": strict_safety_count,
                "overall_safety_satisfaction_rate": strict_safety_count / n_episodes,
                "safe_success_count": successful_safe_count,
                "safe_success_rate": successful_safe_count / n_episodes,
                "unsafe_success_count": successful_unsafe_count,
                "unsafe_success_rate": successful_unsafe_count / n_episodes,
                "failed_safe_count": failed_safe_count,
                "failed_safe_rate": failed_safe_count / n_episodes,
                "failed_unsafe_count": failed_unsafe_count,
                "failed_unsafe_rate": failed_unsafe_count / n_episodes,
                "successful_safe_count": successful_safe_count,
                "successful_safe_rate": successful_safe_count / n_episodes,
                "successful_unsafe_count": successful_unsafe_count,
                "successful_unsafe_rate": successful_unsafe_count / n_episodes,
                "overall_violation_count": failed_unsafe_count + successful_unsafe_count,
                "overall_violation_rate": (failed_unsafe_count + successful_unsafe_count) / n_episodes,
            }
        )
    return rows_by_policy


def _plot10_rows(outcome_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for row in _policy_outcome_rows(outcome_rows):
        n_episodes = _to_float(row["n_episodes"])
        unsafe_success_count = _to_float(row.get("unsafe_success_count", row.get("successful_unsafe_count", 0)))
        rows.append(
            {
                "policy": row.get("policy", ""),
                "n_episodes": n_episodes,
                "overall_task_success_count": row.get("overall_task_success_count", ""),
                "overall_task_success_rate": row.get("overall_task_success_rate", ""),
                "unsafe_success_count": unsafe_success_count,
                "unsafe_success_rate": unsafe_success_count / n_episodes if n_episodes else 0.0,
            }
        )
    return rows


def _plot10_1_rows(outcome_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for row in _policy_outcome_rows(outcome_rows):
        success_count = _to_float(row.get("overall_task_success_count", 0))
        unsafe_success_count = _to_float(row.get("unsafe_success_count", row.get("successful_unsafe_count", 0)))
        rows.append(
            {
                "policy": row.get("policy", ""),
                "n_episodes": row.get("n_episodes", ""),
                "overall_task_success_count": success_count,
                "overall_task_success_rate": row.get("overall_task_success_rate", ""),
                "unsafe_success_count": unsafe_success_count,
                "unsafe_success_per_success": (
                    unsafe_success_count / success_count if success_count else 0.0
                ),
            }
        )
    return rows


def _plot11_rows(
    raw_rows: Sequence[Dict[str, Any]],
    outcome_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    ratio_values_by_policy: Dict[str, List[float]] = defaultdict(list)
    for row in raw_rows:
        value = row.get("violations_per_skill_onset", "")
        if value in (None, ""):
            continue
        ratio_values_by_policy[str(row.get("policy") or "unknown_policy")].append(_to_float(value))

    rows = []
    for row in _policy_outcome_rows(outcome_rows):
        policy = str(row.get("policy") or "unknown_policy")
        rows.append(
            {
                "policy": policy,
                "overall_task_success_count": row.get("overall_task_success_count", ""),
                "overall_task_success_rate": row.get("overall_task_success_rate", ""),
                "avg_violations_per_skill_onset": _mean(ratio_values_by_policy[policy]),
                "n_ratio_rows": len(ratio_values_by_policy[policy]),
            }
        )
    return rows


def _setup_matplotlib(plot_dir: Path):
    cache_dir = Path("/p/safevla/pip_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _policy_id(policy: Any, policy_ids: Dict[str, str]) -> str:
    return policy_ids.get(str(policy or ""), "unmapped")


def _policy_id_color(policy_id: str) -> str:
    return POLICY_ID_COLORS.get(policy_id, "#777777")


def _policy_id_order(policy_id: str) -> Tuple[int, str]:
    preferred_order = {
        "gen": 0,
        "ft": 1,
        "unmapped": 99,
    }
    return preferred_order.get(policy_id, 50), policy_id


def _add_policy_id_legend(plt: Any, ax: Any, policy_ids_present: Sequence[str]) -> None:
    ordered_ids = sorted(set(policy_ids_present), key=_policy_id_order)
    if not ordered_ids:
        return

    handles = [
        ax.scatter(
            [],
            [],
            s=70,
            color=_policy_id_color(policy_id),
            edgecolor="white",
            label=policy_id,
        )
        for policy_id in ordered_ids
    ]
    ax.legend(handles=handles, title="policy id", frameon=False, loc="lower right")


def _point_label(row: Dict[str, Any]) -> str:
    return f"{row.get('task', '')}\n{row.get('policy', '')}"


def _category_point_label(row: Dict[str, Any]) -> str:
    return f"{row.get('category', '')}\n{row.get('taskSuites', '')}"


def _plot1_point_label(row: Dict[str, Any]) -> str:
    return str(row.get("task", ""))


def _save_scatter(
    *,
    plt: Any,
    rows: Sequence[Dict[str, Any]],
    x_key: str,
    y_key: str,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    for row in rows:
        x = _to_float(row[x_key])
        y = _to_float(row[y_key])
        ax.scatter(x, y, s=90, color="#2f6f9f", edgecolor="white", linewidth=0.8)
        ax.annotate(_point_label(row), (x, y), xytext=(6, 6), textcoords="offset points", fontsize=8)

    ax.set_title(title)
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _save_plot1_category_scatter(
    *,
    plt: Any,
    rows: Sequence[Dict[str, Any]],
    policy: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    categories = sorted({row.get("category", "") for row in rows})
    palette = plt.get_cmap("tab20")
    colors = {
        category: palette(index % palette.N)
        for index, category in enumerate(categories)
    }

    for category in categories:
        category_rows = [row for row in rows if row.get("category", "") == category]
        x_values = [_to_float(row["success_rate"]) for row in category_rows]
        y_values = [_to_float(row["category_violation_rate"]) for row in category_rows]
        ax.scatter(
            x_values,
            y_values,
            s=82,
            color=colors[category],
            edgecolor="white",
            linewidth=0.7,
            label=category,
        )

    ax.set_title(f"Success Rate vs Property Category Violation Rate: {policy}")
    ax.set_xlabel("success_rate")
    ax.set_ylabel("category_violation_rate")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, alpha=0.25)
    if categories:
        ax.legend(title="property category", fontsize=7, title_fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _save_plot1_category_scatters_by_policy(
    *,
    plt: Any,
    rows: Sequence[Dict[str, Any]],
    plot_dir: Path,
) -> List[Path]:
    stale_patterns = (
        "01_main_scatter_success_vs_strict_safety*.png",
        "01_main_scatter_success_vs_category_violation*.png",
    )
    for stale_pattern in stale_patterns:
        for stale_path in plot_dir.glob(stale_pattern):
            stale_path.unlink()

    policies = sorted({str(row.get("policy", "")) for row in rows})
    output_paths = []
    for policy in policies:
        policy_rows = [row for row in rows if str(row.get("policy", "")) == policy]
        output_path = plot_dir / (
            f"01_main_scatter_success_vs_category_violation_{_safe_filename(policy)}.png"
        )
        _save_plot1_category_scatter(
            plt=plt,
            rows=policy_rows,
            policy=policy,
            output_path=output_path,
        )
        output_paths.append(output_path)
    return output_paths


def _save_plot1_overall_scatter(
    *,
    plt: Any,
    rows: Sequence[Dict[str, Any]],
    policy: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    x_values = [_to_float(row["success_rate"]) for row in rows]
    y_values = [_to_float(row["category_violation_rate"]) for row in rows]
    ax.scatter(
        x_values,
        y_values,
        s=82,
        color="#2f6f9f",
        edgecolor="white",
        linewidth=0.7,
    )

    ax.set_title(f"Success Rate vs Category Violation Rate: {policy}")
    ax.set_xlabel("success_rate")
    ax.set_ylabel("category_violation_rate")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _save_plot1_overall_scatters_by_policy(
    *,
    plt: Any,
    rows: Sequence[Dict[str, Any]],
    plot_dir: Path,
) -> List[Path]:
    stale_patterns = (
        "01_1_scatter_success_vs_property_violation*.png",
        "01_1_scatter_success_vs_category_violation*.png",
    )
    for stale_pattern in stale_patterns:
        for stale_path in plot_dir.glob(stale_pattern):
            stale_path.unlink()

    policies = sorted({str(row.get("policy", "")) for row in rows})
    output_paths = []
    for policy in policies:
        policy_rows = [row for row in rows if str(row.get("policy", "")) == policy]
        output_path = plot_dir / (
            f"01_1_scatter_success_vs_category_violation_{_safe_filename(policy)}.png"
        )
        _save_plot1_overall_scatter(
            plt=plt,
            rows=policy_rows,
            policy=policy,
            output_path=output_path,
        )
        output_paths.append(output_path)
    return output_paths


def _save_plot1_suite_scatter(
    *,
    plt: Any,
    rows: Sequence[Dict[str, Any]],
    policy: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    suites = sorted({row.get("taskSuites", "") for row in rows})
    palette = plt.get_cmap("tab10")
    colors = {
        suite: palette(index % palette.N)
        for index, suite in enumerate(suites)
    }

    for suite in suites:
        suite_rows = [row for row in rows if row.get("taskSuites", "") == suite]
        x_values = [_to_float(row["success_rate"]) for row in suite_rows]
        y_values = [_to_float(row["property_violation_rate"]) for row in suite_rows]
        ax.scatter(
            x_values,
            y_values,
            s=90,
            color=colors[suite],
            edgecolor="white",
            linewidth=0.7,
            label=suite,
        )

    ax.set_title(f"Success Rate vs Task Suite Property Violation Rate: {policy}")
    ax.set_xlabel("success_rate")
    ax.set_ylabel("task_suite_property_violation_rate")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, alpha=0.25)
    if suites:
        ax.legend(title="taskSuites", fontsize=7, title_fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _save_plot1_suite_scatters_by_policy(
    *,
    plt: Any,
    rows: Sequence[Dict[str, Any]],
    plot_dir: Path,
) -> List[Path]:
    for stale_path in plot_dir.glob("01_2_scatter_success_vs_suite_property_violation*.png"):
        stale_path.unlink()

    policies = sorted({str(row.get("policy", "")) for row in rows})
    output_paths = []
    for policy in policies:
        policy_rows = [row for row in rows if str(row.get("policy", "")) == policy]
        output_path = plot_dir / (
            f"01_2_scatter_success_vs_suite_property_violation_{_safe_filename(policy)}.png"
        )
        _save_plot1_suite_scatter(
            plt=plt,
            rows=policy_rows,
            policy=policy,
            output_path=output_path,
        )
        output_paths.append(output_path)
    return output_paths


def _save_category_scatter(
    *,
    plt: Any,
    rows: Sequence[Dict[str, Any]],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    categories = sorted({row.get("category", "") for row in rows})
    palette = plt.get_cmap("tab20")
    colors = {
        category: palette(index % palette.N)
        for index, category in enumerate(categories)
    }

    for category in categories:
        category_rows = [row for row in rows if row.get("category", "") == category]
        x_values = [_to_float(row["success_rate"]) for row in category_rows]
        y_values = [_to_float(row["strict_safety_rate"]) for row in category_rows]
        ax.scatter(
            x_values,
            y_values,
            s=75,
            color=colors[category],
            edgecolor="white",
            linewidth=0.7,
            label=category,
        )
        for row, x, y in zip(category_rows, x_values, y_values):
            ax.annotate(
                _category_point_label(row),
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=7,
            )

    ax.set_title("Average Success Rate vs Average Strict Safety Rate by Category and Task Suite")
    ax.set_xlabel("success_rate")
    ax.set_ylabel("strict_safety_rate")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, alpha=0.25)
    if categories:
        ax.legend(title="category", fontsize=7, title_fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _save_paired_bar(
    *,
    plt: Any,
    rows: Sequence[Dict[str, Any]],
    output_path: Path,
) -> None:
    labels = [_point_label(row) for row in rows]
    x_positions = list(range(len(rows)))
    width = 0.36
    success = [_to_float(row["success_rate"]) for row in rows]
    safe_success = [_to_float(row["safe_success_rate"]) for row in rows]

    fig_width = max(7.0, 1.2 * len(rows))
    fig, ax = plt.subplots(figsize=(fig_width, 5.0))
    ax.bar([x - width / 2 for x in x_positions], success, width, label="success_rate", color="#5b8db8")
    ax.bar(
        [x + width / 2 for x in x_positions],
        safe_success,
        width,
        label="safe_success_rate",
        color="#6f9d63",
    )

    ax.set_title("Success Rate vs Safe Success Rate")
    ax.set_ylabel("rate")
    ax.set_ylim(0, 1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _save_episode_boxplot(
    *,
    plt: Any,
    rows: Sequence[Dict[str, Any]],
    output_path: Path,
) -> None:
    grouped = {
        "0": [_to_float(row["mean_safety_rate"]) for row in rows if str(row["task_success"]) == "0"],
        "1": [_to_float(row["mean_safety_rate"]) for row in rows if str(row["task_success"]) == "1"],
    }
    labels = [label for label, values in grouped.items() if values]
    values = [grouped[label] for label in labels]

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    if values:
        box = ax.boxplot(values, tick_labels=labels, patch_artist=True)
        for patch, color in zip(box["boxes"], ["#c96f53", "#6f9d63"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
    else:
        ax.text(0.5, 0.5, "No episode rows", ha="center", va="center", transform=ax.transAxes)

    ax.set_title("Episode Mean Safety by Task Success")
    ax.set_xlabel("task_success")
    ax.set_ylabel("mean_safety_rate")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _property_success_rates(raw_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for (property_id,), grouped_rows in sorted(_group_by(raw_rows, ("property",)).items()):
        satisfied = [row for row in grouped_rows if _to_int(row["safety_satisfaction"]) == 1]
        violated = [row for row in grouped_rows if _to_int(row["safety_satisfaction"]) == 0]
        rows.append(
            {
                "property": property_id,
                "satisfied_success_rate": _mean([_to_float(row["task_success"]) for row in satisfied]),
                "violated_success_rate": _mean([_to_float(row["task_success"]) for row in violated]),
                "satisfied_n": len(satisfied),
                "violated_n": len(violated),
            }
        )
    return rows


def _save_property_bar(
    *,
    plt: Any,
    raw_rows: Sequence[Dict[str, Any]],
    output_path: Path,
    title_suffix: str = "",
) -> None:
    property_rows = _property_success_rates(raw_rows)
    labels = [row["property"] for row in property_rows]
    x_positions = list(range(len(property_rows)))
    width = 0.36
    satisfied = [row["satisfied_success_rate"] for row in property_rows]
    violated = [row["violated_success_rate"] for row in property_rows]

    fig_width = max(8.0, 0.55 * len(property_rows))
    fig, ax = plt.subplots(figsize=(fig_width, 5.2))
    ax.bar(
        [x - width / 2 for x in x_positions],
        satisfied,
        width,
        label="property satisfied",
        color="#6f9d63",
    )
    ax.bar(
        [x + width / 2 for x in x_positions],
        violated,
        width,
        label="property violated",
        color="#c96f53",
    )

    title = "Task Success Rate by Property Satisfaction"
    if title_suffix:
        title = f"{title}: {title_suffix}"
    ax.set_title(title)
    ax.set_xlabel("property")
    ax.set_ylabel("success rate")
    ax.set_ylim(0, 1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _safe_filename(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)
    return cleaned.strip("_") or "unknown"


def _normalize_category(value: Any) -> str:
    category = str(value or "").strip()
    return category or "Uncategorized"


def _parse_excluded_categories(values: Sequence[str]) -> List[str]:
    categories = []
    seen = set()
    for value in values:
        for category in str(value).split(","):
            normalized = _normalize_category(category)
            if normalized not in seen:
                categories.append(normalized)
                seen.add(normalized)
    return categories


def _rows_excluding_categories(
    raw_rows: Sequence[Dict[str, Any]],
    excluded_categories: Sequence[str],
) -> List[Dict[str, Any]]:
    excluded = {_normalize_category(category) for category in excluded_categories}
    return [
        row for row in raw_rows
        if _normalize_category(row.get("category")) not in excluded
    ]


def _excluded_category_label(excluded_categories: Sequence[str]) -> str:
    return "_".join(_safe_filename(category) for category in excluded_categories)


def _save_property_bars_by_suite(
    *,
    plt: Any,
    raw_rows: Sequence[Dict[str, Any]],
    task_suites: Dict[str, str],
    plot_dir: Path,
) -> List[Path]:
    output_paths = []
    rows_by_suite: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        suite = task_suites.get(str(row.get("task", "")), "UnmappedSuite")
        rows_by_suite[suite].append(row)

    for suite, suite_rows in sorted(rows_by_suite.items()):
        output_path = plot_dir / f"05_property_bar_success_by_satisfaction_{_safe_filename(suite)}.png"
        _save_property_bar(
            plt=plt,
            raw_rows=suite_rows,
            output_path=output_path,
            title_suffix=suite,
        )
        output_paths.append(output_path)
    return output_paths


def _save_episode_outcome_stacked_bars(
    *,
    plt: Any,
    outcome_rows: Sequence[Dict[str, Any]],
    output_path: Path,
) -> None:
    suites = sorted({str(row.get("taskSuites", "")) for row in outcome_rows})
    if "ALL_SUITES" in suites:
        suites = ["ALL_SUITES"] + [suite for suite in suites if suite != "ALL_SUITES"]

    n_cols = 4
    n_rows = max(2, (len(suites) + n_cols - 1) // n_cols)
    fig, axes_grid = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.8 * n_cols, 3.8 * n_rows),
        sharey=True,
    )
    axes = list(axes_grid.flat)

    stack_specs = [
        ("failed_safe_rate", "failed & safe", "#9bbf88"),
        ("failed_unsafe_rate", "failed & unsafe", "#c96f53"),
        ("successful_safe_rate", "successful & safe", "#3f8f5f"),
        ("successful_unsafe_rate", "successful & unsafe", "#d49a3a"),
    ]

    for ax, suite in zip(axes, suites):
        rows = [row for row in outcome_rows if row.get("taskSuites") == suite]
        policies = sorted({str(row.get("policy", "")) for row in rows})
        row_by_policy = {str(row.get("policy", "")): row for row in rows}
        x_positions = list(range(len(policies)))
        bottoms = [0.0 for _ in policies]

        for key, label, color in stack_specs:
            values = [_to_float(row_by_policy[policy].get(key)) for policy in policies]
            ax.bar(x_positions, values, bottom=bottoms, label=label, color=color)
            bottoms = [bottom + value for bottom, value in zip(bottoms, values)]

        suite_rollouts = sum(_to_int(row["n_episodes"]) for row in rows)
        ax.set_title(f"{suite} ({suite_rollouts} rollouts)")
        ax.set_ylim(0, 1)
        ax.set_ylabel("episode fraction")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(policies, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.25)

    for ax in axes[len(suites):]:
        ax.set_visible(False)

    axes[0].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.35),
        ncol=2,
        frameon=False,
    )
    fig.suptitle("Episode Outcomes by Policy and Task Suite", y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _save_episode_outcome_bar_by_policy(
    *,
    plt: Any,
    outcome_rows: Sequence[Dict[str, Any]],
    output_path: Path,
) -> None:
    policy_rows = [
        row for row in outcome_rows if str(row.get("taskSuites", "")) == "ALL_SUITES"
    ]
    policies = sorted({str(row.get("policy", "")) for row in policy_rows})
    row_by_policy = {str(row.get("policy", "")): row for row in policy_rows}

    if not policies:
        policies = sorted({str(row.get("policy", "")) for row in outcome_rows})
        row_by_policy = {}
        for policy in policies:
            rows = [row for row in outcome_rows if str(row.get("policy", "")) == policy]
            row_by_policy[policy] = {
                "failed_safe_rate": _mean([_to_float(row["failed_safe_rate"]) for row in rows]),
                "failed_unsafe_rate": _mean([_to_float(row["failed_unsafe_rate"]) for row in rows]),
                "successful_safe_rate": _mean(
                    [_to_float(row["successful_safe_rate"]) for row in rows]
                ),
                "successful_unsafe_rate": _mean(
                    [_to_float(row["successful_unsafe_rate"]) for row in rows]
                ),
            }

    outcome_specs = [
        ("successful_safe_rate", "safe & success", "#3f8f5f"),
        ("successful_unsafe_rate", "unsafe & success", "#d49a3a"),
        ("failed_safe_rate", "safe & fail", "#5b8db8"),
        ("failed_unsafe_rate", "unsafe & fail", "#c96f53"),
    ]

    fig_width = max(7.5, 1.2 * len(policies))
    fig, ax = plt.subplots(figsize=(fig_width, 5.2))
    x_positions = list(range(len(policies)))
    width = 0.18
    offsets = [
        (index - (len(outcome_specs) - 1) / 2) * width
        for index in range(len(outcome_specs))
    ]

    for offset, (key, label, color) in zip(offsets, outcome_specs):
        values = [_to_float(row_by_policy[policy].get(key)) for policy in policies]
        ax.bar(
            [x + offset for x in x_positions],
            values,
            width,
            label=label,
            color=color,
        )

    ax.set_title("Episode Outcome Rates by Policy")
    ax.set_xlabel("policy")
    ax.set_ylabel("rate")
    ax.set_ylim(0, 1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(policies, rotation=20, ha="right")
    ax.legend(ncol=2, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _save_episode_outcome_pie_by_policy(
    *,
    plt: Any,
    outcome_rows: Sequence[Dict[str, Any]],
    output_path: Path,
    title: str = "Episode Outcome Percentages by Policy",
) -> None:
    policy_rows = [
        row for row in outcome_rows if str(row.get("taskSuites", "")) == "ALL_SUITES"
    ]
    policies = sorted({str(row.get("policy", "")) for row in policy_rows})
    row_by_policy = {str(row.get("policy", "")): row for row in policy_rows}

    if not policies:
        policies = sorted({str(row.get("policy", "")) for row in outcome_rows})
        row_by_policy = {}
        for policy in policies:
            rows = [row for row in outcome_rows if str(row.get("policy", "")) == policy]
            row_by_policy[policy] = {
                "failed_safe_rate": _mean([_to_float(row["failed_safe_rate"]) for row in rows]),
                "failed_unsafe_rate": _mean([_to_float(row["failed_unsafe_rate"]) for row in rows]),
                "successful_safe_rate": _mean(
                    [_to_float(row["successful_safe_rate"]) for row in rows]
                ),
                "successful_unsafe_rate": _mean(
                    [_to_float(row["successful_unsafe_rate"]) for row in rows]
                ),
            }

    outcome_specs = [
        ("successful_safe_rate", "safe & success", "#3f8f5f"),
        ("successful_unsafe_rate", "unsafe & success", "#d49a3a"),
        ("failed_safe_rate", "safe & fail", "#5b8db8"),
        ("failed_unsafe_rate", "unsafe & fail", "#c96f53"),
    ]

    n_cols = min(4, max(1, len(policies)))
    n_rows = max(1, (len(policies) + n_cols - 1) // n_cols)
    fig, axes_grid = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.8 * n_cols, 4.6 * n_rows),
        squeeze=False,
    )
    axes = list(axes_grid.flat)
    labels = [label for _, label, _ in outcome_specs]
    colors = [color for _, _, color in outcome_specs]

    def autopct(percent: float) -> str:
        return f"{percent:.1f}%" if percent > 0 else ""

    for ax, policy in zip(axes, policies):
        row = row_by_policy[policy]
        values = [_to_float(row.get(key)) for key, _, _ in outcome_specs]
        if sum(values) > 0:
            ax.pie(
                values,
                colors=colors,
                autopct=autopct,
                pctdistance=0.7,
                startangle=90,
                counterclock=False,
                wedgeprops={"linewidth": 0.8, "edgecolor": "white"},
                textprops={"fontsize": 8},
            )
        else:
            ax.text(0.5, 0.5, "No outcome data", ha="center", va="center", transform=ax.transAxes)
        ax.text(
            0.5,
            -0.08,
            policy,
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
            transform=ax.transAxes,
        )
        ax.axis("equal")

    for ax in axes[len(policies):]:
        ax.set_visible(False)

    fig.legend(
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(title, y=1.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_success_vs_unsafe_success_by_policy(
    *,
    plt: Any,
    outcome_rows: Sequence[Dict[str, Any]],
    policy_ids: Dict[str, str],
    color_by_policy_id: bool,
    output_path: Path,
) -> None:
    policy_rows = [
        row for row in outcome_rows if str(row.get("taskSuites", "")) == "ALL_SUITES"
    ]
    if not policy_rows:
        policy_rows = []
        for policy, rows in sorted(_group_by(outcome_rows, ("policy",)).items()):
            n_episodes = sum(_to_int(row["n_episodes"]) for row in rows)
            successful_unsafe_count = sum(
                _to_float(row["successful_unsafe_rate"]) * _to_float(row["n_episodes"])
                for row in rows
            )
            successful_count = sum(
                _to_float(row["overall_task_success_rate"]) * _to_float(row["n_episodes"])
                for row in rows
            )
            policy_rows.append(
                {
                    "policy": policy,
                    "n_episodes": n_episodes,
                    "overall_task_success_rate": successful_count / n_episodes if n_episodes else 0.0,
                    "unsafe_success_count": successful_unsafe_count,
                    "unsafe_success_rate": successful_unsafe_count / n_episodes if n_episodes else 0.0,
                }
            )

    policy_rows = sorted(
        policy_rows,
        key=lambda row: (_to_float(row["overall_task_success_rate"]), str(row["policy"])),
    )
    x_values = [_to_float(row["overall_task_success_rate"]) for row in policy_rows]
    y_values = [
        _to_float(row["unsafe_success_rate"])
        if row.get("unsafe_success_rate", "") != ""
        else (
            _to_float(row.get("unsafe_success_count", ""))
            / _to_float(row["n_episodes"])
            if _to_float(row["n_episodes"]) > 0
            else 0.0
        )
        for row in policy_rows
    ]
    rollout_counts = [_to_int(row.get("n_episodes")) for row in policy_rows]
    total_rollouts = sum(rollout_counts)
    unique_rollout_counts = sorted(set(rollout_counts))
    if len(policy_rows) == 1:
        rollout_note = f"{total_rollouts} total rollouts"
    elif len(unique_rollout_counts) == 1:
        rollout_note = f"{unique_rollout_counts[0]} rollouts per policy, {total_rollouts} total rollouts"
    else:
        rollout_note = f"{total_rollouts} total rollouts across policies"

    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    if policy_rows:
        row_policy_ids = [
            _policy_id(row.get("policy"), policy_ids)
            for row in policy_rows
        ]
        ax.scatter(
            x_values,
            y_values,
            s=95,
            color=(
                [_policy_id_color(policy_id) for policy_id in row_policy_ids]
                if color_by_policy_id
                else "#c96f53"
            ),
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        for row, x, y in zip(policy_rows, x_values, y_values):
            ax.annotate(
                str(row.get("policy", "")),
                (x, y),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=8,
            )
        if color_by_policy_id:
            _add_policy_id_legend(plt, ax, row_policy_ids)
    else:
        ax.text(0.5, 0.5, "No policy rows", ha="center", va="center", transform=ax.transAxes)

    ax.set_title(f"Overall Success vs Unsafe-Success Rate by Policy\n{rollout_note}")
    ax.set_xlabel("overall success rate")
    ax.set_ylabel("unsafe-success rollouts / total rollouts per policy")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _save_success_vs_violation_rate_by_policy(
    *,
    plt: Any,
    outcome_rows: Sequence[Dict[str, Any]],
    policy_ids: Dict[str, str],
    color_by_policy_id: bool,
    output_path: Path,
) -> None:
    policy_rows = [
        row for row in outcome_rows if str(row.get("taskSuites", "")) == "ALL_SUITES"
    ]
    if not policy_rows:
        policy_rows = []
        for policy, rows in sorted(_group_by(outcome_rows, ("policy",)).items()):
            n_episodes = sum(_to_int(row["n_episodes"]) for row in rows)
            successful_count = sum(
                _to_float(row["overall_task_success_rate"]) * _to_float(row["n_episodes"])
                for row in rows
            )
            unsafe_count = sum(
                (
                    _to_float(row["failed_unsafe_rate"])
                    + _to_float(row["successful_unsafe_rate"])
                )
                * _to_float(row["n_episodes"])
                for row in rows
            )
            policy_rows.append(
                {
                    "policy": policy,
                    "n_episodes": n_episodes,
                    "overall_task_success_rate": successful_count / n_episodes if n_episodes else 0.0,
                    "overall_violation_rate": unsafe_count / n_episodes if n_episodes else 0.0,
                }
            )

    plot_rows = []
    for row in policy_rows:
        plot_rows.append(
            {
                "policy": str(row.get("policy") or "unknown_policy"),
                "n_episodes": _to_int(row.get("n_episodes")),
                "overall_task_success_rate": _to_float(row["overall_task_success_rate"]),
                "overall_violation_rate": (
                    _to_float(row["overall_violation_rate"])
                    if row.get("overall_violation_rate", "") != ""
                    else _to_float(row["failed_unsafe_rate"])
                    + _to_float(row["successful_unsafe_rate"])
                ),
            }
        )

    plot_rows = sorted(
        plot_rows,
        key=lambda row: (_to_float(row["overall_task_success_rate"]), str(row["policy"])),
    )
    x_values = [_to_float(row["overall_task_success_rate"]) for row in plot_rows]
    y_values = [_to_float(row["overall_violation_rate"]) for row in plot_rows]

    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    if plot_rows:
        row_policy_ids = [
            _policy_id(row.get("policy"), policy_ids)
            for row in plot_rows
        ]
        ax.scatter(
            x_values,
            y_values,
            s=95,
            color=(
                [_policy_id_color(policy_id) for policy_id in row_policy_ids]
                if color_by_policy_id
                else "#7b5ea7"
            ),
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        for row, x, y in zip(plot_rows, x_values, y_values):
            ax.annotate(
                str(row.get("policy", "")),
                (x, y),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=8,
            )
        if color_by_policy_id:
            _add_policy_id_legend(plt, ax, row_policy_ids)
    else:
        ax.text(0.5, 0.5, "No policy rows", ha="center", va="center", transform=ax.transAxes)

    ax.set_title("Overall Success Rate vs Overall Violation Rate by Policy")
    ax.set_xlabel("overall success rate")
    ax.set_ylabel("rollouts with >=1 violation / total rollouts")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _save_success_vs_unsafe_success_per_success_by_policy(
    *,
    plt: Any,
    outcome_rows: Sequence[Dict[str, Any]],
    output_path: Path,
) -> None:
    policy_rows = [
        row for row in outcome_rows if str(row.get("taskSuites", "")) == "ALL_SUITES"
    ]
    if not policy_rows:
        policy_rows = []
        for policy, rows in sorted(_group_by(outcome_rows, ("policy",)).items()):
            n_episodes = sum(_to_int(row["n_episodes"]) for row in rows)
            successful_unsafe_count = sum(
                _to_float(row["successful_unsafe_rate"]) * _to_float(row["n_episodes"])
                for row in rows
            )
            successful_count = sum(
                _to_float(row["overall_task_success_rate"]) * _to_float(row["n_episodes"])
                for row in rows
            )
            policy_rows.append(
                {
                    "policy": policy,
                    "n_episodes": n_episodes,
                    "overall_task_success_rate": successful_count / n_episodes if n_episodes else 0.0,
                    "unsafe_success_per_success": (
                        successful_unsafe_count / successful_count
                        if successful_count
                        else 0.0
                    ),
                }
            )

    plot_rows = []
    for row in policy_rows:
        success_rate = _to_float(row["overall_task_success_rate"])
        unsafe_success_rate = _to_float(row.get("unsafe_success_rate", ""))
        plot_rows.append(
            {
                "policy": str(row.get("policy") or "unknown_policy"),
                "overall_task_success_rate": success_rate,
                "unsafe_success_per_success": (
                    _to_float(row["unsafe_success_per_success"])
                    if row.get("unsafe_success_per_success", "") != ""
                    else unsafe_success_rate / success_rate
                    if success_rate
                    else 0.0
                ),
            }
        )

    plot_rows = sorted(
        plot_rows,
        key=lambda row: (_to_float(row["overall_task_success_rate"]), str(row["policy"])),
    )
    x_values = [_to_float(row["overall_task_success_rate"]) for row in plot_rows]
    y_values = [_to_float(row["unsafe_success_per_success"]) for row in plot_rows]

    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    if plot_rows:
        ax.scatter(
            x_values,
            y_values,
            s=95,
            color="#d49a3a",
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        for row, x, y in zip(plot_rows, x_values, y_values):
            ax.annotate(
                str(row.get("policy", "")),
                (x, y),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=8,
            )
    else:
        ax.text(0.5, 0.5, "No policy rows", ha="center", va="center", transform=ax.transAxes)

    ax.set_title("Overall Success vs Unsafe-Success Share of Successful Rollouts by Policy")
    ax.set_xlabel("overall success rate")
    ax.set_ylabel("unsafe-success rollouts / successful rollouts")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _save_success_vs_violations_per_skill_onset_by_policy(
    *,
    plt: Any,
    raw_rows: Sequence[Dict[str, Any]],
    outcome_rows: Sequence[Dict[str, Any]],
    output_path: Path,
) -> None:
    policy_outcome_rows = [
        row for row in outcome_rows if str(row.get("taskSuites", "")) == "ALL_SUITES"
    ]
    if not policy_outcome_rows:
        policy_outcome_rows = []
        for policy, rows in sorted(_group_by(outcome_rows, ("policy",)).items()):
            n_episodes = sum(_to_int(row["n_episodes"]) for row in rows)
            successful_count = sum(
                _to_float(row["overall_task_success_rate"]) * _to_float(row["n_episodes"])
                for row in rows
            )
            policy_outcome_rows.append(
                {
                    "policy": policy,
                    "overall_task_success_rate": successful_count / n_episodes if n_episodes else 0.0,
                }
            )

    ratio_values_by_policy: Dict[str, List[float]] = defaultdict(list)
    for row in raw_rows:
        value = row.get("violations_per_skill_onset", "")
        if value in (None, ""):
            continue
        ratio_values_by_policy[str(row.get("policy") or "unknown_policy")].append(
            _to_float(value)
        )

    plot_rows = []
    for row in policy_outcome_rows:
        policy = str(row.get("policy") or "unknown_policy")
        plot_rows.append(
            {
                "policy": policy,
                "overall_task_success_rate": _to_float(row["overall_task_success_rate"]),
                "avg_violations_per_skill_onset": _mean(ratio_values_by_policy[policy]),
                "n_ratio_rows": len(ratio_values_by_policy[policy]),
            }
        )

    plot_rows = sorted(
        plot_rows,
        key=lambda row: (_to_float(row["overall_task_success_rate"]), str(row["policy"])),
    )
    x_values = [_to_float(row["overall_task_success_rate"]) for row in plot_rows]
    y_values = [_to_float(row["avg_violations_per_skill_onset"]) for row in plot_rows]

    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    if plot_rows:
        ax.plot(x_values, y_values, color="#5b8db8", linewidth=1.5, alpha=0.75)
        ax.scatter(
            x_values,
            y_values,
            s=95,
            color="#2f6f9f",
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        for row, x, y in zip(plot_rows, x_values, y_values):
            ax.annotate(
                str(row.get("policy", "")),
                (x, y),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=8,
            )
    else:
        ax.text(0.5, 0.5, "No policy rows", ha="center", va="center", transform=ax.transAxes)

    ax.set_title("Overall Success vs Average Violations per Skill Onset by Policy")
    ax.set_xlabel("overall success rate")
    ax.set_ylabel("average violations / skill onset")
    ax.set_xlim(-0.03, 1.03)
    y_max = max(y_values) if y_values else 0.0
    ax.set_ylim(-0.03, max(1.0, y_max * 1.15 + 0.05))
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _save_suite_success_vs_unsafe_success_count(
    *,
    plt: Any,
    outcome_rows: Sequence[Dict[str, Any]],
    output_path: Path,
) -> None:
    suite_rows = [
        row for row in outcome_rows if str(row.get("taskSuites", "")) != "ALL_SUITES"
    ]
    suites = sorted({str(row.get("taskSuites", "")) for row in suite_rows})
    rows_by_suite = {
        suite: [
            row for row in suite_rows
            if str(row.get("taskSuites", "")) == suite
        ]
        for suite in suites
    }
    all_counts = [
        _to_float(row["successful_unsafe_rate"]) * _to_float(row["n_episodes"])
        for row in suite_rows
    ]
    y_max = max(all_counts) if all_counts else 0.0

    n_rows = 2
    n_cols = 4
    fig, axes_grid = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.8 * n_cols, 4.0 * n_rows),
        sharex=True,
        sharey=True,
    )
    axes = list(axes_grid.flat)

    for ax, suite in zip(axes, suites[: n_rows * n_cols]):
        rows = sorted(
            rows_by_suite[suite],
            key=lambda row: (_to_float(row["overall_task_success_rate"]), str(row["policy"])),
        )
        x_values = [_to_float(row["overall_task_success_rate"]) for row in rows]
        y_values = [
            _to_float(row["successful_unsafe_rate"]) * _to_float(row["n_episodes"])
            for row in rows
        ]

        ax.plot(x_values, y_values, color="#d49a3a", linewidth=1.4, alpha=0.75)
        ax.scatter(
            x_values,
            y_values,
            s=78,
            color="#c96f53",
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        for row, x, y in zip(rows, x_values, y_values):
            ax.annotate(
                str(row.get("policy", "")),
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=7,
            )

        suite_rollouts = sum(_to_int(row["n_episodes"]) for row in rows)
        ax.set_title(f"{suite}\n{suite_rollouts} rollouts", fontsize=9, pad=8)
        ax.text(
            0.02,
            0.95,
            f"N={suite_rollouts}",
            ha="left",
            va="top",
            fontsize=8,
            fontweight="bold",
            transform=ax.transAxes,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 2},
        )
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.05, max(1.0, y_max * 1.15 + 0.2))
        ax.grid(True, alpha=0.25)

    for ax in axes[len(suites[: n_rows * n_cols]):]:
        ax.set_visible(False)

    for ax in axes[-n_cols:]:
        ax.set_xlabel("suite success rate")
    for ax in axes[::n_cols]:
        ax.set_ylabel("unsafe-success rollout count")

    fig.suptitle("Suite Success Rate vs Unsafe-Success Rollout Count", y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _save_category_success_vs_unsafe_success_count(
    *,
    plt: Any,
    category_outcome_rows: Sequence[Dict[str, Any]],
    output_path: Path,
) -> None:
    categories = sorted({str(row.get("category", "")) for row in category_outcome_rows})
    rows_by_category = {
        category: [
            row for row in category_outcome_rows
            if str(row.get("category", "")) == category
        ]
        for category in categories
    }
    all_counts = [
        _to_float(row["successful_unsafe_rate"]) * _to_float(row["n_episodes"])
        for row in category_outcome_rows
    ]
    y_max = max(all_counts) if all_counts else 0.0

    n_cols = 4
    n_rows = max(1, (len(categories) + n_cols - 1) // n_cols)
    fig, axes_grid = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.8 * n_cols, 4.0 * n_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = list(axes_grid.flat)

    for ax, category in zip(axes, categories):
        rows = sorted(
            rows_by_category[category],
            key=lambda row: (_to_float(row["overall_task_success_rate"]), str(row["policy"])),
        )
        x_values = [_to_float(row["overall_task_success_rate"]) for row in rows]
        y_values = [
            _to_float(row["successful_unsafe_rate"]) * _to_float(row["n_episodes"])
            for row in rows
        ]

        ax.plot(x_values, y_values, color="#5b8db8", linewidth=1.4, alpha=0.75)
        ax.scatter(
            x_values,
            y_values,
            s=78,
            color="#2f6f9f",
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        for row, x, y in zip(rows, x_values, y_values):
            ax.annotate(
                str(row.get("policy", "")),
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=7,
            )

        category_rollouts = sum(_to_int(row["n_episodes"]) for row in rows)
        ax.set_title(f"{category}\n{category_rollouts} rollouts", fontsize=9, pad=8)
        ax.text(
            0.02,
            0.95,
            f"N={category_rollouts}",
            ha="left",
            va="top",
            fontsize=8,
            fontweight="bold",
            transform=ax.transAxes,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 2},
        )
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.05, max(1.0, y_max * 1.15 + 0.2))
        ax.grid(True, alpha=0.25)

    for ax in axes[len(categories):]:
        ax.set_visible(False)

    for ax in axes[-n_cols:]:
        ax.set_xlabel("category success rate")
    for ax in axes[::n_cols]:
        ax.set_ylabel("unsafe-success rollout count")

    fig.suptitle("Category Success Rate vs Unsafe-Success Rollout Count", y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def make_plots(
    *,
    raw_rows: Sequence[Dict[str, Any]],
    episode_rows: Sequence[Dict[str, Any]],
    summary_rows: Sequence[Dict[str, Any]],
    plot1_category_rows: Sequence[Dict[str, Any]],
    plot1_overall_rows: Sequence[Dict[str, Any]],
    plot1_suite_rows: Sequence[Dict[str, Any]],
    category_summary_rows: Sequence[Dict[str, Any]],
    outcome_rows: Sequence[Dict[str, Any]],
    task_suites: Dict[str, str],
    policy_ids: Dict[str, str],
    color_by_policy_id: bool,
    plot_dir: Path,
    plot_data_dir: Path,
    excluded_categories: Sequence[str] = (),
) -> List[Path]:
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_data_dir.mkdir(parents=True, exist_ok=True)
    old_combined_property_plot = plot_dir / "05_property_bar_success_by_satisfaction.png"
    if old_combined_property_plot.exists():
        old_combined_property_plot.unlink()
    old_outcome_scatter_plot = plot_dir / "07_episode_outcome_scatter_by_policy.png"
    if old_outcome_scatter_plot.exists():
        old_outcome_scatter_plot.unlink()
    old_unsafe_success_rate_plot = plot_dir / "10_success_vs_unsafe_success_by_policy.png"
    if old_unsafe_success_rate_plot.exists():
        old_unsafe_success_rate_plot.unlink()
    old_unsafe_success_count_plot = plot_dir / "10_success_vs_unsafe_success_count_by_policy.png"
    if old_unsafe_success_count_plot.exists():
        old_unsafe_success_count_plot.unlink()
    for old_excluded_pie in plot_dir.glob("09_episode_outcome_pie_excluding_*.png"):
        old_excluded_pie.unlink()
    plt = _setup_matplotlib(plot_dir)
    category_outcome_rows = build_category_outcome_rows(raw_rows)

    outputs = _save_plot1_category_scatters_by_policy(
        plt=plt,
        rows=plot1_category_rows,
        plot_dir=plot_dir,
    )
    outputs.extend(
        _save_plot1_overall_scatters_by_policy(
            plt=plt,
            rows=plot1_overall_rows,
            plot_dir=plot_dir,
        )
    )
    outputs.extend(
        _save_plot1_suite_scatters_by_policy(
            plt=plt,
            rows=plot1_suite_rows,
            plot_dir=plot_dir,
        )
    )
    outputs.extend([
        plot_dir / "01_5_category_scatter_success_vs_strict_safety.png",
        plot_dir / "02_soft_scatter_success_vs_mean_safety.png",
        plot_dir / "03_paired_bar_success_vs_safe_success.png",
        plot_dir / "04_episode_box_task_success_vs_mean_safety.png",
        plot_dir / "06_episode_outcomes_stacked_by_suite_policy.png",
        plot_dir / "07_episode_outcome_bar_by_policy.png",
        plot_dir / "08_episode_outcome_pie_by_policy.png",
        plot_dir / "10_success_vs_unsafe_success_rate_by_policy.png",
        plot_dir / "10_1_success_vs_unsafe_success_per_success_by_policy.png",
        plot_dir / "11_success_vs_violations_per_skill_onset_by_policy.png",
        plot_dir / "12_suite_success_vs_unsafe_success_count.png",
        plot_dir / "12_5_category_success_vs_unsafe_success_count.png",
        plot_dir / "13_success_vs_overall_violation_rate_by_policy.png",
    ])
    local_outputs = list(outputs)

    for policy in sorted({str(row.get("policy", "")) for row in plot1_category_rows}):
        plot_path = plot_dir / f"01_main_scatter_success_vs_category_violation_{_safe_filename(policy)}.png"
        write_plot_data_csv(
            plot_data_dir,
            plot_path,
            [row for row in plot1_category_rows if str(row.get("policy", "")) == policy],
        )
    for policy in sorted({str(row.get("policy", "")) for row in plot1_overall_rows}):
        plot_path = plot_dir / f"01_1_scatter_success_vs_category_violation_{_safe_filename(policy)}.png"
        write_plot_data_csv(
            plot_data_dir,
            plot_path,
            [row for row in plot1_overall_rows if str(row.get("policy", "")) == policy],
        )
    for policy in sorted({str(row.get("policy", "")) for row in plot1_suite_rows}):
        plot_path = plot_dir / f"01_2_scatter_success_vs_suite_property_violation_{_safe_filename(policy)}.png"
        write_plot_data_csv(
            plot_data_dir,
            plot_path,
            [row for row in plot1_suite_rows if str(row.get("policy", "")) == policy],
        )
    write_plot_data_csv(
        plot_data_dir,
        plot_dir / "01_5_category_scatter_success_vs_strict_safety.png",
        category_summary_rows,
    )
    write_plot_data_csv(
        plot_data_dir,
        plot_dir / "02_soft_scatter_success_vs_mean_safety.png",
        summary_rows,
    )
    write_plot_data_csv(
        plot_data_dir,
        plot_dir / "03_paired_bar_success_vs_safe_success.png",
        summary_rows,
    )
    write_plot_data_csv(
        plot_data_dir,
        plot_dir / "04_episode_box_task_success_vs_mean_safety.png",
        episode_rows,
    )
    write_plot_data_csv(
        plot_data_dir,
        plot_dir / "06_episode_outcomes_stacked_by_suite_policy.png",
        outcome_rows,
    )
    write_plot_data_csv(
        plot_data_dir,
        plot_dir / "07_episode_outcome_bar_by_policy.png",
        _policy_outcome_rows(outcome_rows),
    )
    write_plot_data_csv(
        plot_data_dir,
        plot_dir / "08_episode_outcome_pie_by_policy.png",
        _policy_outcome_rows(outcome_rows),
    )
    write_plot_data_csv(
        plot_data_dir,
        plot_dir / "10_success_vs_unsafe_success_rate_by_policy.png",
        _plot10_rows(outcome_rows),
    )
    write_plot_data_csv(
        plot_data_dir,
        plot_dir / "10_1_success_vs_unsafe_success_per_success_by_policy.png",
        _plot10_1_rows(outcome_rows),
    )
    write_plot_data_csv(
        plot_data_dir,
        plot_dir / "11_success_vs_violations_per_skill_onset_by_policy.png",
        _plot11_rows(raw_rows, outcome_rows),
    )
    write_plot_data_csv(
        plot_data_dir,
        plot_dir / "12_suite_success_vs_unsafe_success_count.png",
        [row for row in outcome_rows if str(row.get("taskSuites", "")) != "ALL_SUITES"],
    )
    write_plot_data_csv(
        plot_data_dir,
        plot_dir / "12_5_category_success_vs_unsafe_success_count.png",
        category_outcome_rows,
    )
    write_plot_data_csv(
        plot_data_dir,
        plot_dir / "13_success_vs_overall_violation_rate_by_policy.png",
        _policy_outcome_rows(outcome_rows),
    )
    for suite in sorted({task_suites.get(str(row.get("task", "")), "UnmappedSuite") for row in raw_rows}):
        suite_rows = [
            row for row in raw_rows
            if task_suites.get(str(row.get("task", "")), "UnmappedSuite") == suite
        ]
        plot_path = plot_dir / f"05_property_bar_success_by_satisfaction_{_safe_filename(suite)}.png"
        write_plot_data_csv(plot_data_dir, plot_path, _property_success_rates(suite_rows))

    _save_category_scatter(
        plt=plt,
        rows=category_summary_rows,
        output_path=plot_dir / "01_5_category_scatter_success_vs_strict_safety.png",
    )
    _save_scatter(
        plt=plt,
        rows=summary_rows,
        x_key="success_rate",
        y_key="mean_safety_rate",
        title="Success Rate vs Mean Safety Rate",
        output_path=plot_dir / "02_soft_scatter_success_vs_mean_safety.png",
    )
    _save_paired_bar(plt=plt, rows=summary_rows, output_path=plot_dir / "03_paired_bar_success_vs_safe_success.png")
    _save_episode_boxplot(plt=plt, rows=episode_rows, output_path=plot_dir / "04_episode_box_task_success_vs_mean_safety.png")
    _save_episode_outcome_stacked_bars(
        plt=plt,
        outcome_rows=outcome_rows,
        output_path=plot_dir / "06_episode_outcomes_stacked_by_suite_policy.png",
    )
    _save_episode_outcome_bar_by_policy(
        plt=plt,
        outcome_rows=outcome_rows,
        output_path=plot_dir / "07_episode_outcome_bar_by_policy.png",
    )
    _save_episode_outcome_pie_by_policy(
        plt=plt,
        outcome_rows=outcome_rows,
        output_path=plot_dir / "08_episode_outcome_pie_by_policy.png",
    )
    _save_success_vs_unsafe_success_by_policy(
        plt=plt,
        outcome_rows=outcome_rows,
        policy_ids=policy_ids,
        color_by_policy_id=color_by_policy_id,
        output_path=plot_dir / "10_success_vs_unsafe_success_rate_by_policy.png",
    )
    _save_success_vs_unsafe_success_per_success_by_policy(
        plt=plt,
        outcome_rows=outcome_rows,
        output_path=plot_dir / "10_1_success_vs_unsafe_success_per_success_by_policy.png",
    )
    _save_success_vs_violations_per_skill_onset_by_policy(
        plt=plt,
        raw_rows=raw_rows,
        outcome_rows=outcome_rows,
        output_path=plot_dir / "11_success_vs_violations_per_skill_onset_by_policy.png",
    )
    _save_suite_success_vs_unsafe_success_count(
        plt=plt,
        outcome_rows=outcome_rows,
        output_path=plot_dir / "12_suite_success_vs_unsafe_success_count.png",
    )
    _save_category_success_vs_unsafe_success_count(
        plt=plt,
        category_outcome_rows=category_outcome_rows,
        output_path=plot_dir / "12_5_category_success_vs_unsafe_success_count.png",
    )
    _save_success_vs_violation_rate_by_policy(
        plt=plt,
        outcome_rows=outcome_rows,
        policy_ids=policy_ids,
        color_by_policy_id=color_by_policy_id,
        output_path=plot_dir / "13_success_vs_overall_violation_rate_by_policy.png",
    )
    property_plot_paths = _save_property_bars_by_suite(
        plt=plt,
        raw_rows=raw_rows,
        task_suites=task_suites,
        plot_dir=plot_dir,
    )
    outputs.extend(property_plot_paths)
    local_outputs.extend(property_plot_paths)
    if excluded_categories:
        filtered_raw_rows = _rows_excluding_categories(raw_rows, excluded_categories)
        excluded_episode_rows = build_episode_rows(filtered_raw_rows)
        excluded_summary_rows = build_summary_rows(excluded_episode_rows)
        excluded_plot1_category_rows = build_plot1_category_rows(filtered_raw_rows)
        excluded_plot1_overall_rows = build_plot1_overall_rows(filtered_raw_rows)
        excluded_plot1_suite_rows = build_plot1_suite_rows(filtered_raw_rows, task_suites)
        excluded_category_summary_rows = build_category_summary_rows(
            filtered_raw_rows,
            task_suites,
        )
        excluded_outcome_rows = build_outcome_rows(excluded_episode_rows, task_suites)
        excluded_label = _excluded_category_label(excluded_categories)
        excluded_display = ", ".join(excluded_categories)
        excluded_output_path = plot_dir / f"09_episode_outcome_pie_excluding_{excluded_label}.png"
        _save_episode_outcome_pie_by_policy(
            plt=plt,
            outcome_rows=excluded_outcome_rows,
            output_path=excluded_output_path,
            title=f"Episode Outcome Percentages by Policy Excluding: {excluded_display}",
        )
        write_plot_data_csv(
            plot_data_dir,
            excluded_output_path,
            _policy_outcome_rows(excluded_outcome_rows),
        )
        outputs.append(excluded_output_path)
        local_outputs.append(excluded_output_path)
        warn_missing_plot_data(local_outputs, plot_data_dir)

        excluded_plot_dir = plot_dir / f"excluding_{excluded_label}"
        excluded_plot_data_dir = plot_data_dir / f"excluding_{excluded_label}"
        outputs.extend(
            make_plots(
                raw_rows=filtered_raw_rows,
                episode_rows=excluded_episode_rows,
                summary_rows=excluded_summary_rows,
                plot1_category_rows=excluded_plot1_category_rows,
                plot1_overall_rows=excluded_plot1_overall_rows,
                plot1_suite_rows=excluded_plot1_suite_rows,
                category_summary_rows=excluded_category_summary_rows,
                outcome_rows=excluded_outcome_rows,
                task_suites=task_suites,
                policy_ids=policy_ids,
                color_by_policy_id=color_by_policy_id,
                plot_dir=excluded_plot_dir,
                plot_data_dir=excluded_plot_data_dir,
            )
        )
    else:
        warn_missing_plot_data(local_outputs, plot_data_dir)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build RQ1 master, episode-level, and task-policy summary CSVs."
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=DEFAULT_PROCESSED_ROOT,
        help=f"Root containing metrics CSVs. Default: {DEFAULT_PROCESSED_ROOT}",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
        help=f"Root containing rawData policy/task folders. Default: {DEFAULT_RAW_ROOT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for RQ1 outputs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=DEFAULT_PLOT_DIR,
        help=f"Directory for RQ1 plots. Default: {DEFAULT_PLOT_DIR}",
    )
    parser.add_argument(
        "--plot-data-dir",
        type=Path,
        default=None,
        help="Directory for per-plot RQ1 CSV data. Default: <output-dir>/plotData",
    )
    parser.add_argument(
        "--suites-csv",
        type=Path,
        default=DEFAULT_SUITES_CSV,
        help=f"CSV mapping taskName to taskSuites. Default: {DEFAULT_SUITES_CSV}",
    )
    parser.add_argument(
        "--policy-csv",
        type=Path,
        default=DEFAULT_POLICY_CSV,
        help=f"CSV mapping policy to policy id/color group. Default: {DEFAULT_POLICY_CSV}",
    )
    parser.add_argument(
        "--color-by-policy-id",
        dest="color_by_policy_id",
        action="store_true",
        default=True,
        help="Color plot 10 and plot 13 by the policy id mapping. Default: on.",
    )
    parser.add_argument(
        "--no-color-by-policy-id",
        dest="color_by_policy_id",
        action="store_false",
        help="Use the original single-color styling for plot 10 and plot 13.",
    )
    parser.add_argument(
        "--pattern",
        default="metrics_*.csv",
        help="CSV glob to concatenate. Default: metrics_*.csv",
    )
    parser.add_argument(
        "--exclude-category",
        action="append",
        default=[],
        help=(
            "Category to exclude from the extra filtered RQ1 plots. "
            "Can be repeated or comma-separated, e.g. --exclude-category CollisionContact,EAS."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_root = args.processed_root.expanduser().resolve()
    raw_root = args.raw_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    plot_dir = args.plot_dir.expanduser().resolve()
    plot_data_dir = (
        args.plot_data_dir.expanduser().resolve()
        if args.plot_data_dir is not None
        else output_dir / "plotData"
    )
    suites_csv = args.suites_csv.expanduser().resolve()
    policy_csv = args.policy_csv.expanduser().resolve()
    excluded_categories = _parse_excluded_categories(args.exclude_category)

    metric_paths = _metric_csvs(processed_root, args.pattern)
    if not metric_paths:
        raise SystemExit(f"No metrics CSV files found under {processed_root} matching {args.pattern!r}.")
    warn_mismatched_metric_paths(metric_paths)

    raw_rows = _read_rows(metric_paths)
    task_list_rows = build_task_list_rows(raw_rows, raw_root=raw_root)
    task_suites = load_task_suites(suites_csv)
    policy_ids = load_policy_ids(policy_csv)
    episode_rows = build_episode_rows(raw_rows)
    summary_rows = build_summary_rows(episode_rows)
    plot1_category_rows = build_plot1_category_rows(raw_rows)
    plot1_overall_rows = build_plot1_overall_rows(raw_rows)
    plot1_suite_rows = build_plot1_suite_rows(raw_rows, task_suites)
    category_summary_rows = build_category_summary_rows(raw_rows, task_suites)
    outcome_rows = build_outcome_rows(episode_rows, task_suites)

    write_csv(output_dir / "master_raw.csv", raw_rows, MASTER_RAW_COLUMNS)
    write_csv(output_dir / "task_list.csv", task_list_rows, TASK_LIST_COLUMNS)
    write_csv(output_dir / "episode_level.csv", episode_rows, EPISODE_COLUMNS)
    write_csv(output_dir / "task_policy_summary.csv", summary_rows, SUMMARY_COLUMNS)
    write_csv(
        output_dir / "category_taskSuite_summary.csv",
        category_summary_rows,
        CATEGORY_SUITE_COLUMNS,
    )
    write_csv(
        output_dir / "episode_outcome_rates.csv",
        outcome_rows,
        OUTCOME_COLUMNS,
    )
    plot_paths = make_plots(
        raw_rows=raw_rows,
        episode_rows=episode_rows,
        summary_rows=summary_rows,
        plot1_category_rows=plot1_category_rows,
        plot1_overall_rows=plot1_overall_rows,
        plot1_suite_rows=plot1_suite_rows,
        category_summary_rows=category_summary_rows,
        outcome_rows=outcome_rows,
        task_suites=task_suites,
        policy_ids=policy_ids,
        color_by_policy_id=args.color_by_policy_id,
        plot_dir=plot_dir,
        plot_data_dir=plot_data_dir,
        excluded_categories=excluded_categories,
    )

    print(f"Read {len(metric_paths)} metrics CSV file(s).")
    print(f"Master raw rows: {len(raw_rows)}")
    print(f"Task list rows: {len(task_list_rows)}")
    print(f"Episode-level rows: {len(episode_rows)}")
    print(f"Task-policy summary rows: {len(summary_rows)}")
    print(f"Plot 1 category-policy rows: {len(plot1_category_rows)}")
    print(f"Plot 1.1 category-task rows: {len(plot1_overall_rows)}")
    print(f"Plot 1.2 suite-policy rows: {len(plot1_suite_rows)}")
    if excluded_categories:
        print(f"Excluded categories for filtered plots: {', '.join(excluded_categories)}")
    print(f"Wrote: {output_dir / 'master_raw.csv'}")
    print(f"Wrote: {output_dir / 'task_list.csv'}")
    print(f"Wrote: {output_dir / 'episode_level.csv'}")
    print(f"Wrote: {output_dir / 'task_policy_summary.csv'}")
    print(f"Wrote: {output_dir / 'category_taskSuite_summary.csv'}")
    print(f"Wrote: {output_dir / 'episode_outcome_rates.csv'}")
    print(f"Wrote plot data CSVs under: {plot_data_dir}")
    for plot_path in plot_paths:
        print(f"Wrote: {plot_path}")
    warn_existing_plot_data_mismatches(plot_dir, plot_data_dir)


if __name__ == "__main__":
    main()
