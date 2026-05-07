from __future__ import annotations

"""Build RQ3 property-level metrics.

For each raw metrics row:
    violated = 1[violation_count_per_property > 0]

For each property p:
    VR_p = mean(violated | p)
    VRsucc_p = mean(violated | p, task_success = 1)
    HFR_p = mean(task_success * violated | p)
    AVD_p = mean(violation_duration_per_property | p, violated = 1)
    AER_p = mean(exposure_rate | p, exposure_rate > 0)

The same rates are also aggregated by property category for a category-level plot.
"""

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


ANALYSIS_DIR = Path(__file__).resolve().parent
DEFAULT_PROCESSED_ROOT = ANALYSIS_DIR / "processedData"
DEFAULT_OUTPUT_DIR = DEFAULT_PROCESSED_ROOT / "RQ3"
DEFAULT_PLOT_DIR = ANALYSIS_DIR / "plots" / "RQ3"
DEFAULT_SUITES_CSV = ANALYSIS_DIR / "idSuites.csv"
DEFAULT_TASK_DIFF_CSV = ANALYSIS_DIR / "taskDiff.csv"


RAW_REQUIRED_COLUMNS = (
    "task",
    "episode",
    "policy",
    "property",
    "violation_count_per_property",
    "violation_duration_per_property",
    "exposure_rate",
    "task_success",
)


RAW_OPTIONAL_COLUMNS = (
    "category",
    "safety_satisfaction",
)


OUTPUT_COLUMNS = (
    "property",
    "VR_p",
    "VRsucc_p",
    "HFR_p",
    "AVD_p",
    "AER_p",
)


CATEGORY_OUTPUT_COLUMNS = (
    "category",
    "VR_category",
    "VRsucc_category",
    "HFR_category",
    "AVD_category",
    "AER_category",
)


SUITE_POLICY_OUTPUT_COLUMNS = (
    "policy",
    "taskSuites",
    "n_episodes",
    "task_success_count",
    "task_success_rate",
    "safety_violation_count",
    "safety_violation_rate",
)


DIFFICULTY_POLICY_OUTPUT_COLUMNS = (
    "policy",
    "difficulty_horizon",
    "n_episodes",
    "task_success_count",
    "task_success_rate",
    "safety_violation_count",
    "safety_violation_rate",
    "safe-success",
    "unsafe-success",
    "safe-fail",
    "unsafe-fail",
)


DIFFICULTY_ACTION_POLICY_OUTPUT_COLUMNS = (
    "policy",
    "difficulty_horizon",
    "n_episodes",
    "task_success_count",
    "task_success_rate",
    "safety_violation_count",
    "total_action_count",
    "violation_per_action",
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


def _group_by(
    rows: Iterable[Dict[str, Any]],
    keys: Sequence[str],
) -> Dict[tuple[Any, ...], List[Dict[str, Any]]]:
    grouped: Dict[tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in keys)].append(row)
    return dict(grouped)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


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
            fieldnames = set(reader.fieldnames or [])
            for row in reader:
                output_row = {column: row.get(column, "") for column in RAW_REQUIRED_COLUMNS}
                for column in RAW_OPTIONAL_COLUMNS:
                    output_row[column] = row.get(column, "") if column in fieldnames else ""
                rows.append(output_row)
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


def load_task_difficulties(path: Path) -> Dict[str, str]:
    if not path.exists():
        raise SystemExit(f"Task difficulty CSV not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit(f"Task difficulty CSV is empty: {path}")

        normalized_fields = {field.lower(): field for field in reader.fieldnames}
        task_field = (
            normalized_fields.get("taskid")
            or normalized_fields.get("taskname")
            or normalized_fields.get("task")
        )
        difficulty_field = (
            normalized_fields.get("difficulty_horizon")
            or normalized_fields.get("difficultyhorizon")
            or normalized_fields.get("difficulty")
        )
        if not task_field or not difficulty_field:
            raise SystemExit(
                "Expected taskID and difficulty_horizon columns in task difficulty CSV: "
                f"{path}"
            )

        difficulties = {}
        for row in reader:
            task = str(row.get(task_field) or "").strip()
            difficulty = str(row.get(difficulty_field) or "").strip()
            if task:
                difficulties[task] = difficulty or "UnmappedDifficulty"
        return difficulties


def load_task_action_counts(path: Path) -> Dict[str, float]:
    if not path.exists():
        raise SystemExit(f"Task difficulty CSV not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit(f"Task difficulty CSV is empty: {path}")

        normalized_fields = {field.lower(): field for field in reader.fieldnames}
        task_field = (
            normalized_fields.get("taskid")
            or normalized_fields.get("taskname")
            or normalized_fields.get("task")
        )
        action_field = (
            normalized_fields.get("numbersubtask")
            or normalized_fields.get("number_subtask")
            or normalized_fields.get("numberaction")
            or normalized_fields.get("number_action")
            or normalized_fields.get("actions")
        )
        if not task_field or not action_field:
            raise SystemExit(
                "Expected taskID and numberSubtask columns in task difficulty CSV: "
                f"{path}"
            )

        action_counts = {}
        for row in reader:
            task = str(row.get(task_field) or "").strip()
            if task:
                action_counts[task] = _to_float(row.get(action_field))
        return action_counts


def _row_violated(row: Dict[str, Any]) -> int:
    if _to_int(row["violation_count_per_property"]) > 0:
        return 1
    safety_satisfaction = row.get("safety_satisfaction", "")
    if safety_satisfaction != "" and _to_int(safety_satisfaction) == 0:
        return 1
    return 0


def _build_group_rows(
    raw_rows: Sequence[Dict[str, Any]],
    *,
    group_key: str,
    fallback_group: str,
    label_column: str,
    metric_suffix: str,
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        group_id = str(row.get(group_key) or fallback_group)
        grouped[group_id].append(row)

    output_rows = []
    for group_id, rows in sorted(grouped.items()):
        violated_values = [
            1 if _to_int(row["violation_count_per_property"]) > 0 else 0 for row in rows
        ]
        success_rows = [row for row in rows if _to_int(row["task_success"]) == 1]
        violated_rows = [
            row for row in rows if _to_int(row["violation_count_per_property"]) > 0
        ]
        positive_exposure_rows = [
            row for row in rows if _to_float(row["exposure_rate"]) > 0
        ]
        n_rows = len(rows)
        violation_rows = sum(violated_values)
        success_violation_rows = sum(
            1 if _to_int(row["violation_count_per_property"]) > 0 else 0
            for row in success_rows
        )
        hfr_sum = sum(
            _to_int(row["task_success"])
            * (1 if _to_int(row["violation_count_per_property"]) > 0 else 0)
            for row in rows
        )
        violation_duration_sum = sum(
            _to_float(row["violation_duration_per_property"]) for row in violated_rows
        )
        positive_exposure_sum = sum(_to_float(row["exposure_rate"]) for row in positive_exposure_rows)

        output_rows.append(
            {
                label_column: group_id,
                "n_rows": n_rows,
                "violation_rows": violation_rows,
                f"VR_{metric_suffix}": _mean(violated_values),
                "success_rows": len(success_rows),
                "success_violation_rows": success_violation_rows,
                f"VRsucc_{metric_suffix}": _mean(
                    [
                        1 if _to_int(row["violation_count_per_property"]) > 0 else 0
                        for row in success_rows
                    ]
                ),
                "task_success_times_violated_sum": hfr_sum,
                f"HFR_{metric_suffix}": hfr_sum / n_rows if n_rows else 0.0,
                "violation_duration_sum": violation_duration_sum,
                "violated_rows": len(violated_rows),
                f"AVD_{metric_suffix}": violation_duration_sum / len(violated_rows) if violated_rows else 0.0,
                "positive_exposure_sum": positive_exposure_sum,
                "positive_exposure_rows": len(positive_exposure_rows),
                f"AER_{metric_suffix}": (
                    positive_exposure_sum / len(positive_exposure_rows)
                    if positive_exposure_rows
                    else 0.0
                ),
            }
        )
    return output_rows


def build_rq3_rows(raw_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return _build_group_rows(
        raw_rows,
        group_key="property",
        fallback_group="UnknownProperty",
        label_column="property",
        metric_suffix="p",
    )


def build_category_rows(raw_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return _build_group_rows(
        raw_rows,
        group_key="category",
        fallback_group="Uncategorized",
        label_column="category",
        metric_suffix="category",
    )


def build_suite_policy_rows(
    raw_rows: Sequence[Dict[str, Any]],
    task_suites: Dict[str, str],
) -> List[Dict[str, Any]]:
    episode_rows = []
    for (policy, task, episode), rows in sorted(
        _group_by(raw_rows, ("policy", "task", "episode")).items()
    ):
        episode_rows.append(
            {
                "policy": policy or "unknown_policy",
                "task": task or "unknown_task",
                "taskSuites": task_suites.get(task or "", "UnmappedSuite"),
                "episode": episode,
                "task_success": max(_to_int(row["task_success"]) for row in rows),
                "safety_violation": max(_row_violated(row) for row in rows),
            }
        )

    suite_policy_rows = []
    for (policy, task_suite), rows in sorted(
        _group_by(episode_rows, ("policy", "taskSuites")).items()
    ):
        n_episodes = len(rows)
        task_success_count = sum(_to_float(row["task_success"]) for row in rows)
        safety_violation_count = sum(_to_float(row["safety_violation"]) for row in rows)
        suite_policy_rows.append(
            {
                "policy": policy,
                "taskSuites": task_suite,
                "n_episodes": n_episodes,
                "task_success_count": task_success_count,
                "task_success_rate": task_success_count / n_episodes if n_episodes else 0.0,
                "safety_violation_count": safety_violation_count,
                "safety_violation_rate": (
                    safety_violation_count / n_episodes if n_episodes else 0.0
                ),
            }
        )
    return suite_policy_rows


def build_difficulty_policy_rows(
    raw_rows: Sequence[Dict[str, Any]],
    task_difficulties: Dict[str, str],
) -> List[Dict[str, Any]]:
    episode_rows = []
    for (policy, task, episode), rows in sorted(
        _group_by(raw_rows, ("policy", "task", "episode")).items()
    ):
        episode_rows.append(
            {
                "policy": policy or "unknown_policy",
                "task": task or "unknown_task",
                "difficulty_horizon": task_difficulties.get(
                    task or "",
                    "UnmappedDifficulty",
                ),
                "episode": episode,
                "task_success": max(_to_int(row["task_success"]) for row in rows),
                "safety_violation": max(_row_violated(row) for row in rows),
            }
        )

    difficulty_policy_rows = []
    for (policy, difficulty_horizon), rows in sorted(
        _group_by(episode_rows, ("policy", "difficulty_horizon")).items()
    ):
        n_episodes = len(rows)
        task_success_count = sum(_to_float(row["task_success"]) for row in rows)
        safety_violation_count = sum(_to_float(row["safety_violation"]) for row in rows)
        safe_success = sum(
            1
            for row in rows
            if _to_int(row["task_success"]) == 1 and _to_int(row["safety_violation"]) == 0
        )
        unsafe_success = sum(
            1
            for row in rows
            if _to_int(row["task_success"]) == 1 and _to_int(row["safety_violation"]) == 1
        )
        safe_fail = sum(
            1
            for row in rows
            if _to_int(row["task_success"]) == 0 and _to_int(row["safety_violation"]) == 0
        )
        unsafe_fail = sum(
            1
            for row in rows
            if _to_int(row["task_success"]) == 0 and _to_int(row["safety_violation"]) == 1
        )
        difficulty_policy_rows.append(
            {
                "policy": policy,
                "difficulty_horizon": difficulty_horizon,
                "n_episodes": n_episodes,
                "task_success_count": task_success_count,
                "task_success_rate": task_success_count / n_episodes if n_episodes else 0.0,
                "safety_violation_count": safety_violation_count,
                "safety_violation_rate": (
                    safety_violation_count / n_episodes if n_episodes else 0.0
                ),
                "safe-success": safe_success,
                "unsafe-success": unsafe_success,
                "safe-fail": safe_fail,
                "unsafe-fail": unsafe_fail,
            }
        )
    return difficulty_policy_rows


def build_difficulty_action_policy_rows(
    raw_rows: Sequence[Dict[str, Any]],
    task_difficulties: Dict[str, str],
    task_action_counts: Dict[str, float],
) -> List[Dict[str, Any]]:
    episode_rows = []
    for (policy, task, episode), rows in sorted(
        _group_by(raw_rows, ("policy", "task", "episode")).items()
    ):
        episode_rows.append(
            {
                "policy": policy or "unknown_policy",
                "task": task or "unknown_task",
                "difficulty_horizon": task_difficulties.get(
                    task or "",
                    "UnmappedDifficulty",
                ),
                "episode": episode,
                "task_success": max(_to_int(row["task_success"]) for row in rows),
                "safety_violation": max(_row_violated(row) for row in rows),
                "number_action": task_action_counts.get(task or "", 0.0),
            }
        )

    difficulty_action_rows = []
    for (policy, difficulty_horizon), rows in sorted(
        _group_by(episode_rows, ("policy", "difficulty_horizon")).items()
    ):
        n_episodes = len(rows)
        task_success_count = sum(_to_float(row["task_success"]) for row in rows)
        safety_violation_count = sum(_to_float(row["safety_violation"]) for row in rows)
        total_action_count = sum(_to_float(row["number_action"]) for row in rows)
        difficulty_action_rows.append(
            {
                "policy": policy,
                "difficulty_horizon": difficulty_horizon,
                "n_episodes": n_episodes,
                "task_success_count": task_success_count,
                "task_success_rate": task_success_count / n_episodes if n_episodes else 0.0,
                "safety_violation_count": safety_violation_count,
                "total_action_count": total_action_count,
                "violation_per_action": (
                    safety_violation_count / total_action_count
                    if total_action_count
                    else 0.0
                ),
            }
        )
    return difficulty_action_rows


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
) -> Path:
    csv_path = _plot_data_path(plot_data_dir, plot_path)
    columns = _columns_for_rows(rows)
    write_csv(csv_path, rows, columns)
    return csv_path


def _setup_matplotlib():
    cache_dir = Path("/p/safevla/pip_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_vr_dumbbell_plot(
    *,
    rq3_rows: Sequence[Dict[str, Any]],
    output_path: Path,
    label_key: str = "property",
    overall_key: str = "VR_p",
    success_key: str = "VRsucc_p",
    title: str = "Overall Violation Rate vs Unsafe-Success Rate by Property",
    y_label: str = "property",
    overall_label: str = "VR_p overall violation rate",
    success_label: str = "VRsucc_p successful-execution violation rate",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt = _setup_matplotlib()

    rows = sorted(
        rq3_rows,
        key=lambda row: (
            -_to_float(row[success_key]),
            -_to_float(row[overall_key]),
            str(row[label_key]),
        ),
    )
    labels = [str(row[label_key]) for row in rows]
    y_positions = list(range(len(rows)))
    vr_values = [_to_float(row[overall_key]) for row in rows]
    vrsucc_values = [_to_float(row[success_key]) for row in rows]

    fig_height = max(6.0, 0.35 * len(rows) + 2.0)
    fig, ax = plt.subplots(figsize=(8.5, fig_height))

    for y, vr, vrsucc in zip(y_positions, vr_values, vrsucc_values):
        ax.hlines(y, min(vr, vrsucc), max(vr, vrsucc), color="#9a9a9a", linewidth=1.6)

    ax.scatter(
        vr_values,
        y_positions,
        s=70,
        color="#2f6f9f",
        edgecolor="white",
        linewidth=0.7,
        label=overall_label,
        zorder=3,
    )
    ax.scatter(
        vrsucc_values,
        y_positions,
        s=70,
        color="#c96f53",
        edgecolor="white",
        linewidth=0.7,
        label=success_label,
        zorder=3,
    )

    ax.set_title(title)
    ax.set_xlabel("rate")
    ax.set_ylabel(y_label)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(-0.03, max(1.0, max(vr_values + vrsucc_values, default=0.0) * 1.08))
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _policy_order_key(policy: str) -> tuple[int, str]:
    normalized = policy.lower().replace("_", "-")
    suffix = normalized.rsplit("-", 1)[-1]
    order = {
        "pt": 0,
        "to": 1,
        "tpt": 2,
    }.get(suffix, 99)
    return order, normalized


def _difficulty_order_key(difficulty_horizon: str) -> tuple[int, str]:
    normalized = difficulty_horizon.strip().lower()
    order = {
        "atomic": 0,
        "short": 1,
        "medium": 2,
        "long": 3,
    }.get(normalized, 99)
    return order, normalized


def save_success_vs_violation_scatter_by_policy(
    *,
    policy_rows: Sequence[Dict[str, Any]],
    output_path: Path,
    label_key: str,
    label_order_key: Any,
    y_key: str,
    x_label: str,
    y_label: str,
    title: str,
    empty_text: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt = _setup_matplotlib()

    policies = sorted(
        {str(row["policy"]) for row in policy_rows},
        key=_policy_order_key,
    )
    if not policies:
        fig, ax = plt.subplots(figsize=(7.0, 4.8))
        ax.text(
            0.5,
            0.5,
            empty_text,
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        return

    rows_by_policy = {
        policy: [
            row for row in policy_rows
            if str(row.get("policy", "")) == policy
        ]
        for policy in policies
    }
    n_cols = min(3, len(policies))
    n_rows = (len(policies) + n_cols - 1) // n_cols
    fig, axes_grid = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.4 * n_cols, 4.9 * n_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = list(axes_grid.flat)

    for ax, policy in zip(axes, policies):
        rows = sorted(
            rows_by_policy[policy],
            key=lambda row: label_order_key(str(row[label_key])),
        )
        x_values = [_to_float(row["task_success_rate"]) for row in rows]
        y_values = [_to_float(row[y_key]) for row in rows]

        ax.scatter(
            x_values,
            y_values,
            s=90,
            color="#2f6f9f",
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        for index, (row, x, y) in enumerate(zip(rows, x_values, y_values)):
            ax.annotate(
                str(row.get(label_key, "")),
                (x, y),
                xytext=(6, 6 + 5 * (index % 2)),
                textcoords="offset points",
                fontsize=7,
            )

        n_episodes = sum(_to_int(row["n_episodes"]) for row in rows)
        ax.set_title(f"{policy}\n{n_episodes} rollouts")
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(True, alpha=0.25)

    for ax in axes[len(policies):]:
        ax.set_visible(False)

    for ax in axes[-n_cols:]:
        ax.set_xlabel(x_label)
    for ax in axes[::n_cols]:
        ax.set_ylabel(y_label)

    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build RQ3 property-level violation and exposure metrics."
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
        help=f"Directory for RQ3 output. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=DEFAULT_PLOT_DIR,
        help=f"Directory for RQ3 plots. Default: {DEFAULT_PLOT_DIR}",
    )
    parser.add_argument(
        "--plot-data-dir",
        type=Path,
        default=None,
        help="Directory for per-plot RQ3 CSV data. Default: <output-dir>/plotData",
    )
    parser.add_argument(
        "--suites-csv",
        type=Path,
        default=DEFAULT_SUITES_CSV,
        help=f"CSV mapping taskName to taskSuites. Default: {DEFAULT_SUITES_CSV}",
    )
    parser.add_argument(
        "--task-diff-csv",
        type=Path,
        default=DEFAULT_TASK_DIFF_CSV,
        help=(
            "CSV mapping taskID/taskName to difficulty_horizon. "
            f"Default: {DEFAULT_TASK_DIFF_CSV}"
        ),
    )
    parser.add_argument(
        "--pattern",
        default="metrics_*.csv",
        help="CSV glob to aggregate. Default: metrics_*.csv",
    )
    parser.add_argument(
        "--output-name",
        default="rq3_property_metrics.csv",
        help="Output CSV filename. Default: rq3_property_metrics.csv",
    )
    parser.add_argument(
        "--category-output-name",
        default="rq3_category_metrics.csv",
        help="Category-level output CSV filename. Default: rq3_category_metrics.csv",
    )
    parser.add_argument(
        "--suite-policy-output-name",
        default="rq3_suite_policy_success_violation_rates.csv",
        help=(
            "Task-suite-policy success and safety violation rate CSV filename. "
            "Default: rq3_suite_policy_success_violation_rates.csv"
        ),
    )
    parser.add_argument(
        "--difficulty-policy-output-name",
        default="rq3_difficulty_horizon_policy_success_violation_rates.csv",
        help=(
            "Difficulty-horizon-policy success and safety violation rate CSV filename. "
            "Default: rq3_difficulty_horizon_policy_success_violation_rates.csv"
        ),
    )
    parser.add_argument(
        "--difficulty-action-output-name",
        default="rq3_difficulty_horizon_policy_success_violation_per_action.csv",
        help=(
            "Difficulty-horizon-policy success and violation-per-action CSV filename. "
            "Default: rq3_difficulty_horizon_policy_success_violation_per_action.csv"
        ),
    )
    parser.add_argument(
        "--plot-name",
        default="rq3_vr_dumbbell.png",
        help="Dumbbell plot filename. Default: rq3_vr_dumbbell.png",
    )
    parser.add_argument(
        "--category-plot-name",
        default="rq3_category_vr_dumbbell.png",
        help="Category-level dumbbell plot filename. Default: rq3_category_vr_dumbbell.png",
    )
    parser.add_argument(
        "--success-violation-plot-name",
        "--suite-success-violation-plot-name",
        dest="suite_success_violation_plot_name",
        default="rq3_suite_success_vs_safety_violation_by_policy.png",
        help=(
            "Task-suite success vs safety violation scatter filename. "
            "Default: rq3_suite_success_vs_safety_violation_by_policy.png"
        ),
    )
    parser.add_argument(
        "--difficulty-success-violation-plot-name",
        default="rq3_difficulty_horizon_success_vs_safety_violation_by_policy.png",
        help=(
            "Difficulty-horizon success vs safety violation scatter filename. "
            "Default: rq3_difficulty_horizon_success_vs_safety_violation_by_policy.png"
        ),
    )
    parser.add_argument(
        "--difficulty-action-plot-name",
        default="rq3_difficulty_horizon_success_vs_violation_per_action_by_policy.png",
        help=(
            "Difficulty-horizon success vs violation-per-action scatter filename. "
            "Default: rq3_difficulty_horizon_success_vs_violation_per_action_by_policy.png"
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
    task_diff_csv = args.task_diff_csv.expanduser().resolve()

    metric_paths = _metric_csvs(processed_root, args.pattern)
    if not metric_paths:
        raise SystemExit(f"No metrics CSV files found under {processed_root} matching {args.pattern!r}.")

    raw_rows = _read_rows(metric_paths)
    task_suites = load_task_suites(suites_csv)
    task_difficulties = load_task_difficulties(task_diff_csv)
    task_action_counts = load_task_action_counts(task_diff_csv)
    rq3_rows = build_rq3_rows(raw_rows)
    category_rows = build_category_rows(raw_rows)
    suite_policy_rows = build_suite_policy_rows(raw_rows, task_suites)
    difficulty_policy_rows = build_difficulty_policy_rows(raw_rows, task_difficulties)
    difficulty_action_rows = build_difficulty_action_policy_rows(
        raw_rows,
        task_difficulties,
        task_action_counts,
    )
    output_path = output_dir / args.output_name
    category_output_path = output_dir / args.category_output_name
    suite_policy_output_path = output_dir / args.suite_policy_output_name
    difficulty_policy_output_path = output_dir / args.difficulty_policy_output_name
    difficulty_action_output_path = output_dir / args.difficulty_action_output_name
    plot_path = plot_dir / args.plot_name
    category_plot_path = plot_dir / args.category_plot_name
    suite_success_violation_plot_path = plot_dir / args.suite_success_violation_plot_name
    difficulty_success_violation_plot_path = (
        plot_dir / args.difficulty_success_violation_plot_name
    )
    difficulty_action_plot_path = plot_dir / args.difficulty_action_plot_name
    stale_paths = [
        output_dir / "rq3_task_policy_success_violation_rates.csv",
        plot_dir / "rq3_success_vs_safety_violation_by_policy.png",
        plot_dir / "rq3_success_vs_safety_violation_by_policy.csv",
        plot_path.with_suffix(".csv"),
        category_plot_path.with_suffix(".csv"),
        suite_success_violation_plot_path.with_suffix(".csv"),
        difficulty_success_violation_plot_path.with_suffix(".csv"),
        difficulty_action_plot_path.with_suffix(".csv"),
        plot_data_dir / "rq3_success_vs_safety_violation_by_policy.csv",
    ]
    current_paths = {
        suite_policy_output_path.resolve(),
        difficulty_policy_output_path.resolve(),
        difficulty_action_output_path.resolve(),
        suite_success_violation_plot_path.resolve(),
        difficulty_success_violation_plot_path.resolve(),
        difficulty_action_plot_path.resolve(),
        _plot_data_path(plot_data_dir, suite_success_violation_plot_path).resolve(),
        _plot_data_path(plot_data_dir, difficulty_success_violation_plot_path).resolve(),
        _plot_data_path(plot_data_dir, difficulty_action_plot_path).resolve(),
    }
    for stale_path in stale_paths:
        if stale_path.exists() and stale_path.resolve() not in current_paths:
            stale_path.unlink()
    plot_data_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_path, rq3_rows, OUTPUT_COLUMNS)
    write_csv(category_output_path, category_rows, CATEGORY_OUTPUT_COLUMNS)
    write_csv(suite_policy_output_path, suite_policy_rows, SUITE_POLICY_OUTPUT_COLUMNS)
    write_csv(
        difficulty_policy_output_path,
        difficulty_policy_rows,
        DIFFICULTY_POLICY_OUTPUT_COLUMNS,
    )
    write_csv(
        difficulty_action_output_path,
        difficulty_action_rows,
        DIFFICULTY_ACTION_POLICY_OUTPUT_COLUMNS,
    )
    write_plot_data_csv(plot_data_dir, plot_path, rq3_rows)
    write_plot_data_csv(plot_data_dir, category_plot_path, category_rows)
    write_plot_data_csv(plot_data_dir, suite_success_violation_plot_path, suite_policy_rows)
    write_plot_data_csv(
        plot_data_dir,
        difficulty_success_violation_plot_path,
        difficulty_policy_rows,
    )
    write_plot_data_csv(
        plot_data_dir,
        difficulty_action_plot_path,
        difficulty_action_rows,
    )
    save_vr_dumbbell_plot(rq3_rows=rq3_rows, output_path=plot_path)
    save_vr_dumbbell_plot(
        rq3_rows=category_rows,
        output_path=category_plot_path,
        label_key="category",
        overall_key="VR_category",
        success_key="VRsucc_category",
        title="Overall Violation Rate vs Unsafe-Success Rate by Property Category",
        y_label="property category",
        overall_label="VR_category overall violation rate",
        success_label="VRsucc_category successful-execution violation rate",
    )
    save_success_vs_violation_scatter_by_policy(
        policy_rows=suite_policy_rows,
        output_path=suite_success_violation_plot_path,
        label_key="taskSuites",
        label_order_key=lambda value: (0, value.lower()),
        y_key="safety_violation_rate",
        x_label="task suite success rate",
        y_label="safety violation rate",
        title="Task Suite Success Rate vs Safety Violation Rate by Policy",
        empty_text="No suite-policy rows",
    )
    save_success_vs_violation_scatter_by_policy(
        policy_rows=difficulty_policy_rows,
        output_path=difficulty_success_violation_plot_path,
        label_key="difficulty_horizon",
        label_order_key=_difficulty_order_key,
        y_key="safety_violation_rate",
        x_label="difficulty-horizon success rate",
        y_label="safety violation rate",
        title="Difficulty-Horizon Success Rate vs Safety Violation Rate by Policy",
        empty_text="No difficulty-policy rows",
    )
    save_success_vs_violation_scatter_by_policy(
        policy_rows=difficulty_action_rows,
        output_path=difficulty_action_plot_path,
        label_key="difficulty_horizon",
        label_order_key=_difficulty_order_key,
        y_key="violation_per_action",
        x_label="difficulty-horizon success rate",
        y_label="violation per action",
        title="Difficulty-Horizon Success Rate vs Violation per Action by Policy",
        empty_text="No difficulty-action rows",
    )

    print(f"Read {len(metric_paths)} metrics CSV file(s).")
    print(f"Raw rows: {len(raw_rows)}")
    print(f"RQ3 rows: {len(rq3_rows)}")
    print(f"RQ3 category rows: {len(category_rows)}")
    print(f"RQ3 suite-policy rows: {len(suite_policy_rows)}")
    print(f"RQ3 difficulty-horizon-policy rows: {len(difficulty_policy_rows)}")
    print(f"RQ3 difficulty-action rows: {len(difficulty_action_rows)}")
    print(f"Wrote: {output_path}")
    print(f"Wrote: {category_output_path}")
    print(f"Wrote: {suite_policy_output_path}")
    print(f"Wrote: {difficulty_policy_output_path}")
    print(f"Wrote: {difficulty_action_output_path}")
    print(f"Wrote plot data CSVs under: {plot_data_dir}")
    print(f"Wrote: {plot_path}")
    print(f"Wrote: {category_plot_path}")
    print(f"Wrote: {suite_success_violation_plot_path}")
    print(f"Wrote: {difficulty_success_violation_plot_path}")
    print(f"Wrote: {difficulty_action_plot_path}")


if __name__ == "__main__":
    main()
