from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


ANALYSIS_DIR = Path(__file__).resolve().parent
DEFAULT_PROCESSED_ROOT = ANALYSIS_DIR / "processedData"
DEFAULT_OUTPUT_DIR = DEFAULT_PROCESSED_ROOT / "extract"
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
    "policy",
    "task",
    "exposureRate",
    "ViolationRate",
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


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("_") or "blank"


def _matches(value: str, selected: str, *, ignore_case: bool) -> bool:
    if ignore_case:
        return value.casefold() == selected.casefold()
    return value == selected


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


def _metric_csvs(processed_root: Path, pattern: str, output_path: Path | None) -> List[Path]:
    excluded_dirs = {
        (processed_root / "RQ1").resolve(),
        (processed_root / "RQ2").resolve(),
        (processed_root / "RQ3").resolve(),
        (processed_root / "extract").resolve(),
        (processed_root / "statistics").resolve(),
    }
    resolved_output = output_path.resolve() if output_path is not None else None

    paths = []
    for path in sorted(processed_root.rglob(pattern)):
        if not path.is_file():
            continue
        resolved = path.resolve()
        if resolved_output is not None and resolved == resolved_output:
            continue
        if any(_is_relative_to(resolved, excluded_dir) for excluded_dir in excluded_dirs):
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
            for row in reader:
                rows.append({column: row.get(column, "") for column in RAW_REQUIRED_COLUMNS})
    return rows


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
            property_info = {
                "property": property_id or ltlf,
                "id": property_id,
                "ltlf": ltlf,
                "category": category,
                "tasks": applicable_tasks,
            }
            properties.append(property_info)

            for alias in (property_id, ltlf):
                if alias:
                    property_lookup[alias] = property_info

    return {
        "property_lookup": property_lookup,
        "properties": properties,
    }


def _property_info_for_row(
    row: Dict[str, Any],
    property_lookup: Dict[str, Dict[str, Any]],
) -> Dict[str, Any] | None:
    property_id = str(row.get("property") or "")
    return property_lookup.get(property_id) or property_lookup.get(_normalize_ltl(property_id))


def _available_categories(applicable_properties: Dict[str, Any]) -> List[str]:
    return sorted({str(row.get("category") or "") for row in applicable_properties["properties"]})


def build_extract_rows(
    raw_rows: Sequence[Dict[str, Any]],
    applicable_properties: Dict[str, Any],
    *,
    category: str,
    policy_filter: str | None,
    ignore_case: bool,
) -> List[Dict[str, Any]]:
    property_lookup: Dict[str, Dict[str, Any]] = applicable_properties["property_lookup"]
    properties: Sequence[Dict[str, Any]] = applicable_properties["properties"]

    applicable_property_ids_by_task: Dict[str, List[str]] = defaultdict(list)
    for property_info in properties:
        property_category = str(property_info.get("category") or "")
        if not _matches(property_category, category, ignore_case=ignore_case):
            continue

        property_id = str(property_info.get("id") or property_info.get("property") or "")
        for task in property_info.get("tasks", set()):
            applicable_property_ids_by_task[task].append(property_id)

    if not applicable_property_ids_by_task:
        available = ", ".join(_available_categories(applicable_properties)) or "<none>"
        raise SystemExit(
            f"No applicable properties found for category {category!r}.\n"
            f"Available categories: {available}"
        )

    episodes_by_policy_task: Dict[Tuple[str, str], set[str]] = defaultdict(set)
    exposure_sum_by_policy_task_property: Dict[Tuple[str, str, str], float] = defaultdict(float)
    violation_episodes_by_policy_task: Dict[Tuple[str, str], set[str]] = defaultdict(set)

    for row in raw_rows:
        policy = str(row.get("policy") or "unknown_policy")
        if policy_filter is not None and not _matches(policy, policy_filter, ignore_case=ignore_case):
            continue

        task = str(row.get("task") or "")
        episode = str(row.get("episode") or "")
        if not task or not episode:
            continue
        if task not in applicable_property_ids_by_task:
            continue

        episodes_by_policy_task[(policy, task)].add(episode)

        property_info = _property_info_for_row(row, property_lookup)
        if not property_info or task not in property_info["tasks"]:
            continue
        if not _matches(str(property_info.get("category") or ""), category, ignore_case=ignore_case):
            continue

        property_id = str(property_info.get("id") or property_info.get("property") or "")
        exposure_sum_by_policy_task_property[(policy, task, property_id)] += _to_float(
            row["exposure_rate"]
        )
        if _to_int(row["violation_count_per_property"]) > 0:
            violation_episodes_by_policy_task[(policy, task)].add(episode)

    output_rows = []
    for (policy, task), episodes in sorted(episodes_by_policy_task.items()):
        rollout_count = len(episodes)
        if rollout_count == 0:
            continue

        property_means = [
            exposure_sum_by_policy_task_property.get((policy, task, property_id), 0.0)
            / rollout_count
            for property_id in applicable_property_ids_by_task[task]
        ]
        violation_rollouts = len(violation_episodes_by_policy_task.get((policy, task), set()))
        output_rows.append(
            {
                "policy": policy,
                "task": task,
                "exposureRate": _mean(property_means),
                "ViolationRate": violation_rollouts / rollout_count,
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


def _default_output_path(output_dir: Path, category: str, policy: str | None) -> Path:
    policy_suffix = f"_{_safe_filename(policy)}" if policy else ""
    return output_dir / f"extract_{_safe_filename(category)}{policy_suffix}.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract per-task category exposure and violation rates."
    )
    parser.add_argument(
        "--category",
        required=True,
        help="Safety category to inspect, for example PreconditionSafe.",
    )
    parser.add_argument(
        "--policy",
        default=None,
        help="Optional policy filter, for example GR00T-tpt. Default: all policies.",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=DEFAULT_PROCESSED_ROOT,
        help=f"Root containing metrics CSVs. Default: {DEFAULT_PROCESSED_ROOT}",
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
        help="CSV glob to scan. Default: metrics_*.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Default: processedData/extract/extract_<category>.csv",
    )
    parser.add_argument(
        "--case-sensitive",
        action="store_true",
        help="Use exact case-sensitive matching for category and policy.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    applicable_path = _resolve_applicable_properties_path(
        args.applicable_properties_csv.expanduser().resolve()
    )
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else _default_output_path(DEFAULT_OUTPUT_DIR, args.category, args.policy).resolve()
    )

    processed_root = args.processed_root.expanduser().resolve()
    metric_paths = _metric_csvs(processed_root, args.pattern, output_path)
    if not metric_paths:
        raise SystemExit(
            f"No metrics CSV files found under {processed_root} matching {args.pattern!r}."
        )

    raw_rows = _read_rows(metric_paths)
    applicable_properties = load_applicable_properties(applicable_path)
    output_rows = build_extract_rows(
        raw_rows,
        applicable_properties,
        category=args.category,
        policy_filter=args.policy,
        ignore_case=not args.case_sensitive,
    )
    write_csv(output_path, output_rows, OUTPUT_COLUMNS)

    print(f"Read {len(metric_paths)} metrics CSV file(s).")
    print(f"Output rows: {len(output_rows)}")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
