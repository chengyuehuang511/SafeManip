from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


ANALYSIS_DIR = Path(__file__).resolve().parent
DEFAULT_PROCESSED_ROOT = ANALYSIS_DIR / "processedData"
DEFAULT_RAW_ROOT = ANALYSIS_DIR / "rawData"
DEFAULT_OUTPUT_DIR = DEFAULT_PROCESSED_ROOT / "statistics"
DEFAULT_ID_CSV = ANALYSIS_DIR / "id.csv"


REQUIRED_COLUMNS = (
    "task",
    "episode",
    "policy",
    "property",
    "category",
    "exposure_rate",
    "violation_count_per_property",
    "task_success",
)


OUTPUT_COLUMNS = (
    "rank",
    "rank_metric",
    "outcome",
    "policy",
    "category",
    "task",
    "video_id",
    "episode",
    "total_exposure_rate",
    "total_violation_count",
    "task_success",
    "safety_violation",
    "violation_start_frames",
    "violation_start_frame_details",
    "max_property_exposure_rate",
    "max_property_violation_count",
    "num_property_rows",
    "raw_json",
    "source_csvs",
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


def _normalize_ltl(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("_") or "blank"


def _episode_from_filename(json_path: Path) -> str:
    match = re.search(r"(?:episode|privileged_information)[_-](\d+)", json_path.stem)
    if match:
        return match.group(1)
    match = re.search(r"(\d+)", json_path.stem)
    return match.group(1) if match else ""


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
        (processed_root / "statistics").resolve(),
    }
    paths = []
    for path in sorted(processed_root.rglob(pattern)):
        if not path.is_file():
            continue
        resolved = path.resolve()
        if any(_is_relative_to(resolved, excluded_dir) for excluded_dir in excluded_dirs):
            continue
        paths.append(path)
    return paths


def _read_rows(paths: Iterable[Path]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = [
                column for column in REQUIRED_COLUMNS if column not in (reader.fieldnames or [])
            ]
            if missing:
                raise SystemExit(f"{path} is missing required columns: {missing}")
            for row in reader:
                output_row = {column: row.get(column, "") for column in REQUIRED_COLUMNS}
                output_row["_source_csv"] = str(path)
                rows.append(output_row)
    return rows


def load_property_lookup(id_csv_path: Path) -> Dict[str, Dict[str, str]]:
    if not id_csv_path.exists():
        raise SystemExit(f"Property ID CSV not found: {id_csv_path}")

    with id_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit(f"Property ID CSV is empty: {id_csv_path}")

        normalized_fields = {field.lower(): field for field in reader.fieldnames}
        ltl_field = normalized_fields.get("ltlf")
        id_field = normalized_fields.get("id")
        category_field = normalized_fields.get("category")
        if not ltl_field or not id_field or not category_field:
            raise SystemExit(
                f"Expected columns LTLf, ID, category in property ID CSV: {id_csv_path}"
            )

        lookup: Dict[str, Dict[str, str]] = {}
        for row in reader:
            ltl = _normalize_ltl(row.get(ltl_field))
            if not ltl:
                continue
            lookup[ltl] = {
                "id": str(row.get(id_field) or "").strip(),
                "category": str(row.get(category_field) or "").strip(),
            }
    return lookup


def _matches(value: str, selected: str, *, ignore_case: bool) -> bool:
    if ignore_case:
        return value.casefold() == selected.casefold()
    return value == selected


def _available_values(rows: Sequence[Dict[str, str]], column: str) -> List[str]:
    return sorted({str(row.get(column) or "") for row in rows})


def _find_json_path(raw_root: Path, policy: str, task: str, episode: str) -> Path | None:
    direct_dir = raw_root / policy / task
    candidates = []
    if direct_dir.exists():
        candidates.extend(sorted(direct_dir.glob("*_monitor.json")))
        candidates.extend(sorted(direct_dir.glob("*.json")))

    if not candidates:
        candidates = [
            path
            for path in sorted(raw_root.rglob("*_monitor.json"))
            if policy in path.parts and task in path.parts
        ]

    for path in candidates:
        if _episode_from_filename(path) == str(episode):
            return path
    return None


def _violation_start_frames_from_entry(entry: Dict[str, Any]) -> List[int]:
    frames: List[int] = []
    repeated = entry.get("repeated")
    repeated = repeated if isinstance(repeated, dict) else {}
    episodes = repeated.get("repeated_violation_episodes")
    if isinstance(episodes, list):
        for episode in episodes:
            if isinstance(episode, dict) and episode.get("start_frame") is not None:
                frames.append(_to_int(episode.get("start_frame")))

    if frames:
        return sorted(set(frames))

    original = entry.get("original")
    original = original if isinstance(original, dict) else entry
    first_frame = original.get("first_non_accepting_frame")
    if first_frame is not None:
        frames.append(_to_int(first_frame))
    return sorted(set(frames))


def _violation_start_frames_for_json(
    *,
    json_path: Path | None,
    category: str,
    property_lookup: Dict[str, Dict[str, str]],
    ignore_case: bool,
) -> Tuple[str, str]:
    if json_path is None or not json_path.exists():
        return "", ""

    with json_path.open("r", encoding="utf-8") as handle:
        monitor = json.load(handle)

    frames_by_property: Dict[str, List[int]] = defaultdict(list)
    for entry in monitor.get("violations") or ():
        if not isinstance(entry, dict):
            continue

        original = entry.get("original")
        original = original if isinstance(original, dict) else entry
        property_ltl = _normalize_ltl(entry.get("ltl") or original.get("ltl"))
        property_info = property_lookup.get(property_ltl, {})
        entry_category = str(property_info.get("category") or "").strip()
        if not _matches(entry_category, category, ignore_case=ignore_case):
            continue

        property_id = str(property_info.get("id") or property_ltl)
        frames_by_property[property_id].extend(_violation_start_frames_from_entry(entry))

    all_frames = sorted({frame for frames in frames_by_property.values() for frame in frames})
    details = []
    for property_id, frames in sorted(frames_by_property.items()):
        unique_frames = sorted(set(frames))
        details.append(f"{property_id}:{'|'.join(str(frame) for frame in unique_frames)}")

    return ";".join(str(frame) for frame in all_frames), ";".join(details)


def _episode_rows(
    rows: Sequence[Dict[str, str]],
    *,
    policy: str,
    category: str,
    ignore_case: bool,
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        row_policy = str(row.get("policy") or "")
        row_category = str(row.get("category") or "")
        if not _matches(row_policy, policy, ignore_case=ignore_case):
            continue
        if not _matches(row_category, category, ignore_case=ignore_case):
            continue

        task = str(row.get("task") or "")
        episode = str(row.get("episode") or "")
        grouped[(row_policy, row_category, task, episode)].append(row)

    episode_rows: List[Dict[str, Any]] = []
    for (row_policy, row_category, task, episode), grouped_rows in sorted(grouped.items()):
        exposure_values = [_to_float(row["exposure_rate"]) for row in grouped_rows]
        violation_values = [
            _to_int(row["violation_count_per_property"]) for row in grouped_rows
        ]
        task_success = max(_to_int(row["task_success"]) for row in grouped_rows)
        total_violation_count = sum(violation_values)
        episode_rows.append(
            {
                "policy": row_policy,
                "category": row_category,
                "task": task,
                "video_id": episode,
                "episode": episode,
                "total_exposure_rate": sum(exposure_values),
                "total_violation_count": total_violation_count,
                "task_success": task_success,
                "safety_violation": int(total_violation_count > 0),
                "max_property_exposure_rate": max(exposure_values) if exposure_values else 0.0,
                "max_property_violation_count": max(violation_values) if violation_values else 0,
                "num_property_rows": len(grouped_rows),
                "source_csvs": ";".join(
                    sorted({str(row.get("_source_csv") or "") for row in grouped_rows})
                ),
            }
        )
    return episode_rows


def _rank_rows(episode_rows: Sequence[Dict[str, Any]], *, top_n: int) -> List[Dict[str, Any]]:
    ranked = sorted(
        episode_rows,
        key=lambda row: (
            -_to_float(row["total_exposure_rate"]),
            -_to_float(row["total_violation_count"]),
            str(row["task"]),
            str(row["episode"]),
        ),
    )

    output_rows = []
    for rank, row in enumerate(ranked[:top_n], start=1):
        output_row = dict(row)
        output_row["rank"] = rank
        output_row["rank_metric"] = "total_exposure_rate"
        output_row["outcome"] = "unsafe-success"
        output_rows.append(output_row)
    return output_rows


def build_statistics_rows(
    rows: Sequence[Dict[str, str]],
    *,
    policy: str,
    category: str,
    top_n: int,
    ignore_case: bool,
    raw_root: Path,
    property_lookup: Dict[str, Dict[str, str]],
) -> List[Dict[str, Any]]:
    episode_rows = _episode_rows(
        rows,
        policy=policy,
        category=category,
        ignore_case=ignore_case,
    )
    if not episode_rows:
        available_policies = ", ".join(_available_values(rows, "policy")) or "<none>"
        available_categories = ", ".join(_available_values(rows, "category")) or "<none>"
        raise SystemExit(
            "No matching rows found.\n"
            f"Requested policy={policy!r}, category={category!r}.\n"
            f"Available policies: {available_policies}\n"
            f"Available categories: {available_categories}"
        )

    unsafe_success_rows = [
        row
        for row in episode_rows
        if _to_int(row["task_success"]) == 1 and _to_int(row["safety_violation"]) == 1
    ]
    output_rows = _rank_rows(unsafe_success_rows, top_n=top_n)
    for row in output_rows:
        raw_json = _find_json_path(
            raw_root,
            str(row["policy"]),
            str(row["task"]),
            str(row["episode"]),
        )
        frames, details = _violation_start_frames_for_json(
            json_path=raw_json,
            category=category,
            property_lookup=property_lookup,
            ignore_case=ignore_case,
        )
        row["raw_json"] = str(raw_json) if raw_json is not None else ""
        row["violation_start_frames"] = frames
        row["violation_start_frame_details"] = details
    return output_rows


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _default_output_path(output_dir: Path, policy: str, category: str) -> Path:
    return output_dir / f"statistics_{_safe_filename(policy)}_{_safe_filename(category)}.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find unsafe-success rollout/video IDs with the largest category "
            "exposure rate for a selected policy."
        )
    )
    parser.add_argument(
        "--policy",
        required=True,
        help="Policy to filter, for example GR00T-tpt.",
    )
    parser.add_argument(
        "--category",
        required=True,
        help="Property category to filter, for example Containment.",
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
        help=f"Root containing raw monitor JSONs. Default: {DEFAULT_RAW_ROOT}",
    )
    parser.add_argument(
        "--id-csv",
        type=Path,
        default=DEFAULT_ID_CSV,
        help=f"Property ID CSV used to map raw LTLf to categories. Default: {DEFAULT_ID_CSV}",
    )
    parser.add_argument(
        "--pattern",
        default="metrics_*.csv",
        help="CSV glob to scan. Default: metrics_*.csv",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=1,
        help="Number of top unsafe-success videos to output. Default: 1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Default: processedData/statistics/statistics_<policy>_<category>.csv",
    )
    parser.add_argument(
        "--case-sensitive",
        action="store_true",
        help="Use exact case-sensitive matching for policy and category.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_n <= 0:
        raise SystemExit("--top-n must be positive.")

    processed_root = args.processed_root.expanduser().resolve()
    raw_root = args.raw_root.expanduser().resolve()
    id_csv = args.id_csv.expanduser().resolve()
    metric_paths = _metric_csvs(processed_root, args.pattern)
    if not metric_paths:
        raise SystemExit(
            f"No metrics CSV files found under {processed_root} matching {args.pattern!r}."
        )

    rows = _read_rows(metric_paths)
    property_lookup = load_property_lookup(id_csv)
    output_rows = build_statistics_rows(
        rows,
        policy=args.policy,
        category=args.category,
        top_n=args.top_n,
        ignore_case=not args.case_sensitive,
        raw_root=raw_root,
        property_lookup=property_lookup,
    )

    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else _default_output_path(DEFAULT_OUTPUT_DIR, args.policy, args.category).resolve()
    )
    write_csv(output_path, output_rows, OUTPUT_COLUMNS)

    print(f"Read {len(metric_paths)} metrics CSV file(s).")
    print(f"Matched unsafe-success output rows: {len(output_rows)}")
    for row in output_rows:
        print(
            f"unsafe-success #{row['rank']}: "
            f"task={row['task']} video_id={row['video_id']} "
            f"total_exposure_rate={row['total_exposure_rate']} "
            f"total_violation_count={row['total_violation_count']} "
            f"violation_start_frames={row['violation_start_frames']} "
            f"policy={row['policy']} category={row['category']}"
        )
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
