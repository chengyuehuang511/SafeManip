from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ANALYSIS_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOT = ANALYSIS_DIR / "rawData"
DEFAULT_OUTPUT_PATH = ANALYSIS_DIR / "processedData" / "taskDifficulty.csv"
DEFAULT_RAW_OUTPUT_PATH = ANALYSIS_DIR / "processedData" / "taskDifficulty_raw.csv"


RAW_COLUMNS = (
    "task",
    "policy",
    "json_path",
    "skill_onset_count",
)


def _to_int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _parse_thresholds(value: str) -> List[float]:
    if not value.strip():
        return []

    thresholds = [float(part.strip()) for part in value.split(",") if part.strip()]
    for previous, current in zip(thresholds, thresholds[1:]):
        if current <= previous:
            raise SystemExit("--difficulty-thresholds must be strictly increasing.")
    return thresholds


def _parse_bin_labels(value: str, thresholds: Sequence[float]) -> List[str]:
    expected_count = len(thresholds) + 1
    if not value.strip():
        if not thresholds:
            return ["all"]
        return [f"bin_{index}" for index in range(expected_count)]

    labels = [part.strip() for part in value.split(",") if part.strip()]
    if len(labels) != expected_count:
        raise SystemExit(
            "--difficulty-bins must have exactly one more label than "
            "--difficulty-thresholds. "
            f"Expected {expected_count}, got {len(labels)}."
        )
    return labels


def _classify_difficulty(
    value: float,
    thresholds: Sequence[float],
    labels: Sequence[str],
) -> str:
    for threshold, label in zip(thresholds, labels):
        if value <= threshold:
            return label
    return labels[-1]


def _json_files(input_path: Path, pattern: str) -> List[Path]:
    if input_path.is_file():
        return [input_path] if input_path.match(pattern) else []
    return sorted(path for path in input_path.rglob(pattern) if path.is_file())


def _policy_for_path(json_path: Path, input_root: Path) -> str:
    input_root = input_root.resolve()
    if input_root.is_file():
        input_root = input_root.parent

    try:
        relative_to_raw = json_path.relative_to(DEFAULT_INPUT_ROOT.resolve())
        if len(relative_to_raw.parts) > 1:
            return relative_to_raw.parts[0]
    except ValueError:
        pass

    if input_root.parent.name == "rawData":
        return input_root.name
    if input_root.parent.parent.name == "rawData":
        return input_root.parent.name

    try:
        relative = json_path.relative_to(input_root)
    except ValueError:
        return json_path.parent.name
    if input_root.name == "rawData" and len(relative.parts) > 1:
        return relative.parts[0]
    if len(relative.parts) > 2:
        return relative.parts[0]
    return json_path.parent.name


def _task_for_path(json_path: Path, input_root: Path) -> str:
    input_root = input_root.resolve()
    if input_root.is_file():
        return json_path.parent.name

    try:
        relative_to_raw = json_path.relative_to(DEFAULT_INPUT_ROOT.resolve())
        if len(relative_to_raw.parts) > 1:
            return relative_to_raw.parts[1]
    except ValueError:
        pass

    try:
        relative = json_path.relative_to(input_root)
    except ValueError:
        return json_path.parent.name

    if input_root.name == "rawData" and len(relative.parts) > 1:
        return relative.parts[1]
    if input_root.parent.name == "rawData" and len(relative.parts) > 1:
        return relative.parts[0]
    if input_root.parent.parent.name == "rawData":
        return input_root.name
    if len(relative.parts) > 1:
        return relative.parts[0]
    return json_path.parent.name


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _task_for_monitor(monitor: Dict[str, Any], json_path: Path, input_root: Path) -> str:
    replay = monitor.get("replay_summary")
    replay = replay if isinstance(replay, dict) else {}
    task = _first_present(
        monitor.get("task_name"),
        replay.get("task_name"),
        monitor.get("task"),
        replay.get("task"),
    )
    return str(task or _task_for_path(json_path, input_root) or "unknown_task")


def _iter_dicts(value: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _iter_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_dicts(child)


def _skill_onset_count_from_evidence(evidence: Dict[str, Any]) -> int:
    return sum(
        _to_int(value)
        for key, value in evidence.items()
        if key.startswith("skill_") and key.endswith("_onset_candidate_count")
    )


def _limit_json_files_per_task_policy(
    json_paths: Sequence[Path],
    input_root: Path,
    max_files_per_group: Optional[int],
) -> List[Path]:
    if max_files_per_group is None:
        return list(json_paths)

    grouped: Dict[Tuple[str, str], List[Path]] = defaultdict(list)
    for json_path in json_paths:
        grouped[
            (
                _policy_for_path(json_path, input_root),
                _task_for_path(json_path, input_root),
            )
        ].append(json_path)

    selected: List[Path] = []
    for key in sorted(grouped):
        selected.extend(sorted(grouped[key])[:max_files_per_group])
    return sorted(selected)


def build_task_difficulty_rows(
    json_paths: Sequence[Path],
    input_root: Path,
    *,
    progress_every: int,
) -> List[Dict[str, Any]]:
    raw_rows = []
    for index, json_path in enumerate(json_paths, start=1):
        with json_path.open("r", encoding="utf-8") as handle:
            monitor = json.load(handle)
        monitor = monitor if isinstance(monitor, dict) else {}

        task = _task_for_monitor(monitor, json_path, input_root)
        policy = _policy_for_path(json_path, input_root)
        skill_onset_count = max(
            (
                _skill_onset_count_from_evidence(item)
                for item in _iter_dicts(monitor)
            ),
            default=0,
        )

        raw_rows.append(
            {
                "task": task,
                "policy": policy,
                "json_path": str(json_path),
                "skill_onset_count": skill_onset_count,
            }
        )

        if progress_every and (
            index == 1 or index % progress_every == 0 or index == len(json_paths)
        ):
            print(f"Processed {index}/{len(json_paths)} JSON file(s)...", flush=True)

    return raw_rows


def summarize_task_difficulty_rows(
    raw_rows: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[str], int]:
    values_by_task_policy: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    policies = set()

    for row in raw_rows:
        task = str(row.get("task") or "unknown_task")
        policy = str(row.get("policy") or "unknown_policy")
        policies.add(policy)
        values_by_task_policy[(task, policy)].append(
            float(row.get("skill_onset_count") or 0.0)
        )

    sorted_policies = sorted(policies)
    tasks = sorted({task for task, _ in values_by_task_policy})
    rows: List[Dict[str, Any]] = []
    for task in tasks:
        row: Dict[str, Any] = {"task": task}
        policy_means = []
        for policy in sorted_policies:
            values = values_by_task_policy.get((task, policy), [])
            if values:
                mean_value = _mean(values)
                row[policy] = mean_value
                policy_means.append(mean_value)
            else:
                row[policy] = ""
        row["average_across_policies"] = _mean(policy_means)
        rows.append(row)

    return rows, sorted_policies, sum(len(values) for values in values_by_task_policy.values())


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Cached raw task difficulty CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in RAW_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"{path} is missing required columns: {missing}")
        return [{column: row.get(column, "") for column in RAW_COLUMNS} for row in reader]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a task difficulty table: one row per task, one column per policy, "
            "filled with average skill-onset count per JSON rollout."
        )
    )
    parser.add_argument(
        "--input",
        "--input-root",
        dest="input_root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help=f"Raw JSON file or folder to scan. Default: {DEFAULT_INPUT_ROOT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output CSV path. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=DEFAULT_RAW_OUTPUT_PATH,
        help=(
            "Cached long-format raw output CSV path. "
            f"Default: {DEFAULT_RAW_OUTPUT_PATH}"
        ),
    )
    parser.add_argument(
        "--raw-input",
        type=Path,
        default=DEFAULT_RAW_OUTPUT_PATH,
        help=(
            "Cached long-format raw input CSV path used when not rebuilding. "
            f"Default: {DEFAULT_RAW_OUTPUT_PATH}"
        ),
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Rescan JSON files and rebuild the raw cache before writing the summary.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only read the cached raw CSV and write the summary. This is the default.",
    )
    parser.add_argument(
        "--pattern",
        default="*.json",
        help="File glob to scan recursively under the input root. Default: *.json",
    )
    parser.add_argument(
        "--max-json-per-task-policy",
        "--max-json-files-per-task-policy",
        dest="max_json_per_task_policy",
        type=int,
        default=None,
        help=(
            "Read at most this many JSON files for each policy/task group. "
            "Files are selected deterministically by sorted path. Default: no limit."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help=(
            "Print progress after this many processed JSON files. "
            "Use 0 to disable progress updates. Default: 100."
        ),
    )
    parser.add_argument(
        "--difficulty-thresholds",
        "--thresholds",
        default="",
        help=(
            "Comma-separated increasing thresholds used to classify "
            "average_across_policies. Example: 20,50"
        ),
    )
    parser.add_argument(
        "--difficulty-bins",
        "--bin-labels",
        default="",
        help=(
            "Comma-separated labels for the bins. Must have one more label than "
            "the number of thresholds. Example with thresholds 20,50: easy,medium,hard"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    raw_output_path = args.raw_output.expanduser().resolve()
    raw_input_path = args.raw_input.expanduser().resolve()

    if args.max_json_per_task_policy is not None and args.max_json_per_task_policy <= 0:
        raise SystemExit("--max-json-per-task-policy must be a positive integer.")
    if args.progress_every < 0:
        raise SystemExit("--progress-every must be 0 or a positive integer.")
    thresholds = _parse_thresholds(args.difficulty_thresholds)
    bin_labels = _parse_bin_labels(args.difficulty_bins, thresholds)

    if args.rebuild_cache:
        json_paths = _json_files(input_root, args.pattern)
        if not json_paths:
            raise SystemExit(f"No JSON files found under {input_root} matching {args.pattern!r}.")

        total_json_paths = len(json_paths)
        json_paths = _limit_json_files_per_task_policy(
            json_paths,
            input_root,
            args.max_json_per_task_policy,
        )

        print(f"JSON files found: {total_json_paths}", flush=True)
        print(f"JSON files selected for processing: {len(json_paths)}", flush=True)
        if len(json_paths) != total_json_paths:
            print(
                "Limited input JSON files: "
                f"selected {len(json_paths)} of {total_json_paths} total "
                f"using max {args.max_json_per_task_policy} per policy/task.",
                flush=True,
            )

        raw_rows = build_task_difficulty_rows(
            json_paths,
            input_root,
            progress_every=args.progress_every,
        )
        write_csv(raw_output_path, raw_rows, RAW_COLUMNS)
        print(f"Wrote raw cache: {raw_output_path}")
    else:
        raw_rows = read_csv(raw_input_path)
        print(f"Read raw cache: {raw_input_path}")

    rows, policies, processed_count = summarize_task_difficulty_rows(raw_rows)
    for row in rows:
        average = float(row.get("average_across_policies") or 0.0)
        row["difficulty_bin"] = _classify_difficulty(average, thresholds, bin_labels)

    columns = ["task", *policies, "average_across_policies", "difficulty_bin"]
    write_csv(output_path, rows, columns)

    print(f"Cached rollout rows used: {processed_count}")
    print(f"Tasks: {len(rows)}")
    print(f"Policies: {len(policies)}")
    if thresholds:
        print(f"Difficulty thresholds: {', '.join(str(value) for value in thresholds)}")
        print(f"Difficulty bins: {', '.join(bin_labels)}")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
