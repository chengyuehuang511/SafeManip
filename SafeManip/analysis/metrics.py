from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ANALYSIS_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOT = ANALYSIS_DIR / "rawData"
DEFAULT_OUTPUT_ROOT = ANALYSIS_DIR / "processedData"
DEFAULT_OUTPUT_NAME = "metrics_{policy}.csv"
DEFAULT_ID_CSV = ANALYSIS_DIR / "id.csv"


COLUMNS = (
    "task",
    "episode",
    "policy",
    "property",
    "category",
    "violation_count_per_property",
    "violations_per_skill_onset",
    "violation_duration_per_property",
    "exposure_rate",
    "task_success",
    "safety_satisfaction",
    "safe_succes_per_property",
)


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _to_bool(value: Any) -> Optional[bool]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "t", "yes", "y"}:
            return True
        if normalized in {"0", "false", "f", "no", "n"}:
            return False
    return None


def _to_int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _indicator(value: Optional[bool]) -> Any:
    return "" if value is None else int(value)


def _safe_divide(numerator: float, denominator: float) -> Any:
    if denominator == 0:
        return ""
    return numerator / denominator


def _skill_onset_count_from_evidence(evidence: Any) -> int:
    if not isinstance(evidence, dict):
        return 0
    return sum(
        _to_int(value)
        for key, value in evidence.items()
        if key.startswith("skill_") and key.endswith("_onset_candidate_count")
    )


def _normalize_ltl(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


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


def _json_files(input_path: Path, pattern: str) -> List[Path]:
    if input_path.is_file():
        return [input_path] if input_path.match(pattern) else []
    return sorted(path for path in input_path.rglob(pattern) if path.is_file())


def _task_for_path(json_path: Path, input_root: Path) -> str:
    try:
        relative_to_default_raw = json_path.relative_to(DEFAULT_INPUT_ROOT.resolve())
        if len(relative_to_default_raw.parts) > 1:
            return relative_to_default_raw.parts[1]
    except ValueError:
        pass

    input_root = input_root.resolve()
    if input_root.is_file():
        return json_path.parent.name

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


def _limit_json_files_per_task_policy(
    json_paths: Sequence[Path],
    input_root: Path,
    max_files_per_group: Optional[int],
) -> List[Path]:
    if max_files_per_group is None:
        return list(json_paths)

    grouped: Dict[Tuple[str, str], List[Path]] = {}
    for json_path in json_paths:
        key = (
            _policy_for_path(json_path, input_root),
            _task_for_path(json_path, input_root),
        )
        grouped.setdefault(key, []).append(json_path)

    selected: List[Path] = []
    for key in sorted(grouped):
        selected.extend(sorted(grouped[key])[:max_files_per_group])
    return sorted(selected)


def _filter_json_files_by_policy(
    json_paths: Sequence[Path],
    input_root: Path,
    policy: Optional[str],
) -> List[Path]:
    if not policy:
        return list(json_paths)
    return [
        json_path
        for json_path in json_paths
        if _policy_for_path(json_path, input_root) == policy
    ]


def _policy_for_path(json_path: Path, input_root: Path) -> str:
    try:
        relative_to_default_raw = json_path.relative_to(DEFAULT_INPUT_ROOT.resolve())
        if len(relative_to_default_raw.parts) > 1:
            return relative_to_default_raw.parts[0]
    except ValueError:
        pass

    input_root = input_root.resolve()
    if input_root.is_file():
        input_root = input_root.parent
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


def _safe_dir_name(value: Any) -> str:
    name = re.sub(r"[/:\\]+", "_", str(value or "unknown_task")).strip()
    return name or "unknown_task"


def _episode_from_filename(json_path: Path) -> Optional[int]:
    match = re.search(r"(?:episode|privileged_information)[_-](\d+)", json_path.stem)
    if match:
        return int(match.group(1))
    match = re.search(r"(\d+)", json_path.stem)
    if match:
        return int(match.group(1))
    return None


def _original_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    original = entry.get("original")
    return original if isinstance(original, dict) else entry


def _repeated_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    repeated = entry.get("repeated")
    return repeated if isinstance(repeated, dict) else {}


def _monitor_entries(monitor: Dict[str, Any]) -> List[Tuple[Dict[str, Any], bool]]:
    entries: List[Tuple[Dict[str, Any], bool]] = []
    for entry in monitor.get("satisfied") or ():
        if isinstance(entry, dict):
            entries.append((entry, True))
    for entry in monitor.get("violations") or ():
        if isinstance(entry, dict):
            entries.append((entry, False))
    return entries


def _episode_duration(episode: Dict[str, Any]) -> int:
    if "duration_frames" in episode:
        return _to_int(episode.get("duration_frames"))

    start = episode.get("start_frame")
    end = episode.get("end_frame")
    if start is None or end is None:
        return 0
    return max(0, _to_int(end) - _to_int(start) + 1)


def _violation_count_and_duration(
    entry: Dict[str, Any],
    *,
    property_satisfied: bool,
    total_frames: int,
) -> Tuple[int, int]:
    if property_satisfied:
        return 0, 0

    original = _original_entry(entry)
    repeated = _repeated_entry(entry)
    episodes = repeated.get("repeated_violation_episodes")
    if isinstance(episodes, list) and episodes:
        durations = [
            _episode_duration(episode)
            for episode in episodes
            if isinstance(episode, dict)
        ]
        repeated_count = _to_int(repeated.get("repeated_violation_count"), len(durations))
        return repeated_count or len(durations), sum(durations)

    repeated_count = _to_int(repeated.get("repeated_violation_count"), 0)
    first_frame = original.get("first_non_accepting_frame")
    if first_frame is not None:
        duration = max(0, total_frames - _to_int(first_frame))
        return repeated_count or 1, duration

    return repeated_count or 1, 0


def _skill_onset_count(entry: Dict[str, Any]) -> int:
    original = _original_entry(entry)
    repeated = _repeated_entry(entry)
    evidence_counts = [
        _skill_onset_count_from_evidence(
            original.get("first_non_accepting_violation_evidence")
        ),
        _skill_onset_count_from_evidence(original.get("final_violation_evidence")),
        _skill_onset_count_from_evidence(entry.get("final_violation_evidence")),
    ]

    episodes = repeated.get("repeated_violation_episodes")
    if isinstance(episodes, list):
        for episode in episodes:
            if not isinstance(episode, dict):
                continue
            evidence_counts.extend(
                [
                    _skill_onset_count_from_evidence(
                        episode.get("start_violation_evidence")
                    ),
                    _skill_onset_count_from_evidence(
                        episode.get("recovery_violation_evidence")
                    ),
                ]
            )

    return max(evidence_counts, default=0)


def build_rows_for_json(
    json_path: Path,
    input_root: Path,
    property_lookup: Dict[str, Dict[str, str]],
) -> List[Dict[str, Any]]:
    with json_path.open("r", encoding="utf-8") as handle:
        monitor = json.load(handle)

    replay = monitor.get("replay_summary")
    replay = replay if isinstance(replay, dict) else {}
    task_success_bool = _to_bool(_first_present(replay.get("success"), monitor.get("success")))
    total_frames = _to_int(
        _first_present(
            monitor.get("num_frames"),
            replay.get("replayed_episode_length"),
            replay.get("expected_episode_length"),
        )
    )

    task = _first_present(
        monitor.get("task_name"),
        replay.get("task_name"),
        monitor.get("task"),
        replay.get("task"),
        "",
    )
    episode = _first_present(
        monitor.get("episode"),
        monitor.get("episode_id"),
        monitor.get("episode_idx"),
        replay.get("episode"),
        replay.get("episode_id"),
        replay.get("episode_idx"),
        _episode_from_filename(json_path),
        "",
    )
    policy = _policy_for_path(json_path, input_root)

    rows: List[Dict[str, Any]] = []
    for property_idx, (entry, property_satisfied) in enumerate(_monitor_entries(monitor)):
        original = _original_entry(entry)
        property_ltl = _normalize_ltl(_first_present(entry.get("ltl"), original.get("ltl"), ""))
        property_info = property_lookup.get(property_ltl, {})
        violation_count, violation_duration = _violation_count_and_duration(
            entry,
            property_satisfied=property_satisfied,
            total_frames=total_frames,
        )
        skill_onset_count = _skill_onset_count(entry)
        safety_satisfaction = int(property_satisfied)
        task_success = _indicator(task_success_bool)
        safe_success = int(task_success_bool is True and property_satisfied)
        rows.append(
            {
                "task": task,
                "episode": episode,
                "policy": policy,
                "property": property_info.get("id") or property_ltl,
                "category": property_info.get("category", ""),
                "violation_count_per_property": violation_count,
                "violations_per_skill_onset": _safe_divide(
                    violation_count,
                    skill_onset_count,
                ),
                "violation_duration_per_property": violation_duration,
                "exposure_rate": _safe_divide(violation_duration, total_frames),
                "task_success": task_success,
                "safety_satisfaction": safety_satisfaction,
                "safe_succes_per_property": safe_success,
            }
        )
    return rows


def _normalize_cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    return value


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _normalize_cell(row.get(key, "")) for key in fieldnames})


def clear_old_csvs(
    output_root: Path,
    *,
    policy: Optional[str] = None,
    output_name_template: str = DEFAULT_OUTPUT_NAME,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    if policy:
        filename = _output_filename(output_name_template, policy)
        paths = output_root.rglob(filename)
    else:
        paths = output_root.rglob("*.csv")
    for path in paths:
        path.unlink()


def _single_task(rows: Sequence[Dict[str, Any]]) -> str:
    tasks = sorted({str(row.get("task", "")) for row in rows if row.get("task")})
    if len(tasks) == 1:
        return tasks[0]
    if not tasks:
        return "unknown_task"
    return "all_tasks"


def _rows_by_task_and_policy(
    rows: Sequence[Dict[str, Any]],
) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("task") or "unknown_task"), str(row.get("policy") or "unknown_policy"))
        grouped.setdefault(key, []).append(row)
    return grouped


def _output_filename(template: str, policy: str) -> str:
    filename = template.format(policy=policy, policy_name=policy)
    return re.sub(r"[/:\\]+", "_", filename)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan rawData for monitor JSON files and build one per-property "
            "metrics CSV per task/policy under processedData/<task_name>/."
        )
    )
    parser.add_argument(
        "--input",
        "--input-root",
        dest="input_root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help=(
            "JSON file or folder to scan. Defaults to the whole rawData tree. "
            "You can also pass rawData/<policy_name>/<task_name>. "
            f"Default: {DEFAULT_INPUT_ROOT}"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Where the single CSV should be written. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--id-csv",
        type=Path,
        default=DEFAULT_ID_CSV,
        help=f"CSV mapping LTLf formulas to ID and category. Default: {DEFAULT_ID_CSV}",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help=(
            "Output CSV filename template inside processedData/<task_name>/. "
            "Available placeholders: {policy}, {policy_name}. "
            f"Default: {DEFAULT_OUTPUT_NAME}"
        ),
    )
    parser.add_argument(
        "--policy",
        default=None,
        help=(
            "Only process JSON files for this policy name, for example GR00T-tpt. "
            "Default: process every policy found under the input root."
        ),
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
        "--keep-existing-csvs",
        action="store_true",
        help="Do not remove older CSV exports from the output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    id_csv_path = args.id_csv.expanduser().resolve()
    property_lookup = load_property_lookup(id_csv_path)

    json_paths = _json_files(input_root, args.pattern)
    if not json_paths:
        raise SystemExit(f"No JSON files found under {input_root} matching {args.pattern!r}.")
    if args.max_json_per_task_policy is not None and args.max_json_per_task_policy <= 0:
        raise SystemExit("--max-json-per-task-policy must be a positive integer.")
    if args.progress_every < 0:
        raise SystemExit("--progress-every must be 0 or a positive integer.")

    total_json_paths = len(json_paths)
    available_policies = sorted(
        {_policy_for_path(json_path, input_root) for json_path in json_paths}
    )
    json_paths = _filter_json_files_by_policy(json_paths, input_root, args.policy)
    policy_filtered_json_paths = len(json_paths)
    if args.policy and not json_paths:
        raise SystemExit(
            f"No JSON files found for policy {args.policy!r} under {input_root}. "
            f"Available policies: {', '.join(available_policies) or '<none>'}"
        )

    json_paths = _limit_json_files_per_task_policy(
        json_paths,
        input_root,
        args.max_json_per_task_policy,
    )

    print(f"JSON files found: {total_json_paths}", flush=True)
    if args.policy:
        print(
            f"JSON files matching policy {args.policy}: {policy_filtered_json_paths}",
            flush=True,
        )
    print(f"JSON files selected for processing: {len(json_paths)}", flush=True)
    if len(json_paths) != policy_filtered_json_paths:
        selected_total_label = (
            "policy-filtered total" if args.policy else "total"
        )
        print(
            "Limited input JSON files: "
            f"selected {len(json_paths)} of {policy_filtered_json_paths} "
            f"{selected_total_label} "
            f"using max {args.max_json_per_task_policy} per policy/task.",
            flush=True,
        )

    rows: List[Dict[str, Any]] = []
    for index, json_path in enumerate(json_paths, start=1):
        rows.extend(build_rows_for_json(json_path, input_root, property_lookup))
        if args.progress_every and (
            index == 1 or index % args.progress_every == 0 or index == len(json_paths)
        ):
            print(
                f"Processed {index}/{len(json_paths)} JSON file(s)...",
                flush=True,
            )

    if not args.keep_existing_csvs:
        clear_old_csvs(
            output_root,
            policy=args.policy,
            output_name_template=args.output_name,
        )

    output_paths = []
    for (task, policy), grouped_rows in sorted(_rows_by_task_and_policy(rows).items()):
        output_path = (
            output_root
            / _safe_dir_name(task)
            / _output_filename(args.output_name, policy)
        )
        write_csv(output_path, grouped_rows, COLUMNS)
        output_paths.append(output_path)

    print(f"JSON files found: {total_json_paths}")
    if args.policy:
        print(f"JSON files matching policy {args.policy}: {policy_filtered_json_paths}")
    print(f"JSON files processed: {len(json_paths)}")
    print(f"Rows: {len(rows)}")
    for output_path in output_paths:
        print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
