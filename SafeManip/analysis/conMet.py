from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


ANALYSIS_DIR = Path(__file__).resolve().parent
DEFAULT_PROCESSED_ROOT = ANALYSIS_DIR / "processedData"
DEFAULT_OUTPUT = DEFAULT_PROCESSED_ROOT / "all_metrics.csv"


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _metric_csvs(processed_root: Path, pattern: str, output_path: Path) -> List[Path]:
    excluded_dirs = {
        (processed_root / "RQ1").resolve(),
        (processed_root / "RQ2").resolve(),
        (processed_root / "RQ3").resolve(),
    }
    output_path = output_path.resolve()

    paths = []
    for path in sorted(processed_root.rglob(pattern)):
        if not path.is_file():
            continue
        resolved = path.resolve()
        if resolved == output_path:
            continue
        if any(_is_relative_to(resolved, excluded_dir) for excluded_dir in excluded_dirs):
            continue
        paths.append(path)
    return paths


def _read_csv(path: Path) -> tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit(f"Metrics CSV is empty: {path}")
        rows = [dict(row) for row in reader]
        return list(reader.fieldnames), rows


def _merge_columns(headers: Iterable[Sequence[str]]) -> List[str]:
    columns: List[str] = []
    seen = set()
    for header in headers:
        for column in header:
            if column not in seen:
                columns.append(column)
                seen.add(column)
    return columns


def write_csv(path: Path, rows: Sequence[Dict[str, str]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Concatenate all processed metrics CSVs into one big CSV."
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=DEFAULT_PROCESSED_ROOT,
        help=f"Root containing processed metrics CSVs. Default: {DEFAULT_PROCESSED_ROOT}",
    )
    parser.add_argument(
        "--pattern",
        default="metrics*.csv",
        help="CSV glob to concatenate. Default: metrics*.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Combined CSV path. Default: {DEFAULT_OUTPUT}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_root = args.processed_root.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    metric_paths = _metric_csvs(processed_root, args.pattern, output_path)
    if not metric_paths:
        raise SystemExit(
            f"No metrics CSV files found under {processed_root} matching {args.pattern!r}."
        )

    headers = []
    rows: List[Dict[str, str]] = []
    for path in metric_paths:
        header, csv_rows = _read_csv(path)
        headers.append(header)
        rows.extend(csv_rows)

    columns = _merge_columns(headers)
    write_csv(output_path, rows, columns)

    print(f"Read {len(metric_paths)} metrics CSV file(s).")
    print(f"Combined rows: {len(rows)}")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
