#!/usr/bin/env python3
"""
Scan every task's task_summary.json under output/ and produce one
aggregate report: completion rate, error breakdown, data_integrity_suspect
rate, comparison-quality outliers, sentinel-frame stats, etc.

Usage:
    python3 aggregate_report.py [--output_root output] [--json_out report.json]
"""
import argparse
import json
from pathlib import Path
from collections import Counter

DEFAULT_OUTPUT_ROOT = str(Path(__file__).parent / "output")


def load_task_summaries(output_root):
    summaries = {}
    for task_dir in sorted(output_root.iterdir()):
        if not task_dir.is_dir():
            continue
        summary_path = task_dir / "task_summary.json"
        if not summary_path.is_file():
            continue
        try:
            summaries[task_dir.name] = json.loads(summary_path.read_text())
        except Exception as e:
            print(f"WARNING: failed to parse {summary_path}: {e}")
    return summaries


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output_root", type=Path, default=Path(DEFAULT_OUTPUT_ROOT))
    ap.add_argument("--json_out", type=Path, default=None)
    args = ap.parse_args()

    summaries = load_task_summaries(args.output_root)
    print(f"tasks with a task_summary.json: {len(summaries)}")

    all_entries = []
    for task, entries in summaries.items():
        for e in entries:
            e = dict(e)
            e["task"] = task
            all_entries.append(e)

    n_total = len(all_entries)
    status_counts = Counter(e["status"] for e in all_entries)
    print(f"\ntotal episode entries: {n_total}")
    for status, count in status_counts.most_common():
        print(f"  {status}: {count}")

    reconstructed = [e for e in all_entries if e["status"] == "reconstructed"]

    # data integrity
    suspect = [e for e in reconstructed if e.get("data_integrity_suspect")]
    print(f"\ndata_integrity_suspect: {len(suspect)}/{len(reconstructed)} reconstructed episodes")
    suspect_by_task = Counter(e["task"] for e in suspect)
    for task, count in suspect_by_task.most_common():
        total_task = sum(1 for e in reconstructed if e["task"] == task)
        print(f"  {task}: {count}/{total_task}")

    # sentinel frames
    with_sentinel = [e for e in reconstructed if e.get("n_sentinel_frames", 0) > 0]
    print(f"\nepisodes with sentinel/padding frames: {len(with_sentinel)}/{len(reconstructed)}")
    if with_sentinel:
        avg_sentinel_frac = sum(
            e["n_sentinel_frames"] / max(1, e["n_frames"]) for e in with_sentinel
        ) / len(with_sentinel)
        print(f"  avg fraction of frames that are sentinel (among affected episodes): {avg_sentinel_frac:.2%}")

    # root_body mismatches (non-sentinel)
    with_root_mismatch = [e for e in reconstructed if e.get("root_body_mismatches", 0) > 0]
    print(f"\nepisodes with non-sentinel root_body mismatches: {len(with_root_mismatch)}/{len(reconstructed)}")

    # comparison quality (mean_abs_diff / ssim), for episodes with a comparison
    with_comparison = [e for e in reconstructed if e.get("comparison_status") == "ok"]
    print(f"\nepisodes with a successful comparison: {len(with_comparison)}/{len(reconstructed)}")
    if with_comparison:
        mads = [e["comparison_summary"]["mean_abs_diff_avg"] for e in with_comparison
                if e["comparison_summary"].get("mean_abs_diff_avg") is not None]
        ssims = [e["comparison_summary"]["ssim_avg"] for e in with_comparison
                 if e["comparison_summary"].get("ssim_avg") is not None]
        if mads:
            mads_sorted = sorted(mads)
            print(f"  mean_abs_diff_avg: min={mads_sorted[0]:.2f} median={mads_sorted[len(mads_sorted)//2]:.2f} "
                  f"max={mads_sorted[-1]:.2f} mean={sum(mads)/len(mads):.2f}")
        if ssims:
            ssims_sorted = sorted(ssims)
            print(f"  ssim_avg: min={ssims_sorted[0]:.3f} median={ssims_sorted[len(ssims_sorted)//2]:.3f} "
                  f"max={ssims_sorted[-1]:.3f} mean={sum(ssims)/len(ssims):.3f}")

        # worst outliers (highest mean_abs_diff, excluding data_integrity_suspect
        # ones since those are expected to look bad for an unrelated reason)
        clean = [e for e in with_comparison if not e.get("data_integrity_suspect")
                  and e["comparison_summary"].get("mean_abs_diff_avg") is not None]
        worst = sorted(clean, key=lambda e: -e["comparison_summary"]["mean_abs_diff_avg"])[:15]
        print("\n  worst 15 (by mean_abs_diff_avg, excluding data_integrity_suspect episodes):")
        for e in worst:
            cs = e["comparison_summary"]
            print(f"    {e['task']} ep{e['episode']}: mad={cs['mean_abs_diff_avg']:.2f} "
                  f"ssim={cs.get('ssim_avg')} n_sentinel={e.get('n_sentinel_frames', 0)} "
                  f"missing_joints={len(e.get('missing_joints', []))}")

    # reconstruct errors
    errors = [e for e in all_entries if e["status"] == "reconstruct_error"]
    if errors:
        print(f"\nreconstruct errors: {len(errors)}")
        err_types = Counter(e["error"].split(":")[0] for e in errors)
        for et, count in err_types.most_common():
            print(f"  {et}: {count}")
        print("  examples:")
        for e in errors[:10]:
            print(f"    {e['task']} ep{e['episode']}: {e['error']}")

    # missing tasks entirely (submitted but no summary yet, or crashed before writing one)
    all_task_dirs = {p.name for p in args.output_root.iterdir() if p.is_dir()}
    no_summary = all_task_dirs - set(summaries.keys())
    if no_summary:
        print(f"\ntask dirs with NO task_summary.json yet (still running or crashed early): {sorted(no_summary)}")

    if args.json_out:
        report = {
            "n_total": n_total,
            "status_counts": dict(status_counts),
            "data_integrity_suspect_count": len(suspect),
            "data_integrity_suspect_by_task": dict(suspect_by_task),
            "n_reconstructed": len(reconstructed),
            "n_with_comparison": len(with_comparison),
        }
        args.json_out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
