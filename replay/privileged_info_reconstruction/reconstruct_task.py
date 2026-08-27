#!/usr/bin/env python3
"""
Reconstruct + compare every episode of one task's latest rollout, reusing
the built env across episodes (only `gym.make()`s once; ~4x faster than
rebuilding from scratch per episode -- see make_env()'s docstring in
reconstruct_video.py). Designed to be one SLURM job per task when
processing many tasks (see submit_all.sh).

Writes, per episode, the same artifacts as reconstruct_video.py +
compare_frames.py:
    output/<Task>/episode_<N>_reconstructed.mp4
    output/<Task>/episode_<N>_reconstruct_meta.json
    output/<Task>/episode_<N>_comparison.json
plus one aggregate:
    output/<Task>/task_summary.json

Usage:
    python3 reconstruct_task.py --task ArrangeBreadBasket \
        [--results_root .../evals/target] [--output_root output] \
        [--max_episodes N] [--skip_existing]
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import reconstruct_video as rv  # noqa: E402 (after sys.path patch, and rv does its own cwd guard)
import compare_frames  # noqa: E402

DEFAULT_RESULTS_ROOT = (
    "/nethome/chuang475/testnvme/projects/SafeManip/results/evals/"
    "all_tasks_3_ckpt_50_rollouts/target_posttraining/evals/target"
)
DEFAULT_OUTPUT_ROOT = str(Path(__file__).parent / "output")


def latest_rollout_dir(results_root, task):
    task_dir = results_root / task / "rollout_data"
    candidates = sorted(
        p for p in task_dir.iterdir() if p.is_dir() and p.name.startswith(f"{task}--")
    )
    if not candidates:
        raise FileNotFoundError(f"no rollout_data/{task}--* dir under {task_dir}")
    return candidates[-1]


def find_episodes(rollout_dir):
    eps = []
    for p in rollout_dir.glob("privileged_information_*.json"):
        if p.name.endswith("_monitor.json"):
            continue
        stem = p.stem  # privileged_information_<N>
        idx = int(stem.rsplit("_", 1)[1])
        eps.append(idx)
    return sorted(eps)


def find_video(rollout_dir, episode):
    matches = list(rollout_dir.glob(f"*--episode={episode}--success=*--task=task.mp4"))
    return matches[0] if matches else None


def process_task(task, results_root, output_root, max_episodes=None, skip_existing=False):
    rollout_dir = latest_rollout_dir(results_root, task)
    episodes = find_episodes(rollout_dir)
    if max_episodes:
        episodes = episodes[:max_episodes]
    print(f"[{task}] rollout={rollout_dir.name} n_episodes={len(episodes)}", flush=True)

    out_dir = output_root / task
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    reuse_env = None
    for ep in episodes:
        t0 = time.time()
        info_path = rollout_dir / f"privileged_information_{ep}.json"
        video_path = find_video(rollout_dir, ep)
        recon_path = out_dir / f"episode_{ep}_reconstructed.mp4"
        meta_path = out_dir / f"episode_{ep}_reconstruct_meta.json"
        comp_path = out_dir / f"episode_{ep}_comparison.json"

        if skip_existing and recon_path.is_file() and comp_path.is_file():
            print(f"[{task}] episode {ep}: skip (already exists)", flush=True)
            summary.append({"episode": ep, "status": "skipped"})
            continue

        entry = {"episode": ep}
        try:
            report = rv.reconstruct(
                info_path, recon_path, rv.CAMERA_NAME_DEFAULT, 10.0, 256, 256,
                original_video=video_path, reuse_env=reuse_env, close_env=False,
            )
            reuse_env = report.pop("env")
            report.pop("missing_joints_detail", None)
            entry.update(report)
            entry["status"] = "reconstructed"
            meta_path.write_text(json.dumps(report, indent=2))
        except Exception as e:
            entry["status"] = "reconstruct_error"
            entry["error"] = f"{type(e).__name__}: {e}"
            entry["traceback"] = traceback.format_exc()
            print(f"[{task}] episode {ep}: RECONSTRUCT FAILED: {e}", flush=True)
            # the reused env may now be in a bad state; force a fresh one next episode
            try:
                if reuse_env is not None:
                    reuse_env[0].close()
            except Exception:
                pass
            reuse_env = None
            summary.append(entry)
            continue

        if video_path is None:
            entry["comparison_status"] = "no_original_video"
            print(f"[{task}] episode {ep}: no original video found, skipping comparison", flush=True)
        else:
            try:
                comp = compare_frames.compare(video_path, recon_path, info_path)
                comp_path.write_text(json.dumps(comp, indent=2))
                entry["comparison_status"] = "ok"
                entry["comparison_summary"] = comp["summary"]
            except Exception as e:
                entry["comparison_status"] = "error"
                entry["comparison_error"] = f"{type(e).__name__}: {e}"
                print(f"[{task}] episode {ep}: COMPARE FAILED: {e}", flush=True)

        entry["elapsed_s"] = round(time.time() - t0, 2)
        print(f"[{task}] episode {ep}: done in {entry['elapsed_s']}s", flush=True)
        summary.append(entry)

    if reuse_env is not None:
        try:
            reuse_env[0].close()
        except Exception:
            pass

    (out_dir / "task_summary.json").write_text(json.dumps(summary, indent=2))
    n_ok = sum(1 for s in summary if s["status"] == "reconstructed")
    print(f"[{task}] finished: {n_ok}/{len(episodes)} reconstructed", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True)
    ap.add_argument("--results_root", type=Path, default=Path(DEFAULT_RESULTS_ROOT))
    ap.add_argument("--output_root", type=Path, default=Path(DEFAULT_OUTPUT_ROOT))
    ap.add_argument("--max_episodes", type=int, default=None)
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()

    process_task(args.task, args.results_root, args.output_root, args.max_episodes, args.skip_existing)


if __name__ == "__main__":
    main()
