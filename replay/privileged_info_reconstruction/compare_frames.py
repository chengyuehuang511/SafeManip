#!/usr/bin/env python3
"""
Frame-by-frame comparison between an original rollout task.mp4 and a
reconstructed video produced by reconstruct_video.py.

The two videos are NOT the same frame rate: the original records every
executed action (or close to it — see KNOWN_BUGS.md's "same 1:8 ratio"
note), while the reconstruction has exactly one frame per recorded
monitor/privileged-info frame (~1:8 sparser). So this script does NOT diff
frame i against frame i; it maps each reconstructed frame index to its
corresponding *original* video frame using the same monitor->video ratio
the viewer uses (ratio = original_frame_count / monitor_frame_count),
then compares those pairs.

Metrics per matched pair: mean absolute pixel difference (0-255 scale) and
SSIM (structural similarity; degrades gracefully to "not computed" if
scikit-image isn't installed). Also explicitly flags original-video frames
that are corrupted (near-uniform-random noise — same signature as the
known video-corruption bug) so a bad match there isn't misread as "our
reconstruction is wrong."

Uses imageio for all video I/O, not a shelled-out ffmpeg/ffprobe CLI: (a)
ffprobe isn't reliably on PATH inside the robocasa conda env (confirmed --
same issue reconstruct_video.py's probe_duration() already works around),
and (b) the original per-frame implementation shelled out to `ffmpeg -vf
select=eq(n\\,N)` once per frame, which decodes from frame 0 every single
call -- O(frames^2) total decode work. This version reads each video in one
sequential pass instead.

Usage:
    python3 compare_frames.py \
        --original /path/to/task.mp4 \
        --reconstructed /path/to/reconstructed.mp4 \
        --privileged_info /path/to/privileged_information_0.json \
        --output /path/to/comparison.json
"""
import argparse
import json
from pathlib import Path

import numpy as np


def probe_duration_fps(path):
    """(fps, duration_s) via imageio -- see module docstring for why not ffprobe."""
    import imageio

    reader = imageio.get_reader(str(path))
    try:
        meta = reader.get_meta_data()
        fps = meta.get("fps") or 10.0
        duration = meta.get("duration")
        if not duration:
            n_frames = meta.get("nframes")
            if n_frames and n_frames != float("inf"):
                duration = float(n_frames) / float(fps)
        return float(fps), (float(duration) if duration else None)
    finally:
        reader.close()


def read_frames_at_indices(path, indices):
    """Read specific 0-indexed frames from a video in a single sequential
    pass (not one ffmpeg subprocess per frame -- see module docstring).
    Returns {index: np.ndarray or None}."""
    import imageio

    wanted = set(indices)
    if not wanted:
        return {}
    max_idx = max(wanted)
    result = {}
    reader = imageio.get_reader(str(path))
    try:
        for i, frame in enumerate(reader):
            if i in wanted:
                result[i] = np.asarray(frame)[:, :, :3]  # drop alpha if present
            if i >= max_idx:
                break
    finally:
        reader.close()
    for idx in wanted:
        result.setdefault(idx, None)  # video shorter than expected
    return result


def is_noise_frame(arr):
    """Heuristic for the known video-corruption bug: a real kitchen-render
    frame has large smooth regions (walls/counters); pure noise doesn't.
    Flag frames where the mean absolute difference between adjacent pixels
    (a crude "how noisy is this" proxy) is implausibly high."""
    if arr is None:
        return True
    gray = arr.mean(axis=2)
    grad = np.abs(np.diff(gray, axis=0)).mean() + np.abs(np.diff(gray, axis=1)).mean()
    return bool(grad > 25.0)  # real frames in this dataset measured well under 10


def compare(original_path, reconstructed_path, privileged_info_path):
    d = json.loads(Path(privileged_info_path).read_text())
    num_monitor_frames = d.get("num_frames") or len(d["privileged_dynamic_info"])

    orig_fps, orig_duration = probe_duration_fps(original_path)
    orig_frame_count = round(orig_duration * orig_fps) if orig_duration else None
    ratio = (orig_frame_count / num_monitor_frames) if orig_frame_count else 8.0

    try:
        from skimage.metrics import structural_similarity as ssim
        have_ssim = True
    except ImportError:
        have_ssim = False

    recon_indices = list(range(num_monitor_frames))
    orig_indices = []
    for recon_idx in recon_indices:
        orig_idx = round(recon_idx * ratio)
        if orig_frame_count and orig_idx >= orig_frame_count:
            orig_idx = orig_frame_count - 1
        orig_indices.append(orig_idx)

    orig_frames = read_frames_at_indices(original_path, orig_indices)
    recon_frames = read_frames_at_indices(reconstructed_path, recon_indices)

    results = []
    for recon_idx, orig_idx in zip(recon_indices, orig_indices):
        orig_frame = orig_frames.get(orig_idx)
        recon_frame = recon_frames.get(recon_idx)

        entry = {
            "monitor_frame": recon_idx,
            "original_video_frame": orig_idx,
            "original_frame_corrupted": is_noise_frame(orig_frame),
        }
        if orig_frame is None or recon_frame is None:
            entry["mean_abs_diff"] = None
            entry["ssim"] = None
        elif orig_frame.shape != recon_frame.shape:
            entry["mean_abs_diff"] = None
            entry["ssim"] = None
            entry["shape_mismatch"] = [list(orig_frame.shape), list(recon_frame.shape)]
        else:
            diff = np.abs(orig_frame.astype(int) - recon_frame.astype(int))
            entry["mean_abs_diff"] = round(float(diff.mean()), 3)
            if have_ssim:
                entry["ssim"] = round(
                    float(ssim(orig_frame, recon_frame, channel_axis=2)), 4
                )
            else:
                entry["ssim"] = None
        results.append(entry)

    valid = [r for r in results if r["mean_abs_diff"] is not None and not r["original_frame_corrupted"]]
    summary = {
        "ratio": ratio,
        "num_monitor_frames": num_monitor_frames,
        "num_compared": len(results),
        "num_skipped_corrupted_original": sum(r["original_frame_corrupted"] for r in results),
        "num_missing_or_mismatched": sum(1 for r in results if r["mean_abs_diff"] is None),
        "mean_abs_diff_avg": round(float(np.mean([r["mean_abs_diff"] for r in valid])), 3) if valid else None,
        "mean_abs_diff_max": round(float(np.max([r["mean_abs_diff"] for r in valid])), 3) if valid else None,
        "ssim_avg": (
            round(float(np.mean([r["ssim"] for r in valid if r["ssim"] is not None])), 4)
            if valid and have_ssim else None
        ),
        "have_ssim": have_ssim,
    }
    return {"summary": summary, "frames": results}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--original", required=True, type=Path)
    ap.add_argument("--reconstructed", required=True, type=Path)
    ap.add_argument("--privileged_info", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    report = compare(args.original, args.reconstructed, args.privileged_info)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
