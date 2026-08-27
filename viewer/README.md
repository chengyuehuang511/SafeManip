# SafeManip monitor viewer

A small local web app for manually cross-checking the automated safety
monitor's output against the actual rollout videos.

No dependencies beyond Python 3 stdlib + `ffprobe` (from ffmpeg) on `PATH`.

## Run

```bash
cd viewer
python3 server.py --root /path/to/.../evals/target --port 8008
```

Then open http://127.0.0.1:8008/ (if running on a remote machine, either
SSH-tunnel the port or pass `--host 0.0.0.0`).

`--root` defaults to the `target` eval dir the viewer was built for:
`SafeManip/results/evals/all_tasks_3_ckpt_50_rollouts/target_posttraining/evals/target`.

If a task has more than one `<Task>--<timestamp>` rollout folder, the
lexicographically-latest one (== newest, timestamps sort correctly as
strings) is used automatically — matching the "use the latest one" rule.

## What it shows, per episode

- The rollout video (`task.mp4`), served with HTTP Range support so seeking
  is instant.
- Every **violation** the monitor flagged, with its property name,
  human-readable description, the monitor's own explanation text, and
  "jump to video" buttons computed from the monitor-frame → video-frame
  ratio (see `KNOWN_BUGS.md`'s "same 1:8 ratio" note — the ratio is derived
  per-episode from the actual video duration/fps rather than assumed, so it
  self-corrects if a given episode's ratio drifts from 8).
- Every **satisfied** property, so you can check the monitor isn't just
  right about the violations but also right that everything *else* held.
- A free-text box per violation/satisfied item plus a verdict
  (confirmed / disputed / unsure), saved to `viewer/annotations/`.
- An episode-level "what did the monitor miss?" box — the point of this is
  to catch **false negatives**: watch the whole video first, not just the
  flagged timestamps, then note anything unsafe you see that isn't in the
  violations list, or anything in the list that doesn't hold up once you
  actually look.

## Recommended review workflow

The jump buttons are a shortcut for double-checking a specific claim, not a
substitute for watching the episode. For a real audit:

1. Watch the whole video top to bottom first.
2. Only then open the violations/satisfied lists and use the jump buttons to
   pinpoint exactly what the monitor is claiming at each timestamp.
3. Mark each item confirmed/disputed/unsure, and use the "missed" box for
   anything you saw that the monitor didn't flag (or flagged as satisfied
   when it shouldn't have).

## Known caveat: corrupted video tails

Some rollout `task.mp4` files decode to pure noise after a point partway
through (observed around t≈77s for one `ArrangeBreadBasket` episode, and a
different point for another episode in the same task) even though `ffmpeg`
reports no decode errors. This is unrelated to this viewer/tooling — the
same corruption shows up decoding frame-by-frame from the start. Any
violation whose jump-to time lands in a corrupted region can't be verified
visually; the viewer doesn't currently auto-detect this, so if the video
looks like static, that's the file, not your browser.

## Annotations storage

`viewer/annotations/<task>__<episode>.json`:

```json
{
  "violations": {"0": {"verdict": "confirmed", "note": "..."}},
  "satisfied": {"0": {"verdict": "disputed", "note": "..."}},
  "missed_notes": "free text",
  "overall_verdict": "matches|mismatches|partial"
}
```
