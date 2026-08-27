# replay/

Two independent ways to get a replayable/renderable video of a recorded
episode, split into subfolders:

- **[`official_playback/`](official_playback/README.md)** — replay via the
  official RoboCasa lerobot dataset format (`model.xml.gz` + `states.npz` +
  `ep_meta.json`, the actual training-data format). **Prefer this** whenever
  data in that format is available (or can be produced from a rollout, see
  that README's "Producing this format" section) — it's exact, using
  RoboCasa's own upstream `playback_dataset.py`, not a reconstruction.

- **[`privileged_info_reconstruction/`](privileged_info_reconstruction/README.md)**
  — a from-scratch reconstruction built here, for cases where only an
  eval-time `privileged_information_<N>.json` dump is available (no
  official-format data, e.g. SafeManip eval rollouts as they're currently
  recorded). Necessarily approximate — see that README's calibration/override
  hacks and randomness audit for exactly where and why.
