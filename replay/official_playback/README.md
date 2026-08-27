# Exact replay via the official RoboCasa lerobot format

**If you have (or can produce) data in the official RoboCasa lerobot dataset
format — like `~/flash/datasets/robocasa/v1.0/target/composite/<Task>/<date>/lerobot`
— use this instead of the [privileged-info reconstruction](../privileged_info_reconstruction/README.md)
approach.** It's exact, not an approximation: no `MobileBaseCalibrator`-style
calibration, no fixture-position override hacks, no ~1:8 frame sparsity. That
whole reconstruction approach exists only because `privileged_information_<N>.json`
is a lossy, sparse-pose eval-time dump; the official dataset format instead
stores the *full* MuJoCo sim state every timestep, which is literally the
ground truth the original video was rendered from.

## Why this is exact

Each episode in the lerobot dataset has, under `extras/episode_XXXXXX/`:
- `model.xml.gz` — the exact procedurally-generated scene XML for that
  episode (fixtures, layout, textures, everything)
- `states.npz` (`states` array, shape `(T, 256)`) — the full **flattened
  MuJoCo sim state** (`sim.get_state().flatten()`) at every timestep
- `ep_meta.json` — layout/style ids, object configs, robot init pose,
  language instruction

Plus, aligned frame-for-frame, `data/chunk-000/episode_XXXXXX.parquet`'s
`action` column (12-dim).

Because `states.npz` has the complete state (not just object/robot poses),
replay is just `env.sim.set_state_from_flattened(states[t]); env.sim.forward()`
per frame — no solving, no drift possible.

## Replaying an existing lerobot dataset

RoboCasa ships an official playback script for exactly this format:
`robocasa/scripts/dataset_scripts/playback_dataset.py` (in the `robocasa`
conda env's editable install, source at
`/coc/testnvme/chuang475/projects/robocasa/robocasa/scripts/dataset_scripts/playback_dataset.py`).
It supports two modes:
- **Default (state loading)** — `env.sim.set_state_from_flattened(states[t])`
  per frame. Exact, guaranteed match to the original video.
- **`--use-actions`** — open-loop replay of the recorded `action` column
  through `env.step()`, with a per-step divergence check (L2 norm) against
  the recorded state, printed as a warning if it doesn't match exactly.
  Useful to test action-level determinism, not needed for exact video
  reconstruction.

Use `run_official_playback.sh` (sbatch wrapper matching
`../privileged_info_reconstruction/run_reconstruct.sh`'s pattern, including
the same `MUJOCO_GL=egl` / GPU-node requirement and cwd guard):

```bash
DATASET=~/flash/datasets/robocasa/v1.0/target/composite/ArrangeBreadBasket/20250809/lerobot \
  sbatch run_official_playback.sh
# or interactively on a GPU node:
DATASET=... bash run_official_playback.sh
# add USE_ACTIONS=1 to replay the recorded action column instead of loading states directly
```

## Producing this format from your own rollouts (e.g. SafeManip eval trajectories)

The recording side of this pipeline doesn't care where actions come from —
teleop, scripted MimicGen policies, or a trained policy's rollout actions all
work the same way. Full chain, each step's script named:

1. **Record raw states+model+actions live during rollout** —
   `robosuite.wrappers.DataCollectionWrapper`. Wrap the env before stepping
   it with your policy's actions:
   ```python
   from robosuite.wrappers import DataCollectionWrapper
   env = DataCollectionWrapper(env, directory="/path/to/raw_demos")
   ```
   On every `env.step(action)` it captures `env.sim.get_state().flatten()`
   (the same 256-dim state format as `states.npz`) and the action taken; on
   episode start it dumps `model.xml` + `ep_meta.json`. Flushes to
   `raw_demos/ep_<timestamp>/{model.xml, ep_meta.json, state_<timestamp>.npz}`.
   This replaces `privileged_information_<N>.json` dumping entirely — it's
   ground truth captured live, no reconstruction step needed afterward.

2. **Consolidate raw dumps → one `demo.hdf5`** —
   `robosuite/scripts/collect_human_demonstrations.py:gather_demonstrations_as_hdf5(directory, out_dir, env_info)`.
   Walks the `ep_*` dirs, concatenates each episode's states/actions into
   per-demo hdf5 groups (`model_file`, `states`, `actions`). Note: only
   keeps episodes it considers successful by default — check that branch if
   failed rollouts should be kept too.

3. **(Optional) attach observations** —
   `robocasa/scripts/dataset_scripts/dataset_states_to_obs.py --dataset demo.hdf5
   --camera_names robot0_agentview_left robot0_agentview_right robot0_eye_in_hand`
   — replays each episode's states through the sim to render image
   observations, rewards, dones into an augmented hdf5.

4. **Convert to the lerobot format** —
   `robocasa/scripts/dataset_scripts/convert_hdf5_lerobot.py
   --raw_dataset_path demo.hdf5 --camera_names ... --camera_height 256 --camera_width 256`
   — writes the same `data/`, `meta/`, `videos/`,
   `extras/episode_XXXXXX/{model.xml.gz,states.npz,ep_meta.json}` structure as
   the official training dataset.

5. **Replay** — feed the resulting dataset dir into `playback_dataset.py` /
   `run_official_playback.sh` as above.

## Batch reconstruction for the viewer's "Training Data" tab

`reconstruct_training_data.py` batch-reconstructs N episodes of one task
(default 10), ground-truth state playback as above, writing one mp4 per
episode with all 3 camera views concatenated side-by-side (matching what
`../../viewer`'s "Training Data" tab expects):

```bash
python3 reconstruct_training_data.py --task ArrangeBreadBasket \
    [--dataset_root ~/flash/datasets/robocasa/v1.0/target] \
    [--output_root output] [--n_episodes 10] [--video_skip 2] \
    [--camera_names robot0_agentview_left robot0_agentview_right robot0_eye_in_hand] \
    [--camera_height 256] [--camera_width 256] [--skip_existing]
```
Writes `output/<Task>/episode_<N>_reconstructed.mp4` +
`episode_<N>_reconstruct_meta.json` (fps, frame count, camera list, and the
episode's recorded language instruction) + `task_summary.json`.

**`video_skip` note**: source states are recorded @20fps; the video writer's
fps is derived as `20/video_skip` so real-time duration stays correct — a
fixed output fps regardless of skip would play the video back too fast
(hit this exact bug once, fixed).

To run this for every task (SLURM array job, one array index per task, all
tasks run in parallel rather than one after another):

```bash
N_EPISODES=10 bash submit_training_data.sh                 # all tasks found under DATASET_ROOT
N_EPISODES=10 bash submit_training_data.sh TaskA TaskB      # just these tasks
```

`submit_training_data.sh` writes its per-array-index task list to
`logs/training_data_tasks.<random>` on the shared filesystem, **not** `/tmp`
(node-local `/tmp` on the login node is invisible to whichever compute node
an array index lands on — hit this exact bug once too, fixed).

The viewer (`../../viewer/`)'s "Training Data" tab reads directly from
`output/<Task>/episode_<N>_reconstructed.mp4` + the sibling meta json,
progressively — task/episode dropdowns show whatever's been reconstructed so
far, no need to wait for the whole batch to finish.
