"""Records per-camera rollout videos matching the official training
dataset's exact layout: `lerobot/videos/chunk-000/observation.images.
<camera>/episode_NNNNNN.mp4`, one file per camera per episode -- instead of
GR00T's pristine single blended-camera `VideoRecordingWrapper` output.

Composed the same way replay_capture.py is: wraps the raw robocasa/robosuite
env (the same object ReplayCapture finds via `_locate_env_holder`), calling
`env.sim.render(camera_name=..., height=.., width=..)` directly -- the same
call `replay/official_playback/reconstruct_training_data.py` uses to
reconstruct training-data videos from replayed states -- so a
`rollout.mp4`/`chunk-000` video produced here and one reconstructed from
`states.npz` later are directly comparable frame-for-frame. Nothing inside
eval/simulators/robocasa is modified.

Frames are written to each camera's mp4 writer as soon as they're captured
(one open imageio writer per camera at a time), NOT buffered in memory for
the whole task and flushed at the end. An earlier version buffered every
frame of every episode in a Python list until `finalize()`; for long-horizon
composite tasks (up to ~4350 raw env steps/episode * 3 cameras *
256x256x3 uint8 bytes, times up to 50 episodes) that grows to tens of GB
resident, which killed real sweep jobs with SIGKILL/OOM at their
`--mem-per-gpu` limit partway through a task (confirmed via sacct: ExitCode
9:0, MaxRSS pinned at the 45G request) -- not a transient scheduling fluke,
since the same buffering design would eventually OOM on any sufficiently
long task/episode-count combination. Streaming to disk keeps memory bounded
to a handful of frames regardless of horizon or episode count.
"""
import json
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

DEFAULT_CAMERA_NAMES = (
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
)


class MultiCameraVideoCapture:
    """Wraps a raw robocasa/robosuite env (the same level ReplayCapture
    wraps with DataCollectionWrapper) to render `camera_names` on every
    step() and stream them straight to disk, one mp4 writer per camera per
    episode, opened lazily on that episode's first frame and closed on the
    next reset(). Delegates everything else to the wrapped env via
    __getattr__, matching DataCollectionWrapper's own proxying convention so
    it composes transparently in the same wrapper chain.

    `lerobot_dir` must be known upfront (it's `ReplayCapture.output_dir /
    "lerobot"`, the same path `finalize_replay_dir` writes `extras/` under --
    see replay_capture.py), so video files land in the right place from the
    very first frame instead of needing a post-hoc rename/move pass."""

    def __init__(
        self,
        env,
        lerobot_dir,
        camera_names: Sequence[str] = DEFAULT_CAMERA_NAMES,
        height: int = 256,
        width: int = 256,
        fps: int = 20,
    ):
        self.env = env
        self.lerobot_dir = Path(lerobot_dir)
        self.camera_names = tuple(camera_names)
        self.height = height
        self.width = width
        self.fps = fps
        # Episodes are recorded strictly in rollout order (one per
        # env.reset(), same reasoning as finalize_replay_dir's own
        # chronological numbering), so a simple incrementing counter IS the
        # final episode_NNNNNN index directly -- no reverse lookup needed.
        self._episode_idx = 0
        self._writers = {}  # cam -> open imageio writer, only while an episode is in progress
        self._episode_has_frames = False

    def __getattr__(self, name):
        return getattr(self.env, name)

    def _camera_dirs(self) -> dict:
        dirs = {}
        for cam in self.camera_names:
            d = self.lerobot_dir / "videos" / "chunk-000" / f"observation.images.{cam}"
            d.mkdir(parents=True, exist_ok=True)
            dirs[cam] = d
        return dirs

    def _open_writers(self) -> None:
        import imageio

        dirs = self._camera_dirs()
        for cam in self.camera_names:
            path = dirs[cam] / f"episode_{self._episode_idx:06d}.mp4"
            self._writers[cam] = imageio.get_writer(str(path), fps=self.fps)

    def _close_writers(self) -> None:
        for writer in self._writers.values():
            writer.close()
        self._writers = {}

    def _capture_frame(self) -> None:
        if not self._writers:
            self._open_writers()
        for cam in self.camera_names:
            frame = self.env.sim.render(height=self.height, width=self.width, camera_name=cam)[
                ::-1
            ]
            self._writers[cam].append_data(np.asarray(frame, dtype=np.uint8))
        self._episode_has_frames = True

    def reset(self, *args, **kwargs):
        # Deliberately does NOT capture a frame here. GR00T's own
        # run_simulation() (gr00t/eval/simulation.py) calls one extra
        # `self.env.reset()` for cleanup after the episode loop ends, right
        # before `close()`, with no step() following it -- if reset()
        # captured a frame, that spurious call would show up as an empty
        # extra "episode" (confirmed by a real run: "recorded 4 episodes
        # but expected 3" for a 3-episode rollout). Only closing writers
        # that actually received a frame (i.e. a real episode happened)
        # means that trailing cleanup reset is correctly skipped, at the
        # cost of not capturing the very first (pre-action) frame of each
        # real episode.
        if self._episode_has_frames:
            self._close_writers()
            self._episode_idx += 1
            self._episode_has_frames = False
        return self.env.reset(*args, **kwargs)

    def step(self, *args, **kwargs):
        result = self.env.step(*args, **kwargs)
        self._capture_frame()
        return result

    def finalize(self, extras_dir: Optional[Path] = None, num_episodes: Optional[int] = None) -> int:
        """Closes any still-open writers (the last episode's, since no
        trailing reset() follows it) and returns the number of episodes
        whose videos were written. `extras_dir`/`num_episodes` are accepted
        (and used only for a sanity-check print) to keep the same call
        signature callers already use."""
        if self._episode_has_frames:
            self._close_writers()
            self._episode_idx += 1
            self._episode_has_frames = False

        written = self._episode_idx
        if num_episodes is not None and written != num_episodes:
            print(
                f"MultiCameraVideoCapture: wrote {written} episodes' videos "
                f"but expected {num_episodes}."
            )
        return written
