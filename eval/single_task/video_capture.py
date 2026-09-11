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
    wraps with DataCollectionWrapper) to render and buffer `camera_names`
    separately on every step/reset, writing one mp4 per camera per episode
    on `finalize()`. Delegates everything else to the wrapped env via
    __getattr__, matching DataCollectionWrapper's own proxying convention so
    it composes transparently in the same wrapper chain."""

    def __init__(
        self,
        env,
        camera_names: Sequence[str] = DEFAULT_CAMERA_NAMES,
        height: int = 256,
        width: int = 256,
        fps: int = 20,
    ):
        self.env = env
        self.camera_names = tuple(camera_names)
        self.height = height
        self.width = width
        self.fps = fps
        self._episode_frames: List[dict] = []  # one dict per completed episode
        self._current_frames = {cam: [] for cam in self.camera_names}

    def __getattr__(self, name):
        return getattr(self.env, name)

    def _capture_frame(self) -> None:
        for cam in self.camera_names:
            frame = self.env.sim.render(height=self.height, width=self.width, camera_name=cam)[
                ::-1
            ]
            self._current_frames[cam].append(np.asarray(frame, dtype=np.uint8))

    def reset(self, *args, **kwargs):
        # Deliberately does NOT capture a frame here. GR00T's own
        # run_simulation() (gr00t/eval/simulation.py) calls one extra
        # `self.env.reset()` for cleanup after the episode loop ends, right
        # before `close()`, with no step() following it -- if reset()
        # captured a frame, that spurious call would show up as an empty
        # extra "episode" (confirmed by a real run: "recorded 4 episodes
        # but expected 3" for a 3-episode rollout). Capturing only on
        # step() means that trailing cleanup reset produces zero frames and
        # is correctly skipped below, at the cost of not capturing the very
        # first (pre-action) frame of each real episode.
        if any(len(v) > 0 for v in self._current_frames.values()):
            self._episode_frames.append(self._current_frames)
        self._current_frames = {cam: [] for cam in self.camera_names}
        return self.env.reset(*args, **kwargs)

    def step(self, *args, **kwargs):
        result = self.env.step(*args, **kwargs)
        self._capture_frame()
        return result

    def finalize(self, extras_dir: Path, num_episodes: int) -> int:
        """Writes videos/chunk-000/observation.images.<camera>/episode_NNNNNN.mp4
        under `extras_dir`'s parent (i.e. the `lerobot/` dir), numbered by
        the SAME chronological order finalize_replay_dir uses for
        `extras/episode_NNNNNN` (this class records episodes strictly in
        rollout order, one per env.reset(), so episode index i here IS
        episode_{i:06d} directly -- no reverse lookup needed, same reasoning
        as replay_capture.py's _attach_episode_videos).

        Returns the number of episodes' videos successfully written (0, with
        a warning, if the recorded episode count doesn't match
        `num_episodes` -- refuses to guess a mapping)."""
        if any(len(v) > 0 for v in self._current_frames.values()):
            self._episode_frames.append(self._current_frames)
            self._current_frames = {cam: [] for cam in self.camera_names}

        if len(self._episode_frames) != num_episodes:
            print(
                f"MultiCameraVideoCapture: recorded {len(self._episode_frames)} episodes "
                f"but expected {num_episodes}; not writing videos."
            )
            return 0

        import imageio

        lerobot_dir = Path(extras_dir).parent
        for cam in self.camera_names:
            (lerobot_dir / "videos" / "chunk-000" / f"observation.images.{cam}").mkdir(
                parents=True, exist_ok=True
            )

        written = 0
        for idx, frames_by_camera in enumerate(self._episode_frames):
            for cam in self.camera_names:
                frames = frames_by_camera[cam]
                if not frames:
                    continue
                out_path = (
                    lerobot_dir
                    / "videos"
                    / "chunk-000"
                    / f"observation.images.{cam}"
                    / f"episode_{idx:06d}.mp4"
                )
                writer = imageio.get_writer(str(out_path), fps=self.fps)
                try:
                    for frame in frames:
                        writer.append_data(frame)
                finally:
                    writer.close()
            written += 1
        return written
