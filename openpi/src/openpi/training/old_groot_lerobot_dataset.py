"""Groot-LeRobot dataset implementation for openpi training."""

import glob
import json
import os
from collections.abc import Iterator, Sequence
from typing import Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import cv2  # Add OpenCV for video frame extraction

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.transforms as _transforms
import openpi.shared.normalize as _normalize

import pathlib

T_co = TypeVar("T_co", covariant=True)


class GrootLeRobotDataset:
    """Dataset class for Groot-LeRobot datasets stored in parquet format."""

    def __init__(
        self,
        data_dir: str,
        action_horizon: int = 1,
        shuffle: bool = False,
        seed: int = 42,
        # action_dim: int = 7,  # Add action_dim parameter to match standard LeRobot
        cache_in_memory: bool = False,  # Avoid caching by default for large datasets
    ):
        """Initialize the Groot-LeRobot dataset.

        Args:
            data_dir: Path to the groot-lerobot dataset directory
            action_horizon: Number of consecutive actions to include
            shuffle: Whether to shuffle episodes
            seed: Random seed for shuffling
            action_dim: Action dimension for padding (matches model config)
            cache_in_memory: If True, cache full episodes in memory (not recommended for large datasets)
        """
        # print("action_dim:", action_dim)
        # exit()
        
        self.data_dir = data_dir
        self.action_horizon = action_horizon
        self.shuffle = shuffle
        self.seed = seed
        # self.action_dim = action_dim  # Store action_dim for padding
        self.cache_in_memory = cache_in_memory

        # Load dataset metadata
        self._load_metadata()
        
        # Get all episode files
        self.episode_files = self._get_episode_files()
        
        # Prepare index (episode lengths) and optionally cache data
        self._prepare_index()
        
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(self.episode_files)

    def _load_metadata(self):
        """Load dataset metadata from JSON files."""
        meta_dir = os.path.join(self.data_dir, "meta")
        
        # Load info.json
        info_path = os.path.join(meta_dir, "info.json")
        if os.path.exists(info_path):
            with open(info_path, "r") as f:
                self.info = json.load(f)
        else:
            self.info = {}
            
        # Load metadata.json for statistics
        metadata_path = os.path.join(meta_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

        # Load tasks mapping (task_index -> language). Try several common locations/keys.
        self.tasks: dict[int, str] = {}
        tasks_json = os.path.join(meta_dir, "tasks.json")
        tasks_jsonl = os.path.join(meta_dir, "tasks.jsonl")
        # Prefer JSONL if present (common in Groot exports)
        if os.path.exists(tasks_jsonl):
            try:
                with open(tasks_jsonl, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                            # attempt common fields: {"task_index": int, "task": str} or {"index": int, "task": str}
                            idx = rec.get("task_index", rec.get("index"))
                            task = rec.get("task", rec.get("text", rec.get("lang", None)))
                            if idx is not None and isinstance(task, str):
                                self.tasks[int(idx)] = task
                        except Exception:
                            continue
            except Exception:
                pass
        elif os.path.exists(tasks_json):
            try:
                with open(tasks_json, "r") as f:
                    raw = json.load(f)
                # Normalize keys to int
                if isinstance(raw, dict):
                    self.tasks = {int(k): v for k, v in raw.items()}
            except Exception:
                pass
        # Fallback to metadata dicts if available
        if not self.tasks and isinstance(self.metadata, dict):
            try:
                if "tasks" in self.metadata and isinstance(self.metadata["tasks"], dict):
                    self.tasks = {int(k): v for k, v in self.metadata["tasks"].items()}
                elif "id_to_task" in self.metadata and isinstance(self.metadata["id_to_task"], dict):
                    self.tasks = {int(k): v for k, v in self.metadata["id_to_task"].items()}
            except Exception:
                pass
        # Fallback default prompt from dataset dir name
        base_name = os.path.basename(os.path.normpath(self.data_dir))
        base_name = base_name.replace("_groot", "").replace("_", " ")
        self.default_prompt = base_name.strip() or "perform the task"

    def _get_episode_files(self) -> list[str]:
        """Get all episode parquet files in the dataset."""
        pattern = os.path.join(self.data_dir, "data", "chunk-*", "episode_*.parquet")
        episode_files = glob.glob(pattern)
        episode_files.sort()  # Ensure consistent ordering
        return episode_files

    def _prepare_index(self):
        """Prepare episode lengths; optionally cache full data if requested."""
        self._episode_data = []
        self._episode_lengths = []
        self._has_images = False

        # Detect images
        videos_dir = os.path.join(self.data_dir, "videos")
        if os.path.exists(videos_dir):
            chunk_dir = os.path.join(videos_dir, "chunk-000")
            if os.path.exists(chunk_dir):
                image_dirs = [d for d in os.listdir(chunk_dir) if 'image' in d.lower()]
                self._has_images = len(image_dirs) > 0
        
        # Try to get lengths via PyArrow metadata for speed
        try:
            import pyarrow.parquet as pq  # type: ignore
            for ep in self.episode_files:
                try:
                    pf = pq.ParquetFile(ep)
                    self._episode_lengths.append(pf.metadata.num_rows)
                except Exception:
                    # Fallback to pandas
                    df = pd.read_parquet(ep, columns=['observation.state'])
                    self._episode_lengths.append(len(df))
            # Optionally cache full data
            if self.cache_in_memory:
                for i, episode_file in enumerate(self.episode_files):
                    try:
                        df = pd.read_parquet(episode_file)
                        required_cols = ['observation.state', 'action', 'task_index', 'episode_index']
                        missing_cols = [col for col in required_cols if col not in df.columns]
                        if missing_cols:
                            print(f"WARNING: Episode {i+1} missing columns: {missing_cols}")
                            continue
                        episode_data = {
                            'state': df['observation.state'].values,
                            'actions': df['action'].values,
                            'task_index': df['task_index'].values,
                            'episode_index': df['episode_index'].values,
                        }
                        self._episode_data.append(episode_data)
                    except Exception as e:
                        print(f"ERROR loading episode {i+1}: {e}")
                        raise
        except Exception:
            # No pyarrow: fallback entirely to pandas scan
            for i, episode_file in enumerate(self.episode_files):
                df = pd.read_parquet(episode_file, columns=['observation.state'])
                self._episode_lengths.append(len(df))
                if self.cache_in_memory:
                    df_full = pd.read_parquet(episode_file)
                    episode_data = {
                        'state': df_full['observation.state'].values,
                        'actions': df_full['action'].values,
                        'task_index': df_full['task_index'].values,
                        'episode_index': df_full['episode_index'].values,
                    }
                    self._episode_data.append(episode_data)

    @property
    def has_images(self) -> bool:
        """Check if the dataset contains image data."""
        return self._has_images

    def _extract_video_frame(self, video_path: str, frame_index: int) -> np.ndarray:
        """Extract a specific frame from a video file.
        
        Args:
            video_path: Path to the MP4 video file
            frame_index: Index of the frame to extract (0-based)
            
        Returns:
            Frame as numpy array in (H, W, C) format with uint8 dtype
        """
        if not os.path.exists(video_path):
            print(f"Warning: Video file not found: {video_path}")
            return np.zeros((128, 128, 3), dtype=np.uint8)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Could not open video file: {video_path}")
            return np.zeros((128, 128, 3), dtype=np.uint8)
        
        # Set the frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        
        # Read the frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print(f"Warning: Could not read frame {frame_index} from {video_path}")
            return np.zeros((128, 128, 3), dtype=np.uint8)
        
        # Convert BGR to RGB (OpenCV uses BGR by default)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Ensure correct shape and dtype
        if frame_rgb.shape != (128, 128, 3):
            frame_rgb = cv2.resize(frame_rgb, (128, 128))
        
        return frame_rgb.astype(np.uint8)

    def _get_video_paths(self, episode_idx: int, timestep_idx: int) -> tuple[str, str]:
        """Get the video file paths for the given episode and timestep.
        
        Args:
            episode_idx: Index of the episode
            timestep_idx: Index of the timestep within the episode
            
        Returns:
            Tuple of (main_image_path, wrist_image_path)
        """
        # Extract chunk number from episode file path
        episode_file = self.episode_files[episode_idx]
        chunk_match = os.path.basename(os.path.dirname(episode_file))
        if not chunk_match.startswith('chunk-'):
            chunk_match = 'chunk-000'  # Default fallback
        
        # Extract episode number from filename
        episode_filename = os.path.basename(episode_file)
        episode_num = episode_filename.replace('episode_', '').replace('.parquet', '')
        
        # Construct video paths
        base_video_dir = os.path.join(self.data_dir, "videos", chunk_match)
        
        # Prefer agentview_left for main camera to match local
        main_left = os.path.join(
            base_video_dir, 
            "observation.images.robot0_agentview_left",
            f"episode_{episode_num}.mp4"
        )
        main_right = os.path.join(
            base_video_dir, 
            "observation.images.robot0_agentview_right",
            f"episode_{episode_num}.mp4"
        )
        main_image_path = main_left if os.path.exists(main_left) else main_right
        
        # Wrist camera (eye-in-hand)
        wrist_image_path = os.path.join(
            base_video_dir,
            "observation.images.robot0_eye_in_hand", 
            f"episode_{episode_num}.mp4"
        )
        
        return main_image_path, wrist_image_path

    def __len__(self) -> int:
        """Return the total number of timesteps across all episodes."""
        total_len = sum(self._episode_lengths)
        return total_len

    def __getitem__(self, index: SupportsIndex) -> dict:
        """Get a single timestep from the dataset.

        Args:
            index: Timestep index (across all episodes)

        Returns:
            Dictionary containing timestep data with keys:
            - image: Dictionary of images (main camera and wrist camera)
            - image_mask: Dictionary of image masks
            - state: State observation
            - actions: Action
            - task_index: Task identifier
            - episode_index: Episode identifier
        """
        # Find which episode and timestep this index corresponds to
        episode_idx = 0
        timestep_idx = index
        
        # Find the correct episode using cached lengths
        for episode_length in self._episode_lengths:
            if timestep_idx < episode_length:
                break
            timestep_idx -= episode_length
            episode_idx += 1
        else:
            # If we've gone through all episodes, loop back to the first
            episode_idx = 0
            timestep_idx = index % self._episode_lengths[0]
        
        # Load data (cached or lazy)
        if self.cache_in_memory and self._episode_data:
            episode_data = self._episode_data[episode_idx]
            state = np.array(episode_data['state'][timestep_idx])
            chunk_end = min(timestep_idx + self.action_horizon, self._episode_lengths[episode_idx])
            raw_actions = [np.array(episode_data['actions'][t]) for t in range(timestep_idx, chunk_end)]
            task_index = episode_data['task_index'][timestep_idx]
            episode_index = episode_data['episode_index'][timestep_idx]
        else:
            # Read minimal columns lazily
            episode_file = self.episode_files[episode_idx]
            cols = ['observation.state', 'action', 'task_index', 'episode_index']
            try:
                import pyarrow.parquet as pq  # type: ignore
                table = pq.read_table(episode_file, columns=cols)
                # Convert needed slices to numpy
                state_col = table.column('observation.state').to_numpy()
                action_col = table.column('action').to_numpy()
                task_col = table.column('task_index').to_numpy()
                ep_col = table.column('episode_index').to_numpy()
                # Guard against potential length mismatches by wrapping index
                episode_len = min(len(state_col), len(action_col), len(task_col), len(ep_col))
                if episode_len <= 0:
                    raise IndexError("Empty episode after column extraction")
                if timestep_idx >= episode_len:
                    timestep_idx = int(timestep_idx % episode_len)
                state = np.array(state_col[timestep_idx])
                chunk_end = min(timestep_idx + self.action_horizon, episode_len)
                raw_actions = [np.array(action_col[t]) for t in range(timestep_idx, chunk_end)]
                task_index = task_col[timestep_idx]
                episode_index = ep_col[timestep_idx]
            except Exception:
                df = pd.read_parquet(episode_file, columns=cols)
                state_vals = df['observation.state'].values
                action_vals = df['action'].values
                task_vals = df['task_index'].values
                ep_vals = df['episode_index'].values
                episode_len = min(len(state_vals), len(action_vals), len(task_vals), len(ep_vals))
                if episode_len <= 0:
                    raise IndexError("Empty episode after pandas extraction")
                if timestep_idx >= episode_len:
                    timestep_idx = int(timestep_idx % episode_len)
                state = np.array(state_vals[timestep_idx])
                chunk_end = min(timestep_idx + self.action_horizon, episode_len)
                raw_actions = [np.array(action_vals[t]) for t in range(timestep_idx, chunk_end)]
                task_index = task_vals[timestep_idx]
                episode_index = ep_vals[timestep_idx]

        if len(raw_actions) == 0:
            raw_actions = [np.zeros(12, dtype=np.float32)]
        
        # Reorder state to match local convention (ee_pos_rel, ee_rot_rel, base_pos, base_rot, gripper_qpos)
        # Groot original slices from modality.json:
        # base_position: [0:3], base_rotation: [3:7], ee_pos_rel: [7:10], ee_rot_rel: [10:14], gripper_qpos: [14:16]
        if len(state) != 16:
            raise ValueError(f"Expected state to have 16 dimensions before padding, got {len(state)}")
        
        # Recompose and reorder each action to match unified local convention [ee_pos, ee_rot, gripper, base, ctrl_mode]
        reordered_actions: list[np.ndarray] = []
        for a in raw_actions:
            if len(a) != 12:
                raise ValueError(f"Expected action to have 12 dimensions, got {len(a)}")
            # Source (Groot) indices from modality.json mapping:
            # base_motion: [0:4], control_mode: [4:5], ee_pos: [5:8], ee_rot: [8:11], gripper_close: [11:12]
            base_motion = a[0:4]
            ctrl_mode = a[4:5].astype(np.float32)
            ee_pos = a[5:8]
            ee_rot = a[8:11]
            grip_close = a[11:12].astype(np.float32)
            # Map gripper from {0,1} to {-1,1} if needed (detect non-negative binary values)
            if grip_close.min() >= -1e-6 and grip_close.max() <= 1.0 + 1e-6:
                # If values are within [0,1] and close to integers, remap
                if grip_close.min() >= -1e-6 and grip_close.max() <= 1.0 + 1e-6 and np.isclose(grip_close, np.round(grip_close)).all():
                    if grip_close.min() >= 0.0 - 1e-6:
                        grip_close = 2.0 * grip_close - 1.0
            a_reordered = np.concatenate([ee_pos, ee_rot, grip_close, base_motion, ctrl_mode], axis=-1)
            reordered_actions.append(a_reordered)
        
        # Keep native 12-d actions; do not pad here. Padding (if needed) is handled by transforms.
        padded_actions = list(reordered_actions)
        
        # If the chunk is shorter than horizon, pad by repeating the last action
        while len(padded_actions) < self.action_horizon:
            padded_actions.append(padded_actions[-1].copy())
        action = np.stack(padded_actions, axis=0)  # Shape: (action_horizon, action_dim or 12)
        
        # task_index and episode_index loaded above
        # Resolve prompt string from task index; fall back to default
        try:
            prompt = self.tasks.get(int(task_index), self.default_prompt)
        except Exception:
            prompt = self.default_prompt
        
        # Handle images based on whether they exist in the dataset
        if self._has_images:
            # Load actual video frames: use agentview_left as main image to match local, and eye_in_hand as wrist
            main_image_path, wrist_image_path = self._get_video_paths(episode_idx, timestep_idx)
            image = self._extract_video_frame(main_image_path, timestep_idx)
            wrist_image = self._extract_video_frame(wrist_image_path, timestep_idx)
        else:
            # Fallback to dummy images
            image = np.zeros((128, 128, 3), dtype=np.uint8)
            wrist_image = np.zeros((128, 128, 3), dtype=np.uint8)
        
        result = {
            "observation/image": image,
            "observation/wrist_image": wrist_image,
            "observation/state": state,
            "actions": action,
            "prompt": prompt,
            "task_index": task_index,
            "episode_index": episode_index,
        }
        
        return result


class GrootMultiDataset:
    """Multi-dataset loader for Groot-LeRobot datasets with weighted sampling."""
    
    def __init__(
        self,
        data_dirs: list[str],
        weights: list[float] | None = None,
        action_horizon: int = 1,
        shuffle: bool = False,
        seed: int = 42,
        action_dim: int = 7,
        *,
        epoch_length: int | None = None,
        include_dataset_id: bool = False,
        alpha: float | None = None,
    ):
        """Initialize the multi-dataset loader.
        
        Args:
            data_dirs: List of paths to Groot dataset directories
            weights: Optional list of weights for each dataset. If None, equal weights are used.
            action_horizon: Number of consecutive actions to include
            shuffle: Whether to shuffle episodes
            seed: Random seed for shuffling
            action_dim: Action dimension for padding
            epoch_length: Optional total number of samples per epoch used to build sampling indices.
                If None, defaults to the sum of lengths of all datasets.
            include_dataset_id: If True, __getitem__ will inject dataset id info into returned samples (for testing only).
            alpha: If weights is None, derive weights proportional to len(dataset) ** alpha.
                Common choices: 0.0 (equal), 1.0 (proportional to size).
        """
        self.data_dirs = data_dirs
        self.action_horizon = action_horizon
        self.shuffle = shuffle
        self.seed = seed
        # self.action_dim = action_dim
        self.include_dataset_id = include_dataset_id
        self.alpha = 0.0 if alpha is None else float(alpha)
        
        # Load individual datasets
        self.datasets = []
        for data_dir in data_dirs:
            dataset = GrootLeRobotDataset(
                data_dir=data_dir,
                action_horizon=action_horizon,
                shuffle=shuffle,
                seed=seed,
                # action_dim=action_dim,
            )
            self.datasets.append(dataset)
        
        # Dataset lengths for weight derivation and info
        self._dataset_lengths = [len(ds) for ds in self.datasets]
        
        # Set up weights
        if weights is None:
            # If not specified, derive weights from dataset sizes using alpha exponent
            # weights_i ∝ (len_i) ** alpha; alpha=0 -> equal; alpha=1 -> proportional to size
            base = np.array([max(1, L) for L in self._dataset_lengths], dtype=float)
            computed = np.power(base, self.alpha)
            total = float(computed.sum()) if computed.sum() > 0 else float(len(self.datasets))
            self.weights = (computed / total).tolist()
        else:
            assert len(weights) == len(self.datasets), f"Number of weights ({len(weights)}) must match number of datasets ({len(self.datasets)})"
            self.weights = list(weights)
        # Normalize weights to sum to 1
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
        # Determine epoch length for sampling index generation
        if epoch_length is None:
            self.epoch_length = int(sum(len(ds) for ds in self.datasets))
        else:
            self.epoch_length = int(epoch_length)
        
        # Create sampling indices proportional to weights
        self._create_sampling_indices()
        
        if len(self.datasets) > 1:
            print(f"Loaded {len(self.datasets)} datasets with weights: {self.weights}")
        print(f"Total samples: {len(self)}")
    
    def _create_sampling_indices(self):
        """Create weighted sampling indices for the multi-dataset.
        
        Builds an epoch of indices with length ~ epoch_length, where each dataset
        contributes approximately weight[i] * epoch_length samples. Within each
        dataset, samples cycle through episodes to cover the dataset uniformly.
        """
        self.sampling_indices = []
        
        # Build per-dataset counts
        per_dataset_counts: list[int] = []
        total_assigned = 0
        for i, w in enumerate(self.weights):
            count = max(1, int(round(w * self.epoch_length)))
            per_dataset_counts.append(count)
            total_assigned += count
        
        # Adjust counts to match epoch_length exactly (fix rounding drift)
        drift = total_assigned - self.epoch_length
        if drift != 0 and len(per_dataset_counts) > 0:
            # Reduce or increase counts starting from the largest contributors
            order = sorted(range(len(per_dataset_counts)), key=lambda i: self.weights[i], reverse=(drift > 0))
            step = 1 if drift < 0 else -1
            remaining = abs(drift)
            idx = 0
            while remaining > 0 and idx < len(order):
                j = order[idx]
                new_count = per_dataset_counts[j] + step
                if new_count >= 1:
                    per_dataset_counts[j] = new_count
                    remaining -= 1
                idx += 1
                if idx == len(order) and remaining > 0:
                    idx = 0
        
        # Construct indices by cycling through each dataset
        for dataset_idx, (dataset, count) in enumerate(zip(self.datasets, per_dataset_counts)):
            ds_len = max(1, len(dataset))
            for k in range(count):
                sample_idx = k % ds_len
                self.sampling_indices.append((dataset_idx, sample_idx))
        
        # Shuffle if requested
        if self.shuffle:
            np.random.seed(self.seed)
            np.random.shuffle(self.sampling_indices)
    
    def __len__(self) -> int:
        """Return the total number of samples across all datasets."""
        return len(self.sampling_indices)
    
    def __getitem__(self, index: SupportsIndex) -> dict:
        """Get a sample from the multi-dataset using weighted sampling."""
        if index >= len(self):
            # Wrap around if index exceeds dataset size
            index = index % len(self)
        
        dataset_idx, sample_idx = self.sampling_indices[index]
        sample = self.datasets[dataset_idx][sample_idx]
        
        # Optionally include dataset id info for testing purposes
        if self.include_dataset_id and isinstance(sample, dict):
            sample["dataset_index"] = int(dataset_idx)
            # Provide the canonical dataset directory for reference
            sample["dataset_dir"] = str(self.datasets[dataset_idx].data_dir)
        
        return sample
    
    def get_dataset_info(self) -> dict:
        """Get information about the loaded datasets."""
        info = {
            "num_datasets": len(self.datasets),
            "weights": self.weights,
            "total_samples": len(self),
            "datasets": []
        }
        
        for i, (dataset, weight) in enumerate(zip(self.datasets, self.weights)):
            dataset_info = {
                "index": i,
                "data_dir": dataset.data_dir,
                "weight": weight,
                "num_samples": len(dataset),
                "has_images": dataset.has_images,
                "dataset_length": self._dataset_lengths[i],
            }
            info["datasets"].append(dataset_info)
        
        return info 


def _load_and_convert_stats_from_meta_dir(meta_dir: pathlib.Path) -> dict[str, _transforms.NormStats] | None:
    """Shared loader that reads stats from a meta directory and converts to norm_stats.

    Applies reordering/padding and gripper remapping to actions and pads state to target dim.
    """
    target = int(os.environ.get("OPENPI_ACTION_DIM", "32"))
    for name in ("stats.json", "statistics.json"):
        path = meta_dir / name
        try:
            if not path.exists():
                continue
            data = json.loads(path.read_text())
            root = data.get("statistics", data) if isinstance(data, dict) else None
            if not isinstance(root, dict):
                continue

            def to_padded_state(entry: dict) -> _transforms.NormStats:
                mean = np.asarray(entry.get("mean"))
                std = np.asarray(entry.get("std"))
                q01 = entry.get("q01")
                q99 = entry.get("q99")
                std = np.where(std < 1e-2, 1.0, std)
                def pad(vec, fill):
                    if vec is None:
                        return None
                    v = np.asarray(vec)
                    if v.shape[-1] >= target:
                        return v
                    return np.pad(v, (0, target - v.shape[-1]), constant_values=fill)
                return _normalize.NormStats(
                    mean=pad(mean, 0.0),
                    std=pad(std, 1.0),
                    q01=pad(q01, 0.0) if q01 is not None else None,
                    q99=pad(q99, 0.0) if q99 is not None else None,
                )

            def reorder_and_pad_actions(entry: dict) -> _transforms.NormStats:
                mean = np.asarray(entry.get("mean"))
                std = np.asarray(entry.get("std"))
                q01 = np.asarray(entry.get("q01")) if entry.get("q01") is not None else None
                q99 = np.asarray(entry.get("q99")) if entry.get("q99") is not None else None

                idx = np.array([5,6,7, 8,9,10, 11, 0,1,2,3, 4], dtype=int)
                def reord(v):
                    return v[idx] if v is not None and v.shape[-1] >= 12 else v
                mean_r = reord(mean)
                std_r = reord(std)
                q01_r = reord(q01) if q01 is not None else None
                q99_r = reord(q99) if q99 is not None else None

                grip_i = 6
                if mean_r is not None and mean_r.shape[0] >= grip_i+1:
                    mean_r = mean_r.copy()
                    mean_r[grip_i] = 2.0 * mean_r[grip_i] - 1.0
                if std_r is not None and std_r.shape[0] >= grip_i+1:
                    std_r = std_r.copy()
                    std_r[grip_i] = 2.0 * std_r[grip_i]
                if q01_r is not None and q01_r.shape[0] >= grip_i+1:
                    q01_r = q01_r.copy()
                    q01_r[grip_i] = 2.0 * q01_r[grip_i] - 1.0
                if q99_r is not None and q99_r.shape[0] >= grip_i+1:
                    q99_r = q99_r.copy()
                    q99_r[grip_i] = 2.0 * q99_r[grip_i] - 1.0

                def pad(vec, fill):
                    if vec is None:
                        return None
                    v = np.asarray(vec)
                    if v.shape[-1] >= target:
                        return v
                    return np.pad(v, (0, target - v.shape[-1]), constant_values=fill)

                if std_r is not None:
                    std_r = np.where(std_r < 1e-2, 1.0, std_r)

                return _normalize.NormStats(
                    mean=pad(mean_r, 0.0),
                    std=pad(std_r, 1.0),
                    q01=pad(q01_r, 0.0) if q01_r is not None else None,
                    q99=pad(q99_r, 0.0) if q99_r is not None else None,
                )

            state_key = next((k for k in ("state", "observation.state", "observation/state") if k in root), None)
            act_key = next((k for k in ("actions", "action") if k in root), None)
            if state_key and act_key and isinstance(root[state_key], dict) and isinstance(root[act_key], dict):
                return {
                    "state": to_padded_state(root[state_key]),
                    "actions": reorder_and_pad_actions(root[act_key]),
                }
        except Exception:
            continue
    return None


def _load_stats_from_local_data_dir(data_dir: str) -> dict[str, _transforms.NormStats] | None:
    """Load/convert normalization stats from a local dataset directory via the shared meta loader."""
    return _load_and_convert_stats_from_meta_dir(pathlib.Path(data_dir) / "meta")


def _convert_stats_from_repo_meta(repo_id: str) -> dict[str, _transforms.NormStats] | None:
    """Convert dataset meta stats located in the repo cache into model-ready norm_stats."""
    home = os.environ.get("HF_LEROBOT_HOME") or os.environ.get("LEROBOT_HOME")
    if not home:
        return None
    meta_dir = pathlib.Path(home) / repo_id / "meta"
    return _load_and_convert_stats_from_meta_dir(meta_dir)


def _estimate_dataset_length(data_dir: str) -> int | None:
    """Estimate total number of timesteps in a Groot dataset directory.

    Tries fast path via pyarrow parquet metadata; falls back to None if unavailable.
    """
    try:
        import glob  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except Exception:
        return None
    try:
        pattern = str((pathlib.Path(data_dir) / "data" / "chunk-*" / "episode_*.parquet").resolve())
        files = glob.glob(pattern)
        total = 0
        for ep in files:
            try:
                pf = pq.ParquetFile(ep)
                total += pf.metadata.num_rows
            except Exception:
                continue
        return total if total > 0 else None
    except Exception:
        return None


def _combine_norm_stats_from_data_dirs(data_dirs: list[str]) -> dict[str, _transforms.NormStats] | None:
    """Combine norm stats from multiple Groot data dirs using weighted pooling.

    Weights are derived from estimated dataset lengths (timesteps) when available; otherwise equal weights.
    Q01/Q99 are combined by weighted average.
    """
    loaded: list[tuple[dict[str, _transforms.NormStats], int]] = []
    counts: list[int] = []
    for d in data_dirs:
        stats = _load_stats_from_local_data_dir(d)
        if stats is None:
            continue
        n = _estimate_dataset_length(d) or 1
        loaded.append((stats, n))
        counts.append(n)
    if not loaded:
        return None
    total = sum(counts)
    # Helper to pool one key
    def pool(key: str) -> _transforms.NormStats:
        means = []
        stds = []
        q01s = []
        q99s = []
        ws = []
        for (st, n) in loaded:
            s = st[key]
            means.append(np.asarray(s.mean))
            stds.append(np.asarray(s.std))
            q01s.append(np.asarray(s.q01) if s.q01 is not None else None)
            q99s.append(np.asarray(s.q99) if s.q99 is not None else None)
            ws.append(float(n))
        ws_arr = np.array(ws, dtype=float)
        ws_arr = ws_arr / (ws_arr.sum() if ws_arr.sum() > 0 else 1.0)
        # Combined mean
        mean_stack = np.stack(means, axis=0)
        mean_comb = np.tensordot(ws_arr, mean_stack, axes=(0, 0))
        # Combined variance via law of total variance: E[X^2] - (E[X])^2
        std_stack = np.stack(stds, axis=0)
        ex2 = np.tensordot(ws_arr, (std_stack ** 2 + mean_stack ** 2), axes=(0, 0))
        var = np.maximum(0.0, ex2 - mean_comb ** 2)
        std_comb = np.sqrt(var)
        # Combine quantiles by weighted average if available for all; else None
        def combine_q(qs: list[np.ndarray | None]) -> np.ndarray | None:
            if any(q is None for q in qs):
                return None
            q_stack = np.stack([np.asarray(q) for q in qs], axis=0)
            return np.tensordot(ws_arr, q_stack, axes=(0, 0))
        q01_comb = combine_q(q01s)
        q99_comb = combine_q(q99s)
        return _normalize.NormStats(mean=mean_comb, std=std_comb, q01=q01_comb, q99=q99_comb)

    return {"state": pool("state"), "actions": pool("actions")}