import os
os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
import dataclasses
import pathlib
import pytest
import numpy as np

from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

# Defaults
DEFAULT_LEROBOT_HOME = "/data2/.cache/huggingface/lerobot"
GROOT_REPO_ID = "robocasa/BreadSetupSlicing_groot"


def _ensure_env_defaults():
    # Use LEROBOT_HOME for Groot local discovery
    os.environ.setdefault("LEROBOT_HOME", DEFAULT_LEROBOT_HOME)


def _groot_local_dir() -> pathlib.Path:
    root = pathlib.Path(os.environ.get("LEROBOT_HOME", DEFAULT_LEROBOT_HOME))
    return root / GROOT_REPO_ID


def test_groot_dataset_direct_transform():
    _ensure_env_defaults()
    local_dir = _groot_local_dir()
    if not (local_dir / "meta" / "modality.json").exists():
        pytest.skip(f"Missing Groot dataset at {local_dir}")

    # Load config
    cfg = _config.get_config("pi0_groot_lerobot_low_mem_finetune")

    # Build DataConfig and dataset
    data_conf = cfg.data.create(cfg.assets_dirs, cfg.model)
    ds = _data_loader.create_torch_dataset(
        data_conf,
        action_horizon=cfg.model.action_horizon,
        model_config=cfg.model,
    )

    # Transform without norm stats and fetch samples directly
    tds = _data_loader.transform_dataset(ds, data_conf, skip_norm_stats=True)

    # Basic sampling: check first 3 timesteps
    for idx in [0, 1, 2]:
        sample = tds[idx]
        # Required keys
        for k in ["state", "image", "image_mask"]:
            assert k in sample, f"Missing key {k} at idx {idx}"
        # Shapes
        assert np.shape(sample["state"]) == (cfg.model.action_dim,)
        assert np.shape(sample["image"]["base_0_rgb"]) == (224, 224, 3)
        assert np.shape(sample["image"]["left_wrist_0_rgb"]) == (224, 224, 3)
        # Actions
        assert "actions" in sample
        assert np.shape(sample["actions"]) == (cfg.model.action_horizon, cfg.model.action_dim) 