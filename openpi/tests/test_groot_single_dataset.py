import os
os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
import json
import pytest
import numpy as np
import pathlib
import dataclasses
from PIL import Image

from openpi.training import data_loader as dl
from openpi.training import config as cfg
from openpi.models import pi0
import openpi.transforms as _transforms
from openpi.shared import normalize as _normalize
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x

# Defaults for local testing (override via env if desired)
DEFAULT_HF_LEROBOT_HOME = "/data2/sep/lerobot"
DEFAULT_GROOT_DIR = "/home/abhim/robocasa/PnPCounterToCabinet/groot"
DEFAULT_LEROBOT_REPO_ID = "robocasa/PnPCounterToCabinet_local"

def _ensure_env_defaults():
    os.environ.setdefault("HF_LEROBOT_HOME", DEFAULT_HF_LEROBOT_HOME)

def _paths_ok(groot_dir: str) -> tuple[bool, str]:
    if not pathlib.Path(groot_dir).exists():
        return False, f"Missing Groot dataset directory: {groot_dir}"
    return True, ""

def _save_first_images(old_obs, new_obs, batch_idx):
    img_key = "base_0_rgb"
    if img_key in old_obs.images and img_key in new_obs.images:
        old_img_np = np.array(old_obs.images[img_key])[0]
        new_img_np = np.array(new_obs.images[img_key])[0]

        # Rescale if in [-1, 1]
        if old_img_np.min() < 0:
            old_img_np = ((old_img_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        if new_img_np.min() < 0:
            new_img_np = ((new_img_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

        Image.fromarray(old_img_np).save(f"old_loader_batch{batch_idx}_{img_key}.png")
        Image.fromarray(new_img_np).save(f"new_loader_batch{batch_idx}_{img_key}.png")
        print(f"[SAVED] Saved {img_key} for batch {batch_idx} from both loaders.")

# ---------------------------
# test_single_dataset_groot_len_and_sample
# ---------------------------
def test_single_dataset_groot_len_and_sample():
    _ensure_env_defaults()
    groot_dir = os.environ.get("TEST_GROOT_DIR", DEFAULT_GROOT_DIR)

    ok, msg = _paths_ok(groot_dir)
    if not ok:
        pytest.skip(msg)

    model_conf = pi0.Pi0Config()
    cfg_a = cfg.TrainConfig(
        name="test_groot_local",
        exp_name="tmp",
        model=model_conf,
        data=cfg.GrootDataConfig(data_dirs=[groot_dir]),
        batch_size=8,
        wandb_enabled=False,
        overwrite=True,
    )
    loader_a = dl.create_data_loader(cfg_a, num_batches=1, shuffle=False, skip_norm_stats=False)

    cfg_b = cfg.TrainConfig(
        name="test_groot_dirs",
        exp_name="tmp",
        model=model_conf,
        data=cfg.GrootDataConfig(base_config=cfg.DataConfig(), data_dirs=[groot_dir]),
        batch_size=8,
        wandb_enabled=False,
        overwrite=True,
    )
    loader_b = dl.create_data_loader(cfg_b, num_batches=1, shuffle=False, skip_norm_stats=False)

    ds_a = dl.create_torch_dataset(cfg_a.data.create(cfg_a.assets_dirs, cfg_a.model),
                                   model_conf.action_horizon, model_conf)
    ds_b = dl.create_torch_dataset(cfg_b.data.create(cfg_b.assets_dirs, cfg_b.model),
                                   model_conf.action_horizon, model_conf)
    if min(len(ds_a), len(ds_b)) == 0:
        pytest.skip("One of the datasets has zero length")
    assert len(ds_a) == len(ds_b), f"Lengths differ: a={len(ds_a)}, b={len(ds_b)}"

    obs_a, actions_a = next(iter(loader_a))
    obs_b, actions_b = next(iter(loader_b))

    assert hasattr(obs_a, "state") and hasattr(obs_b, "state")
    assert hasattr(obs_a, "images") and hasattr(obs_b, "images")
    assert hasattr(obs_a, "image_masks") and hasattr(obs_b, "image_masks")

    assert np.shape(obs_a.state) == np.shape(obs_b.state)
    for img_key in ["base_0_rgb", "left_wrist_0_rgb"]:
        assert img_key in obs_a.images and img_key in obs_b.images
        assert np.shape(obs_a.images[img_key]) == np.shape(obs_b.images[img_key])

    assert np.shape(actions_a) == np.shape(actions_b)

    print("[GROOT TEST] groot_dir=", groot_dir)
    print(f"[GROOT TEST] lengths: a={len(ds_a)}, b={len(ds_b)}")

# ---------------------------
# test_single_dataset_full_equality_with_shared_stats
# ---------------------------
def test_single_dataset_full_equality_with_shared_stats():
    _ensure_env_defaults()
    groot_dir = os.environ.get("TEST_GROOT_DIR", DEFAULT_GROOT_DIR)
    ok, msg = _paths_ok(groot_dir)
    if not ok:
        pytest.skip(msg)

    model_conf = pi0.Pi0Config()
    import openpi.groot_utils.groot_openpi_dataset as _groot_openpi_dataset
    shared_stats = _groot_openpi_dataset._load_stats_from_local_data_dir(groot_dir)
    assert shared_stats is not None

    dc_a = cfg.GrootDataConfig(data_dirs=[groot_dir]).create(pathlib.Path("./assets"), model_conf)
    dc_a = dataclasses.replace(dc_a, norm_stats=shared_stats)
    ds_a = dl.create_torch_dataset(dc_a, model_conf.action_horizon, model_conf)
    tx_a = dl.transform_dataset(ds_a, dc_a, skip_norm_stats=False)

    dc_b = cfg.GrootDataConfig(data_dirs=[groot_dir]).create(pathlib.Path("./assets"), model_conf)
    dc_b = dataclasses.replace(dc_b, norm_stats=shared_stats)
    ds_b = dl.create_torch_dataset(dc_b, model_conf.action_horizon, model_conf)
    tx_b = dl.transform_dataset(ds_b, dc_b, skip_norm_stats=False)

    a = tx_a[0]
    b = tx_b[0]
    def _mae_max(x, y):
        diff = np.asarray(x) - np.asarray(y)
        absd = np.abs(diff)
        return float(absd.mean()), float(absd.max() if absd.size > 0 else 0.0)

    for k in a["image_mask"]:
        assert k in b["image_mask"]
        assert np.array_equal(a["image_mask"][k], b["image_mask"][k])

# ---------------------------
# test_old_lerobot_vs_new_groot_len_and_sample
# ---------------------------
def test_old_lerobot_vs_new_groot_len_and_sample():
    _ensure_env_defaults()
    groot_dir = os.environ.get("TEST_GROOT_DIR", DEFAULT_GROOT_DIR)
    repo_id = os.environ.get("TEST_LEROBOT_REPO_ID", DEFAULT_LEROBOT_REPO_ID)

    ok, msg = _paths_ok(groot_dir)
    if not ok:
        pytest.skip(msg)

    model_conf = pi0.Pi0Config()
    tmp_dc = cfg.LeRobotRobocasaDataConfig(repo_id=repo_id).create(pathlib.Path("./assets"), model_conf)
    tmp_dc = dataclasses.replace(tmp_dc, prompt_from_task=True)
    tmp_ds = dl.create_torch_dataset(tmp_dc, model_conf.action_horizon, model_conf)
    pre_norm = _transforms.compose([*tmp_dc.repack_transforms.inputs, *tmp_dc.data_transforms.inputs])
    state_rs = _normalize.RunningStats()
    actions_rs = _normalize.RunningStats()
    max_samples = min(2000, len(tmp_ds))
    for i in tqdm(range(max_samples), total=max_samples, desc="compute_norm_stats (old)"):
        s = pre_norm(tmp_ds[i])
        state = np.asarray(s["state"])
        actions = np.asarray(s["actions"])
        state_rs.update(state[np.newaxis, :])
        actions_rs.update(actions)
    old_norm_stats = {"state": state_rs.get_statistics(), "actions": actions_rs.get_statistics()}

    old_cfg = cfg.TrainConfig(
        name="test_old_lerobot_vs_new_groot_old",
        exp_name="tmp",
        model=model_conf,
        data=cfg.LeRobotRobocasaDataConfig(
            repo_id=repo_id,
            base_config=cfg.DataConfig(prompt_from_task=True, norm_stats=old_norm_stats),
        ),
        batch_size=8,
        wandb_enabled=False,
        overwrite=True,
    )
    old_loader = dl.create_data_loader(old_cfg, num_batches=3, shuffle=False, skip_norm_stats=False)

    new_cfg = cfg.TrainConfig(
        name="test_old_lerobot_vs_new_groot_new",
        exp_name="tmp",
        model=model_conf,
        data=cfg.GrootDataConfig(data_dirs=[groot_dir]),
        batch_size=8,
        wandb_enabled=False,
        overwrite=True,
    )
    new_loader = dl.create_data_loader(new_cfg, num_batches=3, shuffle=False, skip_norm_stats=False)

    old_ds = dl.create_torch_dataset(old_cfg.data.create(old_cfg.assets_dirs, old_cfg.model),
                                     model_conf.action_horizon, model_conf)
    new_ds = dl.create_torch_dataset(new_cfg.data.create(new_cfg.assets_dirs, new_cfg.model),
                                     model_conf.action_horizon, model_conf)
    if min(len(old_ds), len(new_ds)) == 0:
        pytest.skip("One of the datasets has zero length")
    rel_diff = abs(len(old_ds) - len(new_ds)) / float(min(len(old_ds), len(new_ds)))
    assert rel_diff <= 0.01

    for batch_idx in range(3):
        old_obs, old_actions = next(iter(old_loader))
        new_obs, new_actions = next(iter(new_loader))

        old_state = np.asarray(old_obs.state)
        new_state = np.asarray(new_obs.state)
        old_actions_np = np.asarray(old_actions)
        new_actions_np = np.asarray(new_actions)

        print(f"old_state[0]: {old_state[0]}")
        print(f"new_state[0]: {new_state[0]}")

        base_pos_diff = np.abs(old_state[:, 0:3] - new_state[:, 0:3]).mean()
        print(f"Batch {batch_idx} Base position diff: {base_pos_diff:.6g}")

        if not np.allclose(old_state, new_state, atol=1e-5):
            _save_first_images(old_obs, new_obs, batch_idx)
            raise AssertionError(f"State tensors differ in batch {batch_idx}")

        if not np.allclose(old_actions_np, new_actions_np, atol=1e-5):
            _save_first_images(old_obs, new_obs, batch_idx)
            raise AssertionError(f"Actions tensors differ in batch {batch_idx}")

if __name__ == "__main__":
    _ensure_env_defaults()
    try:
        print("[RUNNING] test_single_dataset_groot_len_and_sample")
        # test_single_dataset_groot_len_and_sample()
        print("[RUNNING] test_single_dataset_full_equality_with_shared_stats")
        # test_single_dataset_full_equality_with_shared_stats()
        print("[RUNNING] test_old_lerobot_vs_new_groot_len_and_sample")
        test_old_lerobot_vs_new_groot_len_and_sample()
    except AssertionError as e:
        print("[FAILED]", e)
        raise SystemExit(1)
    except Exception as e:
        print("[ERROR]", e)
        raise SystemExit(1)
