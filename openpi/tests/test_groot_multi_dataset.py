import os
import pytest
import numpy as np
import torch

from openpi.training.groot_lerobot_dataset import GrootMultiDataset


@pytest.mark.skipif(
    os.environ.get("TEST_GROOT_DIR_A") is None or os.environ.get("TEST_GROOT_DIR_B") is None,
    reason="Set TEST_GROOT_DIR_A and TEST_GROOT_DIR_B to run this test",
)
def test_groot_multi_sampling_weights():
    dir_a = os.environ["TEST_GROOT_DIR_A"]
    dir_b = os.environ["TEST_GROOT_DIR_B"]
    weights = [0.9, 0.1]

    # Wrap the original class to annotate samples with dataset_index for counting
    class AnnotatedMulti(GrootMultiDataset):
        def __getitem__(self, index):
            sample = super().__getitem__(index)
            # Add testing-only field
            sample["dataset_index"] = int(self.sampling_indices[index][0])
            return sample

    multi = AnnotatedMulti(
        data_dirs=[dir_a, dir_b],
        weights=weights,
        action_horizon=4,
        shuffle=True,
        action_dim=12,
    )

    # Iterate over a DataLoader and accumulate counts to reach ~1000 samples
    loader = torch.utils.data.DataLoader(multi, batch_size=16, shuffle=True)
    target_samples = 1000
    counts = np.zeros(2, dtype=int)
    seen = 0
    for batch in loader:
        # dataset_index will be a tensor of shape (B,)
        di = batch["dataset_index"].numpy()
        # Accumulate counts from this batch
        binc = np.bincount(di, minlength=2)
        counts[: len(binc)] += binc
        seen += di.shape[0]
        if seen >= target_samples:
            break

    empirical = counts / counts.sum()
    target = np.array(weights) / np.sum(weights)

    assert np.allclose(empirical, target, atol=0.05), f"Empirical {empirical.tolist()} vs target {target.tolist()}" 