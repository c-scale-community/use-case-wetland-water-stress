import numpy as np
from xarray import Dataset, DataArray

from rattlinbog.sampling import sample_uniformly


def test_uniformly_sample_dataset():
    dataset = Dataset({'var0': DataArray(np.ones((1, 1, 32, 32)), dims=('time', 'parameter', 'y', 'x')),
                       'var1': DataArray(np.ones((1, 1, 32, 32)), dims=('time', 'parameter', 'y', 'x'))})
    dataset['var1'][:, :, :16, :] = 0
    sample_mask = DataArray(np.full((32, 32), True), dims=('y', 'x'), name="total")
    sample_mask[:, :16] = False

    sampled = sample_uniformly(dataset, [sample_mask], 32, seed=42)
    assert sampled['sampled'].shape[0] == 32
    assert sampled['var0'].loc['total'].sum('sampled').item() == 32
    assert 14 <= sampled['var1'].loc['total'].sum('sampled').item() <= 18


def test_uniformly_sample_dataset_from_multiple_masks():
    dataset = Dataset({'var0': DataArray(np.ones((1, 1, 32, 32)), dims=('time', 'parameter', 'y', 'x'))})
    dataset['var0'][:, :, :16, :] = 0
    sample_masks = [
        DataArray(np.full((32, 32), True), dims=('y', 'x'), name="upper_quarter"),
        DataArray(np.full((32, 32), True), dims=('y', 'x'), name="left_quater"),
    ]
    sample_masks[0][:, :8] = False
    sample_masks[1][:8, :] = False

    sampled = sample_uniformly(dataset, sample_masks, 32, seed=42)
    assert 14 <= sampled['var0'].loc['upper_quarter'].sum('sampled').item() <= 18
    assert 22 <= sampled['var0'].loc['left_quater'].sum('sampled').item() <= 26
