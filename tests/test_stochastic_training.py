import time
from typing import Optional

import numpy as np
import pytest
import torch as th
from eotransform.utilities.profiling import PerformanceClock
from numpy.random import RandomState
from torch.utils.data import DataLoader
from xarray import Dataset

from doubles import DelayingSplit
from factories import make_raster
from rattlinbog.config import SamplingConfig
from rattlinbog.sampling.sample_patches_from_dataset import sample_patches_from_dataset, \
    make_balanced_sample_indices_for
from rattlinbog.th_extensions.utils.data.streamed_xarray_dataset import StreamedXArrayDataset
from rattlinbog.th_extensions.utils.dataset_splitters import split_to_params_and_ground_truth, PARAMS_KEY, \
    GROUND_TRUTH_KEY

HYSTERESIS = 0.2


@pytest.fixture
def tile_dataset():
    values = np.random.randn(64, 512, 512).astype(np.float32)
    gt = np.ones((512, 512), dtype=bool)
    gt[:256, :] = False

    return Dataset({
        PARAMS_KEY: make_raster(values),
        GROUND_TRUTH_KEY: make_raster(gt)
    })


@pytest.fixture
def tile_dataset_with_nan():
    mostly_nan = np.ones((2, 128, 128), dtype=np.float32)
    mostly_nan[0, :96, :] = np.nan
    mostly_nan[1, :, :96] = np.nan

    gt = np.ones((128, 128), dtype=bool)
    gt[:112, :] = False

    return Dataset({
        PARAMS_KEY: make_raster(mostly_nan),
        GROUND_TRUTH_KEY: make_raster(gt)
    })


@pytest.fixture
def fixed_rng():
    return np.random.default_rng(42)


@pytest.fixture
def torch_tile_dataset(tile_dataset, fixed_rng):
    indices = make_balanced_sample_indices_for(tile_dataset, SamplingConfig(32, 19), fixed_rng)
    sampled = sample_patches_from_dataset(tile_dataset, indices, 19, fixed_rng)
    return StreamedXArrayDataset(sampled, split_to_params_and_ground_truth)


@pytest.fixture
def tile_dataset_dummy():
    gt = np.ones((128, 128), dtype=bool)
    gt[:112, :] = False
    return Dataset({
        PARAMS_KEY: make_raster(np.ones((2, 128, 128), dtype=np.float32)),
        GROUND_TRUTH_KEY: make_raster(gt)
    })


@pytest.fixture
def splitter_with_loading_time():
    return DelayingSplit()


@pytest.fixture
def torch_tile_dataset_with_loading_delay(tile_dataset_dummy, fixed_rng, splitter_with_loading_time):
    indices = make_balanced_sample_indices_for(tile_dataset_dummy, SamplingConfig(8, 16), fixed_rng)
    sampled = sample_patches_from_dataset(tile_dataset_dummy, indices, 16, fixed_rng)
    return StreamedXArrayDataset(sampled, splitter_with_loading_time)


def test_stochastic_patch_samples_from_dataset(tile_dataset, fixed_rng):
    indices = make_balanced_sample_indices_for(tile_dataset, SamplingConfig(32, 16), fixed_rng)
    patches = list(sample_patches_from_dataset(tile_dataset, indices, 16, fixed_rng))
    assert len(patches) == 16


def test_patch_samples_are_balanced(tile_dataset, fixed_rng):
    balanced_indices = make_balanced_sample_indices_for(tile_dataset, SamplingConfig(32, 2), fixed_rng)
    patches = list(sample_patches_from_dataset(tile_dataset, balanced_indices, 2, fixed_rng))
    a_has_mask_pixels = patches[0][GROUND_TRUTH_KEY].sum().values.item() > 0
    b_has_mask_pixels = patches[1][GROUND_TRUTH_KEY].sum().values.item() > 0
    assert (a_has_mask_pixels and not b_has_mask_pixels) or (b_has_mask_pixels and not a_has_mask_pixels)


def test_never_sample_patches_with_nans(tile_dataset_with_nan, fixed_rng):
    indices_with_nans = make_balanced_sample_indices_for(tile_dataset_with_nan, SamplingConfig(8, 4, never_nans=False),
                                                         fixed_rng)
    patches = list(sample_patches_from_dataset(tile_dataset_with_nan, indices_with_nans, 4, fixed_rng))
    assert any(patch[PARAMS_KEY].isnull().any() for patch in patches)

    indices_no_nans = make_balanced_sample_indices_for(tile_dataset_with_nan, SamplingConfig(8, 4, never_nans=True),
                                                       fixed_rng)
    patches = list(sample_patches_from_dataset(tile_dataset_with_nan, indices_no_nans, 4, fixed_rng))
    assert not any(patch[PARAMS_KEY].isnull().any() for patch in patches)


def test_sampled_patches_as_torch_dataset_can_be_loaded_by_dataloader(torch_tile_dataset):
    loader = DataLoader(torch_tile_dataset, batch_size=8)
    assert_torch_batch_sizes(loader, [8, 8, 3])


def assert_torch_batch_sizes(loader: DataLoader, expected_sizes):
    def asser_tensor_and_get_batch_size(x, y):
        assert th.is_tensor(x) and th.is_tensor(y)
        assert x.shape[0] == y.shape[0]
        return x.shape[0]

    assert [asser_tensor_and_get_batch_size(*b) for b in loader] == expected_sizes


def test_interleaved_loading_and_training(tile_dataset_dummy, splitter_with_loading_time):
    splitter_with_loading_time.set_loading_time(0.1)

    indices = make_balanced_sample_indices_for(tile_dataset_dummy, SamplingConfig(8, 16))
    sampled = sample_patches_from_dataset(tile_dataset_dummy, indices, 16)
    clock_init = PerformanceClock('init')
    with clock_init.measure():
        stream = StreamedXArrayDataset(sampled, splitter_with_loading_time)

    assert clock_init.total_measures < 1.0

    clock_loading = PerformanceClock('load')
    with clock_loading.measure():
        for _ in DataLoader(stream, batch_size=4):
            time.sleep(0.4)

    assert clock_loading.total_measures < 2.0 + HYSTERESIS  # < 2 # prev 1.617
