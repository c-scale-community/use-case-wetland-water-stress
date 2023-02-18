import time
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import torch as th
import xarray as xr
from approvaltests import verify_with_namer_and_writer
from eotransform.utilities.profiling import PerformanceClock
from numpy.random import RandomState
from pytest_approvaltests_geo import GeoOptions, CompareGeoZarrs, ReportGeoZarrs, ExistingDirWriter
from torch.utils.data import DataLoader
from xarray import Dataset

from doubles import DelayingSplit
from factories import make_raster
from rattlinbog.io_xarray.store_as_compressed_zarr import store_as_compressed_zarr
from rattlinbog.sampling.sample_patches_from_dataset import sample_patches_from_dataset, SamplingConfig
from rattlinbog.th_extensions.utils.data.streamed_xarray_dataset import StreamedXArrayDataset
from rattlinbog.th_extensions.utils.dataset_splitters import split_to_params_and_ground_truth, PARAMS_KEY, \
    GROUND_TRUTH_KEY

HYSTERESIS = 0.2


@pytest.fixture
def verify_raster_as_geo_zarr(tmp_path):
    def verify(dataset: Dataset,
               *,  # enforce keyword arguments - https://www.python.org/dev/peps/pep-3102/
               options: Optional[GeoOptions] = None):
        options = options or GeoOptions()
        file_name = tmp_path / f"verifying-{dataset.name}-tmp.zarr"
        store_as_compressed_zarr(dataset, file_name)
        zarr_comparator = CompareGeoZarrs(options.scrub_tags, options.tolerance)
        zarr_reporter = ReportGeoZarrs(options.scrub_tags, options.tolerance)
        options = options.with_comparator(zarr_comparator)
        options = options.with_reporter(zarr_reporter)

        namer = options.namer
        namer.set_extension(file_name.suffix)
        verify_with_namer_and_writer(
            namer=namer,
            writer=ExistingDirWriter(file_name.as_posix()),
            options=options)

    return verify


@pytest.fixture
def tile_dataset():
    zarr = Path(
        "/eodc/private/tuwgeo/users/braml/data/wetland/hparam/V1M0R1/EQUI7_EU020M/E051N015T3/SIG0-HPAR-MASK____RAMSAR-AT-01-ROI_E051N015T3_EU020M__.zarr")
    return xr.open_zarr(zarr)


@pytest.fixture
def tile_dataset_with_nan():
    mostly_nan = np.ones((2, 128, 128), dtype=np.float32)
    mostly_nan[0, :96, :] = np.nan
    mostly_nan[1, :, :96] = np.nan

    gt = np.ones((128, 128), dtype=np.bool)
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
    sampled = sample_patches_from_dataset(tile_dataset, SamplingConfig(32, 19), fixed_rng)
    return StreamedXArrayDataset(sampled, split_to_params_and_ground_truth)


@pytest.fixture
def tile_dataset_dummy():
    gt = np.ones((128, 128), dtype=np.bool)
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
    sampled = sample_patches_from_dataset(tile_dataset_dummy, SamplingConfig(8, 16), fixed_rng)
    return StreamedXArrayDataset(sampled, splitter_with_loading_time)


def test_stochastic_patch_samples_from_dataset(tile_dataset, verify_raster_as_geo_zarr, fixed_rng):
    patches = list(sample_patches_from_dataset(tile_dataset, SamplingConfig(32, 16), fixed_rng))
    assert len(patches) == 16
    verify_raster_as_geo_zarr(patches[0])


def test_patch_samples_are_balanced(tile_dataset, verify_raster_as_geo_zarr, fixed_rng):
    patches = list(sample_patches_from_dataset(tile_dataset, SamplingConfig(32, 2), fixed_rng))
    a_has_mask_pixels = patches[0][GROUND_TRUTH_KEY].sum().values.item() > 0
    b_has_mask_pixels = patches[1][GROUND_TRUTH_KEY].sum().values.item() > 0
    assert (a_has_mask_pixels and not b_has_mask_pixels) or (b_has_mask_pixels and not a_has_mask_pixels)


def test_patch_samples_caches_valid_indices_in_dataset(tile_dataset, fixed_rng):
    init = PerformanceClock("init")
    with init.measure():
        next(iter(sample_patches_from_dataset(tile_dataset, SamplingConfig(32, 2), fixed_rng)))
    cached = PerformanceClock("cached")
    with cached.measure():
        next(iter(sample_patches_from_dataset(tile_dataset, SamplingConfig(32, 2), fixed_rng)))

    assert cached.total_measures < init.total_measures / 2


@pytest.mark.parametrize('new_config', [
    SamplingConfig(patch_size=16, n_samples=2, never_nans=False),
    SamplingConfig(patch_size=32, n_samples=3, never_nans=False),
    SamplingConfig(patch_size=32, n_samples=2, never_nans=True),
])
def test_patch_samples_invalidates_caches_appropriately(tile_dataset, new_config):
    init = PerformanceClock("init")
    with init.measure():
        next(iter(sample_patches_from_dataset(tile_dataset,
                                              SamplingConfig(patch_size=32, n_samples=2, never_nans=False))))
    cached = PerformanceClock("cached")
    with cached.measure():
        next(iter(sample_patches_from_dataset(tile_dataset, new_config)))

    assert cached.total_measures > init.total_measures / 2


def test_never_sample_patches_with_nans(tile_dataset_with_nan, verify_raster_as_geo_zarr, fixed_rng):
    patches = list(sample_patches_from_dataset(tile_dataset_with_nan,
                                               SamplingConfig(8, 4, never_nans=False), fixed_rng))
    assert any(patch[PARAMS_KEY].isnull().any() for patch in patches)
    patches = list(sample_patches_from_dataset(tile_dataset_with_nan,
                                               SamplingConfig(8, 4, never_nans=True), fixed_rng))
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

    sampled = sample_patches_from_dataset(tile_dataset_dummy, SamplingConfig(8, 16))
    clock_init = PerformanceClock('init')
    with clock_init.measure():
        stream = StreamedXArrayDataset(sampled, splitter_with_loading_time)

    assert clock_init.total_measures < 1.0

    clock_loading = PerformanceClock('load')
    with clock_loading.measure():
        for _ in DataLoader(stream, batch_size=4):
            time.sleep(0.4)

    assert clock_loading.total_measures < 2.0 + HYSTERESIS  # < 2 # prev 1.617
