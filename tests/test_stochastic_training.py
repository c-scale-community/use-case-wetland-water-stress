from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import torch as th
import xarray as xr
from approvaltests import verify_with_namer_and_writer
from numpy.random import RandomState
from pytest_approvaltests_geo import GeoOptions, CompareGeoZarrs, ReportGeoZarrs, ExistingDirWriter
from torch.utils.data import DataLoader
from xarray import Dataset

from rattlinbog.io_xarray.store_as_compressed_zarr import store_as_compressed_zarr
from rattlinbog.sampling.sample_patches_from_dataset import sample_patches_from_dataset
from rattlinbog.th_extensions.utils.data.streamed_xarray_dataset import StreamedXArrayDataset
from rattlinbog.th_extensions.utils.dataset_splitters import split_to_params_and_labels


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
        "/eodc/private/tuwgeo/users/braml/data/wetland/hparam/V1M0R1/EQUI7_EU020M/E051N015T3/SIG0-HPAR-MASK____RAMSAR-AT-01_E051N015T3_EU020M__.zarr")
    return xr.open_zarr(zarr)


@pytest.fixture
def torch_tile_dataset(tile_dataset):
    sampled = sample_patches_from_dataset(tile_dataset, 32, 19, np.random.default_rng(42))
    return StreamedXArrayDataset(sampled, split_to_params_and_labels)


def test_stochastic_patch_samples_from_dataset(tile_dataset, verify_raster_as_geo_zarr):
    patches = list(sample_patches_from_dataset(tile_dataset, 32, 16, np.random.default_rng(42)))
    assert len(patches) == 16
    verify_raster_as_geo_zarr(patches[0])


def test_patch_samples_are_balanced(tile_dataset, verify_raster_as_geo_zarr):
    patches = list(sample_patches_from_dataset(tile_dataset, 32, 2, np.random.default_rng(42)))
    a_has_mask_pixels = patches[0]['mask'].sum().values.item() > 0
    b_has_mask_pixels = patches[1]['mask'].sum().values.item() > 0
    assert (a_has_mask_pixels and not b_has_mask_pixels) or (b_has_mask_pixels and not a_has_mask_pixels)


def test_sampled_patches_as_torch_dataset_can_be_loaded_by_dataloader(torch_tile_dataset):
    loader = DataLoader(torch_tile_dataset, batch_size=8)
    assert_torch_batch_sizes(loader, [8, 8, 3])


def assert_torch_batch_sizes(loader: DataLoader, expected_sizes):
    def asser_tensor_and_get_batch_size(x, y):
        assert th.is_tensor(x) and th.is_tensor(y)
        assert x.shape[0] == y.shape[0]
        return x.shape[0]

    assert [asser_tensor_and_get_batch_size(*b) for b in loader] == expected_sizes
