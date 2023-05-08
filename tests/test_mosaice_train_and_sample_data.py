from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
import xarray as xr
from xarray import Dataset

from rattlinbog.th_extensions.utils.dataset_splitters import PARAMS_KEY, GROUND_TRUTH_KEY
from tests.helpers.factories import make_raster


@pytest.fixture
def train_west(tmp_path):
    ds = make_train_dataset(np.arange(100, 0, -1), np.arange(0, 100))
    zarr = tmp_path / "train_west.zarr"
    ds.to_zarr(zarr)
    return zarr


@pytest.fixture
def train_east(tmp_path):
    ds = make_train_dataset(np.arange(100, 0, -1), np.arange(100, 200))
    zarr = tmp_path / "train_east.zarr"
    ds.to_zarr(zarr)
    return zarr


@pytest.fixture
def train_offset_by_one_east(tmp_path):
    ds = make_train_dataset(np.arange(100, 0, -1), np.arange(200, 300))
    zarr = tmp_path / "train_offset_by_one_east.zarr"
    ds.to_zarr(zarr)
    return zarr


def make_train_dataset(ys, xs):
    gt = np.ones((100, 100), dtype=np.bool)
    gt[:50, :] = False
    return Dataset({
        PARAMS_KEY: make_raster(np.ones((2, 100, 100), dtype=np.float32), coords={'y': ys, 'x': xs}),
        GROUND_TRUTH_KEY: make_raster(gt, coords={'y': ys, 'x': xs})
    })


def concatenate_training_dataset(*zarrs: Tuple[Path, ...]) -> Dataset:
    return xr.open_mfdataset(zarrs, engine='zarr', combine='by_coords', chunks={})


def test_concatenate_two_training_datasets(train_west, train_east):
    train = concatenate_training_dataset(train_west, train_east)
    assert_coordinates(train.x.values, np.arange(0, 200))
    assert_coordinates(train.y.values, np.arange(100, 0, -1))


def assert_coordinates(actual_coords, expected_coords):
    np.testing.assert_array_equal(actual_coords, expected_coords)


def test_concatenate_non_neighbouring_training_datasets(train_west, train_offset_by_one_east):
    train = concatenate_training_dataset(train_west, train_offset_by_one_east)
    assert_coordinates(train.x.values, np.concatenate([np.arange(0, 100), np.arange(200, 300)]))
    assert_coordinates(train.y.values, np.arange(100, 0, -1))


def test_selecting_from_offset_produces_zero_size_patch(train_west, train_offset_by_one_east):
    train = concatenate_training_dataset(train_west, train_offset_by_one_east)
    out_of_bounds = train.sel(y=slice(30, 40), x=slice(150, 160))
    assert out_of_bounds.dims['y'] == 0 and out_of_bounds.dims['x'] == 0
