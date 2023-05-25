from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from affine import Affine
from equi7grid.equi7grid import Equi7Grid
from geopathfinder.naming_conventions.yeoda_naming import YeodaFilename
from xarray import DataArray

from rattlinbog.config import Restructure
from rattlinbog.restructure_data import restructure
from rattlinbog.th_extensions.utils.dataset_splitters import PARAMS_KEY, GROUND_TRUTH_KEY


@pytest.fixture
def tile_under_test():
    return "EU500M_E048N012T6"


@pytest.fixture
def hparam_root(tmp_path, tile_under_test):
    root = tmp_path / "cscale_hparam/SIG0-HPAR/V0M2R1"
    hparams = ["SIG0-HPAR-C1", "SIG0-HPAR-C2", "SIG0-HPAR-C3",
               "SIG0-HPAR-M0", "SIG0-HPAR-NOBS", "SIG0-HPAR-STD",
               "SIG0-HPAR-S1", "SIG0-HPAR-S2", "SIG0-HPAR-S3"]
    generate_filled_tile_arrays(root, tile_under_test, np.arange(9), hparams, "A001",
                                datetime(2019, 1, 1), datetime(2020, 1, 1))
    generate_filled_tile_arrays(root, tile_under_test, np.arange(9) * 2, hparams, "D001",
                                datetime(2019, 1, 1), datetime(2020, 1, 1))
    generate_filled_tile_arrays(root, tile_under_test, np.arange(9), hparams, "A001",
                                datetime(2018, 1, 1), datetime(2020, 1, 1))
    generate_filled_tile_arrays(root, tile_under_test, np.arange(9) * 3, hparams, "D001",
                                datetime(2018, 1, 1), datetime(2020, 1, 1))
    return root


def generate_filled_tile_arrays(root: Path, tile_long: str, var_values, var_names, extra_fields, datetime_1, datetime_2):
    e7tile = Equi7Grid(500).create_tile(tile_long)
    grid = grid_from(tile_long)
    tile = tile_name_from(tile_long)
    if not isinstance(var_names, list):
        var_names = [var_names] * len(var_values)
    if not isinstance(extra_fields, list):
        extra_fields = [extra_fields] * len(var_values)
    for val, name, extra in zip(var_values, var_names, extra_fields):
        tif_name = str(YeodaFilename(
            dict(var_name=name, datetime_1=datetime_1, datetime_2=datetime_2, extra_field=extra)))
        tif_path = root / grid / tile / tif_name
        tif_path.parent.mkdir(parents=True, exist_ok=True)
        da = DataArray(np.full((1, 1200, 1200), val, dtype=np.int16), dims=['band', 'y', 'x'])
        da.rio.write_nodata(-9999, inplace=True)
        da.rio.write_crs(e7tile.core.projection.proj4, inplace=True)
        da.rio.write_transform(Affine.from_gdal(*e7tile.geotransform()), inplace=True)
        da.rio.to_raster(tif_path, compress='ZSTD', tags=dict(scale_factor="1", scales=[1]))


def grid_from(tile_long: str) -> str:
    return f"EQUI7_{tile_long.split('_')[0]}"


def tile_name_from(tile_long: str) -> str:
    return tile_long.split('_')[1]


@pytest.fixture
def mask_root(tmp_path, tile_under_test):
    root = tmp_path / "rasterized/CCI/V1M0R1"
    generate_filled_tile_arrays(root, tile_under_test, [1],
                                ["MASK-CCI"], "2018",
                                datetime(2018, 1, 1), datetime(2018, 12, 31))
    generate_filled_tile_arrays(root, tile_under_test, [0],
                                ["MASK-CCI"], "2016",
                                datetime(2016, 1, 1), datetime(2016, 12, 31))
    return root


@pytest.fixture
def hparam_dst_root(tmp_path):
    return tmp_path / "hparam/V1M0R1"

@pytest.fixture
def mmean_dst_root(tmp_path):
    return tmp_path / "mmean/V1M0R1"

@pytest.fixture
def hparam_config():
    return Restructure(500, [(0, 0, 15000, 15000)], 'hparam', 2019, 2020, "2018")


@pytest.fixture
def hparam_zarr():
    return "SIG0-HPAR-MASK-CCI_20190101T000000_20200101T000000__2018-ROI-0-0-15000-15000_E048N012T6_EU500M__.zarr"


def test_restructure_hparam_train_data(tile_under_test, hparam_root, mask_root, hparam_dst_root, hparam_config, hparam_zarr):
    restructure(tile_under_test, hparam_root, mask_root, hparam_dst_root, hparam_config)
    restructured_zarr = hparam_dst_root / grid_from(tile_under_test) / tile_name_from(tile_under_test) / hparam_zarr
    assert restructured_zarr.exists()
    restructured_ds = xr.open_zarr(restructured_zarr)
    assert_selected_correct_hparams(restructured_ds[PARAMS_KEY])
    assert_selected_correct_mask(restructured_ds[GROUND_TRUTH_KEY])


def assert_selected_correct_hparams(hparams: DataArray) -> None:
    assert np.all(hparams[0] == 0).values.item()
    assert np.all(hparams[1] == 5 / 3).values.item()
    assert np.all(hparams[2] == 3 + 1 / 3).values.item()
    assert np.all(hparams[3] == 5).values.item()
    assert np.all(hparams[4] == 12).values.item()
    assert np.all(hparams[5] == 10).values.item()
    assert np.all(hparams[6] == 11 + 2 / 3).values.item()
    assert np.all(hparams[7] == 13 + 1 / 3).values.item()
    assert np.all(hparams[8] == 8 + 1 / 3).values.item()
    assert np.all(hparams[9] == np.sqrt(10 ** 2)).values.item()
    assert np.all(hparams[10] == np.arctan2(10, 0)).values.item()


def assert_selected_correct_mask(mask: DataArray) -> None:
    assert np.all(mask == 1)


@pytest.fixture
def mmean_root(tmp_path, tile_under_test):
    root = tmp_path / "mmean/MEAN-SIG0-MONTH/V1M0R1"
    generate_filled_tile_arrays(root, tile_under_test, np.arange(12), "MEAN-SIG0-MONTH",
                                [f"OAVG-CLIM-{m}" for m in range(1, 13)],
                                datetime(2019, 1, 1), datetime(2019, 12, 31))
    generate_filled_tile_arrays(root, tile_under_test, np.arange(12) * 2, "MEAN-SIG0-MONTH",
                                [f"D001-CLIM-{m}" for m in range(1, 13)],
                                datetime(2019, 1, 1), datetime(2019, 12, 31))
    generate_filled_tile_arrays(root, tile_under_test, np.arange(12) * 3, "MEAN-SIG0-MONTH",
                                [f"OAVG-CLIM-{m}" for m in range(1, 13)],
                                datetime(2018, 1, 1), datetime(2018, 12, 31))
    return root

@pytest.fixture
def mmean_config():
    return Restructure(500, [(0, 0, 15000, 15000)], 'mmeans', 2019, 2019, "2018")


@pytest.fixture
def mmean_zarr():
    return "MEAN-SIG0-MONTH-MASK-CCI_20190101T000000_20190101T000000__2018-ROI-0-0-15000-15000_E048N012T6_EU500M__.zarr"


def test_restructure_mmean_train_data(tile_under_test, mmean_root, mask_root, mmean_dst_root, mmean_config, mmean_zarr):
    restructure(tile_under_test, mmean_root, mask_root, mmean_dst_root, mmean_config)
    restructured_zarr = mmean_dst_root / grid_from(tile_under_test) / tile_name_from(tile_under_test) / mmean_zarr
    assert restructured_zarr.exists()
    restructured_ds = xr.open_zarr(restructured_zarr)
    assert_selected_correct_mmean(restructured_ds[PARAMS_KEY])
    assert_selected_correct_mask(restructured_ds[GROUND_TRUTH_KEY])


def assert_selected_correct_mmean(mmean: DataArray) -> None:
    for i in range(mmean.shape[0]):
        assert np.all(mmean[i] == i).values.item()
