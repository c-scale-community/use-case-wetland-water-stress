import pytest
import xarray as xr
from xarray import DataArray

from assertions import assert_arrays_identical
from factories import make_raster


def preprocess_hparams(da: DataArray) -> DataArray:
    nobs = da.sel(parameter='SIG0-HPAR-NOBS')
    mean = da.weighted(nobs).mean(dim='orbit', skipna=True, keep_attrs=True)
    mean.loc['SIG0-HPAR-NOBS', ...] = nobs.sum('orbit', skipna=True, keep_attrs=True)
    return mean.assign_coords(spatial_ref=da.coords['spatial_ref'])


@pytest.fixture
def hparams_per_orbit_da():
    return xr.concat([
        make_raster([[[1.5]], [[-1.0]]], param_dim=('orbit', ['A044', 'D022'])).expand_dims(parameter=['SIG0-HPAR-C1']),
        make_raster([[[-20]], [[-10]]], param_dim=('orbit', ['A044', 'D022'])).expand_dims(parameter=['SIG0-HPAR-M0']),
        make_raster([[[120]], [[60]]], param_dim=('orbit', ['A044', 'D022'])).expand_dims(parameter=['SIG0-HPAR-NOBS']),
        make_raster([[[-0.3]], [[-0.6]]], param_dim=('orbit', ['A044', 'D022'])).expand_dims(parameter=['SIG0-HPAR-S1']),
        make_raster([[[2.6]], [[1.2]]], param_dim=('orbit', ['A044', 'D022'])).expand_dims(parameter=['SIG0-HPAR-STD']),
    ], dim='parameter')


@pytest.fixture
def hparams_processed_da():
    return xr.concat([
        make_raster([[[2 / 3]]], param_dim=('parameter', ['SIG0-HPAR-C1'])),
        make_raster([[[-16 - (2 / 3)]]], param_dim=('parameter', ['SIG0-HPAR-M0'])),
        make_raster([[[180]]], param_dim=('parameter', ['SIG0-HPAR-NOBS'])),
        make_raster([[[-0.4]]], param_dim=('parameter', ['SIG0-HPAR-S1'])),
        make_raster([[[2 + (2 / 15)]]], param_dim=('parameter', ['SIG0-HPAR-STD'])),
    ], dim='parameter')


def test_create_weighted_average_of_hparams_per_orbit(
        hparams_per_orbit_da, hparams_processed_da):
    assert_arrays_identical(preprocess_hparams(hparams_per_orbit_da), hparams_processed_da)
