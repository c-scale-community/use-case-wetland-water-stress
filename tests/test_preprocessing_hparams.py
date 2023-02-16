import numpy as np
import pytest
import xarray as xr
from xarray import DataArray

from assertions import assert_arrays_identical
from factories import make_raster


def preprocess_hparams(da: DataArray) -> DataArray:
    c1 = da.sel(parameter='SIG0-HPAR-C1')
    s1 = da.sel(parameter='SIG0-HPAR-S1')
    da_w_amp_a_phs = xr.concat([da,
                                np.sqrt(c1 ** 2 + s1 ** 2).expand_dims(parameter=['SIG0-HPAR-AMP']),
                                np.arctan2(s1, c1).expand_dims(parameter=['SIG0-HPAR-PHS'])], dim='parameter')
    nobs = da_w_amp_a_phs.sel(parameter='SIG0-HPAR-NOBS')
    mean = da_w_amp_a_phs.weighted(nobs).mean(dim='orbit', skipna=True, keep_attrs=True)
    mean.loc['SIG0-HPAR-NOBS', ...] = nobs.sum('orbit', skipna=True, keep_attrs=True)
    return mean.assign_coords(spatial_ref=da.coords['spatial_ref'])


@pytest.fixture
def hparams_per_orbit_da():
    return xr.concat([
        make_raster([[[1.5]], [[-1.0]]], param_dim=('orbit', ['A044', 'D022'])).expand_dims(parameter=['SIG0-HPAR-C1']),
        make_raster([[[-20]], [[-10]]], param_dim=('orbit', ['A044', 'D022'])).expand_dims(parameter=['SIG0-HPAR-M0']),
        make_raster([[[120]], [[60]]], param_dim=('orbit', ['A044', 'D022'])).expand_dims(parameter=['SIG0-HPAR-NOBS']),
        make_raster([[[-0.3]], [[-0.6]]], param_dim=('orbit', ['A044', 'D022'])).expand_dims(
            parameter=['SIG0-HPAR-S1']),
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
        make_raster([[[(np.sqrt(2.34) * 120 + np.sqrt(1.36) * 60) / 180]]], param_dim=('parameter', ['SIG0-HPAR-AMP'])),
        make_raster([[[(np.arctan2(-0.3, 1.5) * 120 + np.arctan2(-0.6, -1.0) * 60) / 180]]],
                    param_dim=('parameter', ['SIG0-HPAR-PHS'])),
    ], dim='parameter')


def test_create_weighted_average_of_hparams_per_orbit_and_provide_phase_and_amplitude(
        hparams_per_orbit_da, hparams_processed_da):
    assert_arrays_identical(preprocess_hparams(hparams_per_orbit_da), hparams_processed_da)
