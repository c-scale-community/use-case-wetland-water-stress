import numpy as np
import pytest
import xarray as xr

from assertions import assert_arrays_identical
from factories import make_raster
from rattlinbog.preprocessing import preprocess_hparams


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
