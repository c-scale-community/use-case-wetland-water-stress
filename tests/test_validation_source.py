import numpy as np
import pytest
from xarray import Dataset

from doubles import AlwaysTrueEstimatorSpy
from factories import make_raster
from rattlinbog.evaluate.validation_source_from_dataset import ValidationSourceFromDataset
from rattlinbog.th_extensions.utils.dataset_splitters import PARAMS_KEY, GROUND_TRUTH_KEY


@pytest.fixture
def validation_ds():
    return Dataset({PARAMS_KEY: (make_raster(np.ones((3, 32, 32)))),
                    GROUND_TRUTH_KEY: (make_raster(np.zeros((32, 32))))}).chunk()


@pytest.fixture
def validation_source(validation_ds):
    return ValidationSourceFromDataset(validation_ds)


@pytest.fixture
def validation_gt(validation_ds):
    return validation_ds[GROUND_TRUTH_KEY]


@pytest.fixture
def validation_params(validation_ds):
    return validation_ds[PARAMS_KEY]


@pytest.fixture
def estimator():
    return AlwaysTrueEstimatorSpy()


def test_produce_estimates_from_validation_dataset_using_estimator(validation_source, estimator, validation_params):
    estimate = validation_source.make_estimation_using(estimator, dict(param='an arg'))
    assert estimator.received_param == 'an arg'
    assert_arrays_eq(estimator.received_estimation_input,
                     np.pad(validation_params, int(validation_params.shape[1] * 0.1), mode='reflect'))
    assert np.all(estimate == 1)


def assert_arrays_eq(actual, expected):
    for a, e in zip(actual, expected):
        np.testing.assert_array_equal(a, e)
