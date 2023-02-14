import numpy as np
import pytest
from numpy._typing import NDArray
from xarray import Dataset, DataArray

from doubles import ScoreableEstimatorSpy
from factories import make_raster
from rattlinbog.estimators.apply import apply
from rattlinbog.estimators.base import Estimator
from rattlinbog.evaluate.validator_of_dataset import ValidatorOfDataset


@pytest.fixture
def validation_ds():
    return Dataset({'params': (make_raster(np.ones((3, 32, 32)))),
                    'ground_truth': (make_raster(np.zeros((1, 32, 32)), param_dim=('class', ['yes'])))}).chunk()


@pytest.fixture
def validation_da(validation_ds):
    return validation_ds['params']


@pytest.fixture
def estimator():
    return ScoreableEstimatorSpy()


def test_validate_on_given_dataset(validation_ds, estimator):
    validate = ValidatorOfDataset(validation_ds)
    validation = validate(estimator)
    assert_arrays_eq(estimator.scorer_received, (estimator.returned_estimate, validation_ds['ground_truth'].values))
    assert validation.loss == estimator.returned_loss and validation.score == estimator.returned_score


def assert_arrays_eq(actual, expected):
    for a, e in zip(actual, expected):
        np.testing.assert_array_equal(a, e)


class ImageProducerFromDataArray:
    def __init__(self, validation_da: DataArray):
        self.validation_da = validation_da

    def __call__(self, estimator: Estimator) -> NDArray:
        return apply(estimator).to(self.validation_da).compute().values


def test_produce_image_from_given_data_array(validation_da, estimator):
    imager = ImageProducerFromDataArray(validation_da)
    assert_array_eq(imager(estimator), estimator.returned_estimate)


def assert_array_eq(actual, expected):
    np.testing.assert_array_equal(actual, expected)
