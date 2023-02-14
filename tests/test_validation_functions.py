import numpy as np
import pytest
from xarray import Dataset

from doubles import ScoreableEstimatorSpy
from factories import make_raster
from rattlinbog.evaluate.validator_of_dataset import ValidatorOfDataset


@pytest.fixture
def validation_ds():
    return Dataset({'params': (make_raster(np.ones((3, 32, 32)))),
                    'ground_truth': (make_raster(np.zeros((1, 32, 32)), param_dim=('class', ['yes'])))}).chunk()


@pytest.fixture
def estimator():
    return ScoreableEstimatorSpy()


def test_validate_on_given_data_array(validation_ds, estimator):
    validate = ValidatorOfDataset(validation_ds)
    validation = validate(estimator)
    assert_arrays_eq(estimator.scorer_received, (estimator.returned_estimate, validation_ds['ground_truth'].values))
    assert validation.loss == estimator.returned_loss and validation.score == estimator.returned_score


def assert_arrays_eq(actual, expected):
    for a, e in zip(actual, expected):
        np.testing.assert_array_equal(a, e)
