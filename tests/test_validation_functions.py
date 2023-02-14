import numpy as np
import pytest
from numpy._typing import NDArray
from xarray import Dataset

from factories import make_raster
from rattlinbog.estimators.apply import apply
from rattlinbog.estimators.base import Estimator, Validation, EstimateDescription, Score


class ValidatorOfDataset:
    def __init__(self, validation_ds: Dataset):
        self._validation_ds = validation_ds

    def __call__(self, estimator: Estimator) -> Validation:
        estimate = apply(estimator).to(self._validation_ds['params']).compute()
        ground_truth = self._validation_ds['ground_truth'].load()
        loss = estimator.loss_for_estimate(estimate.values, ground_truth.values)
        scores = estimator.score_estimate(estimate.values, ground_truth.values)
        return Validation(loss, scores)


@pytest.fixture
def validation_ds():
    return Dataset({'params': (make_raster(np.ones((3, 32, 32)))),
                    'ground_truth': (make_raster(np.zeros((1, 32, 32)), param_dim=('class', ['yes'])))}).chunk()


class EstimatorSpy(Estimator):
    def __init__(self):
        self.returned_estimate = np.zeros((1, 32, 32))
        self.returned_loss = 0.042
        self.returned_score = {'A': 42, 'B': 0.42}
        self.scorer_received = None

    def predict(self, X: NDArray) -> NDArray:
        return self.returned_estimate

    def loss_for_estimate(self, estimate: NDArray, ground_truth: NDArray) -> float:
        return self.returned_loss

    def score_estimate(self, estimate: NDArray, ground_truth: NDArray) -> Score:
        self.scorer_received = (estimate, ground_truth)
        return self.returned_score

    @property
    def out_description(self) -> EstimateDescription:
        return EstimateDescription({'classes': ['yes']}, 0)


@pytest.fixture
def estimator():
    return EstimatorSpy()


def test_validate_on_given_data_array(validation_ds, estimator):
    validate = ValidatorOfDataset(validation_ds)
    validation = validate(estimator)
    assert_arrays_eq(estimator.scorer_received, (estimator.returned_estimate, validation_ds['ground_truth'].values))
    assert validation.loss == estimator.returned_loss and validation.score == estimator.returned_score


def assert_arrays_eq(actual, expected):
    for a, e in zip(actual, expected):
        np.testing.assert_array_equal(a, e)
