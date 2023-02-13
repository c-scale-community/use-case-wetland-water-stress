from typing import Sequence

import numpy as np
import pytest
import xarray as xr
from numpy.typing import NDArray

from factories import make_raster
from rattlinbog.estimators.apply import apply
from rattlinbog.estimators.base import Estimator, EstimateDescription


# noinspection PyPep8Naming,PyAttributeOutsideInit
class AlwaysTrue(Estimator):
    def __init__(self):
        self.num_predictions = 0

    def predict(self, X: NDArray) -> NDArray:
        self.num_predictions += 1
        return np.ones((1,) + X.shape[1:])

    @property
    def out_description(self) -> EstimateDescription:
        return EstimateDescription({'estimate': ['yes']})


@pytest.fixture
def estimate_always_true():
    return AlwaysTrue()


# noinspection PyPep8Naming,PyAttributeOutsideInit
class MultiClassEstimator(Estimator):
    def predict(self, X: NDArray) -> NDArray:
        return np.full((4,) + X.shape[1:], 0.25)

    @property
    def out_description(self) -> EstimateDescription:
        return EstimateDescription({'classes': ["a", "b", "c", "d"]})


@pytest.fixture
def estimate_multiple_classes():
    return MultiClassEstimator()


def test_applying_classification_estimator_to_data_array_chunks(estimate_always_true):
    estimated = apply(estimate_always_true).to(make_raster(np.zeros((2, 4, 4))).chunk(2))
    out_dim, = tuple(estimate_always_true.out_description.dims.items())
    assert_arrays_identical(estimated.load(), make_raster(np.ones((1, 4, 4)), param_dim=out_dim))


def assert_arrays_identical(actual, expected):
    xr.testing.assert_identical(actual, expected)


def test_estimator_is_called_once_per_chunk(estimate_always_true):
    apply(estimate_always_true).to(make_raster(np.zeros((2, 4, 4))).chunk(2)).compute()
    assert estimate_always_true.num_predictions == 4


def test_estimate_multiple_classes(estimate_multiple_classes):
    estimated = apply(estimate_multiple_classes).to(make_raster(np.zeros((2, 4, 4))).chunk(2)).compute()
    out_dims, = tuple(estimate_multiple_classes.out_description.dims.items())
    assert_arrays_identical(estimated.load(), make_raster(np.full((4, 4, 4), 0.25),
                                                          param_dim=out_dims))