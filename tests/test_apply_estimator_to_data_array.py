import numpy as np
import pytest

from assertions import assert_arrays_identical
from doubles import AlwaysTrueEstimatorSpy, NNPredictorStub, MultiClassEstimator
from factories import make_raster
from rattlinbog.estimators.apply import apply
from rattlinbog.th_extensions.nn.unet import UNet


@pytest.fixture
def estimate_always_true():
    return AlwaysTrueEstimatorSpy()


@pytest.fixture
def model_with_3_divisions():
    return NNPredictorStub(UNet(1, [2, 4, 8], 1))


@pytest.fixture
def estimate_multiple_classes():
    return MultiClassEstimator()


def test_applying_classification_estimator_to_data_array_chunks(estimate_always_true):
    estimated = apply(estimate_always_true).to(make_raster(np.zeros((2, 4, 4))).chunk(2))
    out_dim, = tuple(estimate_always_true.out_description.dims.items())
    assert_arrays_identical(estimated.load(), make_raster(np.ones((1, 4, 4)), param_dim=out_dim))


def test_estimator_is_called_once_per_chunk(estimate_always_true):
    apply(estimate_always_true).to(make_raster(np.zeros((2, 4, 4))).chunk(2)).compute()
    assert estimate_always_true.num_predictions == 4


def test_pass_kwargs_to_estimator_predict_fn(estimate_always_true):
    apply(estimate_always_true, dict(param='some arg')).to(make_raster(np.zeros((2, 4, 4))).chunk(2)).compute()
    assert estimate_always_true.received_param == 'some arg'


def test_estimate_multiple_classes(estimate_multiple_classes):
    estimated = apply(estimate_multiple_classes).to(make_raster(np.zeros((2, 4, 4))).chunk(2)).compute()
    out_dims, = tuple(estimate_multiple_classes.out_description.dims.items())
    assert_arrays_identical(estimated.load(), make_raster(np.full((4, 4, 4), 0.25), param_dim=out_dims))


def test_handle_uneven_divisions_when_up_sampling(model_with_3_divisions):
    raster_with_even_size = make_raster(np.zeros((1, 40, 40), dtype=np.float32)).chunk({'y': 10, 'x': 10})
    estimate = apply(model_with_3_divisions).to(raster_with_even_size).compute()
    assert estimate.shape == (1, 40, 40)


def test_handle_uneven_padding_divisions_correctly(model_with_3_divisions):
    raster_with_even_size = make_raster(np.zeros((1, 10, 10), dtype=np.float32)).chunk({'y': 5, 'x': 5})
    estimate = apply(model_with_3_divisions).to(raster_with_even_size).compute()
    assert estimate.shape == (1, 10, 10)
