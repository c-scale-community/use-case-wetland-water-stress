from typing import Iterator, Tuple

import numpy as np
import pytest
from approval_utilities.utilities.exceptions.multiple_exceptions import MultipleExceptions
from numpy.typing import NDArray
from sklearn import clone
from sklearn.utils.estimator_checks import check_estimator
from torch.nn import Sigmoid, MSELoss
from torch.optim.adam import Adam
from torch.utils.data import IterableDataset

from rattlinbog.estimators.nn_regression import NNEstimator
from rattlinbog.th_extensions.nn.unet import UNet


@pytest.fixture
def unet():
    return UNet(2, [4, 8], 1, out_activation=Sigmoid())


@pytest.fixture
def nn_regression_params(unet):
    return dict(net=unet, batch_size=16, optim_factory=lambda p: Adam(p, lr=1e-2), loss_fn=MSELoss())


@pytest.fixture
def nn_regression(unet, nn_regression_params):
    return NNEstimator(**nn_regression_params)


@pytest.fixture
def one_input():
    return np.ones((2, 8, 8), dtype=np.float32)


@pytest.fixture
def zero_input():
    return np.zeros((2, 8, 8), dtype=np.float32)


@pytest.fixture
def one_output():
    return np.ones((1, 8, 8), dtype=np.float32)


@pytest.fixture
def zero_output():
    return np.zeros((1, 8, 8), dtype=np.float32)


@pytest.fixture
def generated_dataset(one_input, zero_input, one_output, zero_output):
    class _GeneratedDataset(IterableDataset):
        def __init__(self, size):
            self.size = size

        def __iter__(self) -> Iterator[Tuple[NDArray, NDArray]]:
            def generate_data(size: int):
                for _ in range(size):
                    if np.random.randint(2) == 0:
                        yield one_input, zero_output
                    else:
                        yield zero_input, one_output

            return iter(generate_data(self.size))

    return _GeneratedDataset


def test_unet_regression_is_sklearn_compatible(nn_regression, nn_regression_params):
    assert nn_regression.get_params() == nn_regression_params
    nn_regression.set_params(batch_size=4)

    new_params = nn_regression_params.copy()
    new_params['batch_size'] = 4
    assert nn_regression.get_params() == new_params
    check_estimator(clone(nn_regression))


def test_fit_unet_regression_label_data(nn_regression, generated_dataset, one_input, zero_input, one_output,
                                        zero_output, fixed_seed):
    gather_all_exceptions([(one_input, zero_input), (zero_input, one_output)],
                          lambda p: assert_prediction_ne(nn_regression.predict(p[0]), p[1])).assert_any_is_true()

    nn_regression.fit(generated_dataset(1000))

    assert_prediction_eq(nn_regression.predict(one_input), zero_output)
    assert_prediction_eq(nn_regression.predict(zero_input), one_output)


def gather_all_exceptions(params, code_to_execute):
    class _Collector:
        def __init__(self):
            self.exceptions = []

        def add(self, exception):
            self.exceptions.append(exception)

        def assert_any_is_true(self):
            if len(params) == len(self.exceptions):
                raise MultipleExceptions(self.exceptions)

    collector = _Collector()
    for p in params:
        try:
            code_to_execute(p)
        except Exception as e:
            collector.add(e)

    return collector


def assert_prediction_ne(actual: NDArray, expected: NDArray) -> None:
    assert np.any(np.not_equal(np.around(actual), expected))


def assert_prediction_eq(actual: NDArray, expected: NDArray) -> None:
    np.testing.assert_array_equal(np.around(actual), expected)
