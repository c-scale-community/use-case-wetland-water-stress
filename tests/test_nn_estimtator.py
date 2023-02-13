from collections import defaultdict
from typing import Iterator, Tuple

import numpy as np
import pytest
import torch as th
import torch.cuda
from approval_utilities.utilities.exceptions.multiple_exceptions import MultipleExceptions
from numpy.typing import NDArray
from sklearn import clone
from sklearn.utils.estimator_checks import check_estimator
from torch.nn import Sigmoid, MSELoss
from torch.optim.adam import Adam
from torch.utils.data import IterableDataset

from factories import make_raster
from rattlinbog.estimators.apply import apply
from rattlinbog.estimators.nn_estimator import NNEstimator, LogSink, LogConfig
from rattlinbog.estimators.wetland_classifier import WetlandClassifier
from rattlinbog.th_extensions.nn.unet import UNet


@pytest.fixture
def unet():
    return UNet(2, [4, 8], 1, out_activation=Sigmoid())


@pytest.fixture
def nn_estimator_params(unet):
    return dict(net=unet, batch_size=16, optim_factory=lambda p: Adam(p, lr=1e-2), loss_fn=MSELoss(), log_cfg=None)


@pytest.fixture
def nn_estimator(nn_estimator_params):
    return NNEstimator(**nn_estimator_params)


@pytest.fixture
def nn_estimator_gpu(unet, nn_estimator_params):
    nn_estimator_params['net'] = nn_estimator_params['net'].to(device=th.device('cuda'))
    return NNEstimator(**nn_estimator_params)


@pytest.fixture
def wl_estimator(unet):
    return WetlandClassifier(unet, 16)


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


@pytest.fixture
def log_sink():
    class _LogSpy(LogSink):
        def __init__(self):
            self.received_steps = defaultdict(list)

        def add_scalar(self, tag, scalar_value, global_step=None):
            self.received_steps[tag].append(global_step)

    return _LogSpy()


def test_unet_estimator_is_sklearn_compatible(nn_estimator, nn_estimator_params):
    assert nn_estimator.get_params() == nn_estimator_params
    nn_estimator.set_params(batch_size=4)

    new_params = nn_estimator_params.copy()
    new_params['batch_size'] = 4
    assert nn_estimator.get_params() == new_params
    check_estimator(clone(nn_estimator))


def test_fit_unet_estimator_to_label_data(nn_estimator, generated_dataset, one_input, zero_input, one_output,
                                          zero_output, fixed_seed):
    gather_all_exceptions([(one_input, zero_input), (zero_input, one_output)],
                          lambda p: assert_prediction_ne(nn_estimator.predict(p[0]), p[1])).assert_any_is_true()

    nn_estimator.fit(generated_dataset(1000))

    assert_prediction_eq(nn_estimator.predict(one_input), zero_output)
    assert_prediction_eq(nn_estimator.predict(zero_input), one_output)


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason='this test needs a cuda device')
def test_nn_estimator_automagically_moves_input_data_to_device_of_nn(nn_estimator_gpu, generated_dataset, one_input,
                                                                     zero_input, one_output, zero_output, fixed_seed):
    nn_estimator_gpu.fit(generated_dataset(1000))

    assert_prediction_eq(nn_estimator_gpu.predict(one_input), zero_output)
    assert_prediction_eq(nn_estimator_gpu.predict(zero_input), one_output)


def test_write_train_statistics_to_logging_facilities_if_provided(nn_estimator_params, generated_dataset, log_sink,
                                                                  fixed_seed):
    nn_estimator_params['log_cfg'] = LogConfig(log_sink)
    estimator = NNEstimator(**nn_estimator_params)
    estimator.fit(generated_dataset(10 * nn_estimator_params['batch_size']))
    assert log_sink.received_steps['loss'] == list(range(10))


def test_wetland_classification_estimator_protocol(wl_estimator, one_input):
    assert wl_estimator.out_description.dims == {'class_probs': ['is_wetland']}
    assert apply(wl_estimator).to(make_raster(one_input).chunk()).load().shape == (1, *one_input.shape[1:])
