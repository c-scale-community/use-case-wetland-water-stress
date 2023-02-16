from typing import Iterator, Tuple

import numpy as np
import pytest
import torch as th
import torch.cuda
from numpy.typing import NDArray
from sklearn import clone
from sklearn.utils.estimator_checks import check_estimator
from torch.nn import Sigmoid, MSELoss
from torch.optim.adam import Adam
from torch.utils.data import IterableDataset
from xarray import Dataset

from assertions import gather_all_exceptions
from doubles import NNEstimatorStub, LogSpy, NNEstimatorSpy
from factories import make_raster
from rattlinbog.estimators.apply import apply
from rattlinbog.estimators.base import Estimator, ValidationLogging, LogConfig, Validation, \
    ImageLogging
from rattlinbog.estimators.wetland_classifier import WetlandClassifier
from rattlinbog.evaluate.validator_of_dataset import ValidatorOfDataset
from rattlinbog.th_extensions.nn.unet import UNet
from rattlinbog.th_extensions.utils.dataset_splitters import PARAMS_KEY, GROUND_TRUTH_KEY


@pytest.fixture
def unet():
    return UNet(2, [4, 8], 1, out_activation=Sigmoid())


@pytest.fixture
def nn_estimator_params(unet):
    return dict(net=unet, batch_size=16, optim_factory=lambda p: Adam(p, lr=1e-2), loss_fn=MSELoss(), log_cfg=None)


@pytest.fixture
def nn_estimator(nn_estimator_params):
    return NNEstimatorStub(**nn_estimator_params)


@pytest.fixture
def nn_estimator_gpu(unet, nn_estimator_params):
    nn_estimator_params['net'] = nn_estimator_params['net'].to(device=th.device('cuda'))
    return NNEstimatorStub(**nn_estimator_params)


@pytest.fixture
def nn_estimator_logging(nn_estimator_params, log_sink):
    nn_estimator_params['log_cfg'] = LogConfig(log_sink)
    return NNEstimatorSpy(**nn_estimator_params)


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
    return make_log_sink()


def make_log_sink():
    return LogSpy()


@pytest.fixture
def validator(one_input, zero_output):
    return ValidatorOfDataset(Dataset({
        PARAMS_KEY: make_raster(one_input),
        GROUND_TRUTH_KEY: make_raster(zero_output[0])
    }).chunk())


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


def test_write_train_statistics_to_logging_facilities_if_provided(nn_estimator_logging, nn_estimator_params,
                                                                  generated_dataset, log_sink):
    nn_estimator_logging.fit(generated_dataset(10 * nn_estimator_params['batch_size']))
    assert log_sink.received_scalar_steps['loss'] == list(range(10))


def test_write_validation_score_statistics_to_logging_facilities_at_specified_frequency_if_provided(
        nn_estimator_params, nn_estimator_logging, generated_dataset):
    def validation_fn(model: Estimator) -> Validation:
        assert model is not None
        return Validation(loss=0.42, score={'VAL_SCORE': 0.21})

    train_sink = make_log_sink()
    valid_sink = make_log_sink()
    nn_estimator_logging.set_params(log_cfg=LogConfig(train_sink, ValidationLogging(2, valid_sink, validation_fn)))

    nn_estimator_logging.fit(generated_dataset(10 * nn_estimator_params['batch_size']))

    assert_received_log_at_correct_frequency(train_sink, valid_sink, n_training_steps=10, valid_freq=2)


def assert_received_log_at_correct_frequency(train_sink, valid_sink, n_training_steps, valid_freq):
    assert train_sink.received_scalar_steps['loss'] == list(range(n_training_steps))
    assert train_sink.received_scalar_steps['SCORE_A'] == list(range(0, n_training_steps, valid_freq))
    assert train_sink.received_scalar_steps['SCORE_B'] == list(range(0, n_training_steps, valid_freq))
    assert valid_sink.received_scalar_steps['loss'] == list(range(0, n_training_steps, valid_freq))
    assert valid_sink.received_scalar_steps['VAL_SCORE'] == list(range(0, n_training_steps, valid_freq))


def test_log_image_at_specified_frequency(nn_estimator_params, nn_estimator_logging, generated_dataset, log_sink):
    def image_fn(estimator: Estimator) -> NDArray:
        return np.zeros((1, 8, 8))

    nn_estimator_logging.set_params(log_cfg=LogConfig(log_sink, image=ImageLogging(5, log_sink, image_fn)))

    nn_estimator_logging.fit(generated_dataset(10 * nn_estimator_params['batch_size']))

    assert_received_images_at_correct_frequency(log_sink, n_training_steps=10, img_freq=5)


def assert_received_images_at_correct_frequency(log_sink, n_training_steps, img_freq):
    assert log_sink.received_image_steps['images'] == list(range(0, n_training_steps, img_freq))


def test_logging_sets_model_only_temporarily_in_eval_mode(
        nn_estimator_params, nn_estimator_logging, generated_dataset, log_sink, zero_output):
    nn_estimator_logging.set_params(log_cfg=LogConfig(log_sink,
                                                      ValidationLogging(2, log_sink, lambda _: Validation(0, {})),
                                                      ImageLogging(5, log_sink, lambda _: zero_output)))

    nn_estimator_logging.fit(generated_dataset(10 * nn_estimator_params['batch_size']))

    assert nn_estimator_logging.is_net_training_during_optimization_step == [True] * 10


def test_wetland_classification_estimator_protocol(wl_estimator, one_input, one_output, validator):
    assert wl_estimator.out_description.dims == {'class_probs': ['is_wetland']}
    assert wl_estimator.out_description.num_divisions == 2
    assert apply(wl_estimator).to(make_raster(one_input).chunk()).load().shape == (1, *one_input.shape[1:])
    assert_has_scores_as_scalars(wl_estimator.score(one_input, one_output), ['F1', 'BA', 'TPR', 'TNR', 'FPR', 'FNR'])
    assert_has_scores_as_scalars(validator(wl_estimator).score, ['F1', 'BA', 'TPR', 'TNR', 'FPR', 'FNR'])


def assert_has_scores_as_scalars(actual, expected_scores):
    for expected_score in expected_scores:
        assert np.isscalar(actual[expected_score])
