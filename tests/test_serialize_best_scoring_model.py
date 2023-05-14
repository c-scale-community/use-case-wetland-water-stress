import numpy as np
import pytest
from torch.nn import MSELoss, Sigmoid
from torch.optim import Adam

from doubles import NNEstimatorStub
from rattlinbog.persist.serialize_best_scoring_nn_model import SerializeBestScoringNNModel
from rattlinbog.th_extensions.nn.unet import UNet


@pytest.fixture
def nn_estimator_additional_params():
    return dict(batch_size=16, optim_factory=lambda p: Adam(p, lr=1e-2), loss_fn=MSELoss(), log_cfg=None)


@pytest.fixture
def make_estimator(nn_estimator_additional_params):
    def _fn():
        return NNEstimatorStub(net=UNet(2, [4, 8], 1, out_activation=Sigmoid()), **nn_estimator_additional_params)

    return _fn


@pytest.fixture
def out_path(tmp_path):
    return tmp_path


def test_serialize_best_scoring_model(make_estimator, nn_estimator_additional_params, out_path):
    estimator_a = make_estimator()
    estimator_b = make_estimator()

    serialize_best_f1 = SerializeBestScoringNNModel(out_path, score='F1')

    serialize_best_f1.snapshot(estimator_a, {'F1': 0.6})
    serialize_best_f1.snapshot(estimator_b, {'F1': 0.5})

    loaded = NNEstimatorStub.from_snapshot(out_path / "NNEstimatorStub-F1-best.pt", **nn_estimator_additional_params)

    x = np.random.randn(2, 16, 16).astype(np.float32)
    assert_estimates_eq(loaded.predict(x), estimator_a.predict(x))
    assert_estimates_ne(loaded.predict(x), estimator_b.predict(x))


def assert_estimates_ne(actual, expected):
    assert np.any(np.not_equal(actual, expected))


def assert_estimates_eq(actual, expected):
    np.testing.assert_array_equal(actual, expected)
