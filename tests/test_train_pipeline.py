import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy.sparse as sparse
import torch as th
from numpy.random import Generator
from scipy.ndimage import binary_dilation
from xarray import Dataset

from doubles import LogSpy
from factories import make_raster
from rattlinbog.estimators.wetland_classifier import WetlandClassifier
from rattlinbog.pipeline.factory_functions import make_validation_log_cfg
from rattlinbog.pipeline.train import train
from rattlinbog.sampling.sample_patches_from_dataset import SamplingConfig
from rattlinbog.th_extensions.nn.unet import UNet
from rattlinbog.th_extensions.utils.dataset_splitters import PARAMS_KEY, GROUND_TRUTH_KEY


@pytest.fixture
def train_ds(fixed_seed):
    return make_random_ml_dataset(fixed_seed)


def make_random_ml_dataset(seed=None):
    gt = make_ground_truth(seed)
    ps = make_features_matching_gt(gt, np.random.default_rng(seed))
    return Dataset({
        PARAMS_KEY: ps,
        GROUND_TRUTH_KEY: gt.astype(np.float32)
    }).chunk({'parameters': 3, 'y': 50, 'x': 50})


def make_ground_truth(rnd_seed=None):
    rnd_m = sparse.rand(500, 500, 0.0001, random_state=rnd_seed)
    rnd_m.data[:] = 1
    gt = rnd_m.astype('uint8').todense()
    return make_raster(binary_dilation(gt, np.ones((30, 30)), iterations=1).astype(np.uint8))


def make_features_matching_gt(ground_truth, rnd_state=None):
    shape = ground_truth.shape
    bg = np.empty((3, *shape), dtype=np.float32)
    bg[0, ...] = rnd_state.normal(0.2, 0.01, shape)
    bg[1, ...] = rnd_state.normal(0.6, 0.01, shape)
    bg[2, ...] = rnd_state.normal(0.4, 0.01, shape)

    fg = np.empty((3, *shape), dtype=np.float32)
    fg[0, ...] = rnd_state.normal(0.6, 0.1, shape)
    fg[1, ...] = rnd_state.normal(0.2, 0.1, shape)
    fg[2, ...] = rnd_state.normal(0.3, 0.1, shape)

    gt_mask = ground_truth == 1
    bg[0, gt_mask] = fg[0, gt_mask]
    bg[1, gt_mask] = fg[1, gt_mask]
    bg[2, gt_mask] = fg[2, gt_mask]
    return make_raster(bg)


@pytest.fixture
def valid_ds(fixed_seed):
    return make_random_ml_dataset(fixed_seed)


@pytest.fixture
def train_log():
    return LogSpy()


@pytest.fixture
def valid_log():
    return LogSpy()


@pytest.fixture
def log_cfg(valid_ds, train_log, valid_log):
    return make_validation_log_cfg(valid_ds, train_log, valid_log, 10, 99)


@pytest.fixture
def estimator(log_cfg):
    unet = UNet(3, [16, 32], 1).to(device=th.device('cuda'))
    return WetlandClassifier(unet, 16, log_cfg)


@pytest.fixture
def sampling_cfg(estimator):
    return SamplingConfig(patch_size=16, n_samples=200 * estimator.batch_size, never_nans=True)


@pytest.mark.skipif(not th.cuda.is_available(), reason='this test needs a cuda device')
def test_train_wetland_estimator(estimator, train_ds, sampling_cfg, fixed_seed, valid_log, train_log, valid_ds,
                                 should_plot):
    trained_model = train(estimator, train_ds, sampling_cfg, np.random.default_rng(fixed_seed))

    if should_plot:
        _plot_results(trained_model, train_ds, train_log, valid_ds, valid_log)

    assert trained_model.is_fitted_
    assert valid_log.received_last_score['F1'] > 0.95


def _plot_results(trained_model, train_ds, train_log, valid_ds, valid_log):
    _, axes = plt.subplots(ncols=3, nrows=1, figsize=(10, 5))
    axes[0].imshow(valid_ds[GROUND_TRUTH_KEY])
    axes[1].imshow(valid_log.received_last_image['images'][0])
    axes[2].imshow(trained_model.predict(train_ds[PARAMS_KEY].values)[0])
    plt.show()
    plt.plot(train_log.received_loss)
    plt.show()
