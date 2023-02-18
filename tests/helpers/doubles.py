import time
from collections import defaultdict
from typing import Optional

import numpy as np
import torch as th
from numpy._typing import NDArray

from rattlinbog.estimators.base import Estimator, Score, EstimateDescription, LogSink, ScoreableEstimator
from rattlinbog.estimators.nn_estimator import NNEstimator
from rattlinbog.th_extensions.utils.dataset_splitters import split_to_params_and_ground_truth


# noinspection PyPep8Naming,PyAttributeOutsideInit
class AlwaysTrue(Estimator):
    def __init__(self):
        self.num_predictions = 0
        self.received_param = None

    def predict(self, X: NDArray, param: Optional[str] = None) -> NDArray:
        self.num_predictions += 1
        self.received_param = param
        return np.ones((1,) + X.shape[1:])

    def score(self, X: NDArray, y: NDArray) -> Score:
        raise NotImplementedError

    @property
    def out_description(self) -> EstimateDescription:
        return EstimateDescription({'estimate': ['yes']}, 0)


# noinspection PyPep8Naming,PyAttributeOutsideInit
class NNPredictorStub(Estimator):
    def __init__(self, net):
        self.net = net

    def predict(self, X: NDArray) -> NDArray:
        return self.net(th.from_numpy(X).unsqueeze(0)).squeeze(0).detach().numpy()

    def score(self, X: NDArray, y: NDArray) -> Score:
        raise NotImplementedError

    @property
    def out_description(self) -> EstimateDescription:
        return EstimateDescription({'nn_out': ['garbage']}, 3)


# noinspection PyPep8Naming,PyAttributeOutsideInit
class MultiClassEstimator(Estimator):
    def predict(self, X: NDArray) -> NDArray:
        return np.full((4,) + X.shape[1:], 0.25)

    @property
    def out_description(self) -> EstimateDescription:
        return EstimateDescription({'classes': ["a", "b", "c", "d"]}, 0)


class NNEstimatorStub(NNEstimator):
    def score_estimate(self, estimate: NDArray, ground_truth: NDArray) -> Score:
        return {'SCORE_A': 0.42, 'SCORE_B': 42}

    @property
    def out_description(self) -> EstimateDescription:
        return EstimateDescription({'class_prob': ['is_class']}, 0)


class NNEstimatorSpy(NNEstimatorStub):
    def __init__(self, net, batch_size, optim_factory, loss_fn, log_cfg=None):
        super().__init__(net, batch_size, optim_factory, loss_fn, log_cfg)
        self.is_net_training_during_optimization_step = []

    def _optimization_step(self, optimizer, x_batch, y_batch, model_device):
        self.is_net_training_during_optimization_step.append(self.net.training)
        return super()._optimization_step(optimizer, x_batch, y_batch, model_device)


class LogSpy(LogSink):
    def __init__(self):
        self.received_scalar_steps = defaultdict(list)
        self.received_image_steps = defaultdict(list)
        self.received_images_steps = defaultdict(list)
        self.received_last_score = dict()
        self.received_last_image = dict()
        self.received_last_images = dict()
        self.received_loss = []

    def add_scalar(self, tag, scalar_value, global_step=None):
        self.received_scalar_steps[tag].append(global_step)
        self.received_last_score[tag] = scalar_value
        if tag == 'loss':
            self.received_loss.append(scalar_value)

    def add_image(self, tag, img_tensor, global_step=None):
        self.received_image_steps[tag].append(global_step)
        self.received_last_image[tag] = img_tensor

    def add_images(self, tag, img_tensor, global_step=None):
        self.received_images_steps[tag].append(global_step)
        self.received_last_images[tag] = img_tensor


class DelayingSplit:
    def __init__(self):
        self.loading_time = 0.0

    def set_loading_time(self, seconds):
        self.loading_time = seconds

    def __call__(self, *args, **kwargs):
        time.sleep(self.loading_time)
        return split_to_params_and_ground_truth(*args, **kwargs)


class ScoreableEstimatorSpy(ScoreableEstimator):
    def __init__(self):
        self.returned_raw_estimate = np.zeros((1, 32, 32))
        self.returned_refined_estimate = np.ones((1, 32, 32))
        self.returned_loss = 0.042
        self.returned_score = {'A': 42, 'B': 0.42}
        self.loss_received = None
        self.scorer_received = None

    def predict(self, X: NDArray, **kwargs) -> NDArray:
        return self.returned_raw_estimate

    def refine_raw_estimate(self, estimate: NDArray) -> NDArray:
        return self.returned_refined_estimate

    def loss_for_estimate(self, estimate: NDArray, ground_truth: NDArray) -> float:
        self.loss_received = (estimate, ground_truth)
        return self.returned_loss

    def score_estimate(self, estimate: NDArray, ground_truth: NDArray) -> Score:
        self.scorer_received = (estimate, ground_truth)
        return self.returned_score

    @property
    def out_description(self) -> EstimateDescription:
        return EstimateDescription({'classes': ['yes']}, 0)
