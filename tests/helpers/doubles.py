import time
from collections import defaultdict

import numpy as np
import torch as th
from numpy._typing import NDArray

from rattlinbog.estimators.base import Estimator, Score, EstimateDescription, LogSink
from rattlinbog.estimators.nn_estimator import NNEstimator
from rattlinbog.th_extensions.utils.dataset_splitters import split_to_params_and_labels


# noinspection PyPep8Naming,PyAttributeOutsideInit
class AlwaysTrue(Estimator):
    def __init__(self):
        self.num_predictions = 0

    def predict(self, X: NDArray) -> NDArray:
        self.num_predictions += 1
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


class LogSpy(LogSink):
    def __init__(self):
        self.received_scalar_steps = defaultdict(list)
        self.received_scalars_steps = defaultdict(list)
        self.received_image_steps = defaultdict(list)
        self.received_scalars_names = dict()

    def add_scalar(self, tag, scalar_value, global_step=None):
        self.received_scalar_steps[tag].append(global_step)

    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None):
        self.received_scalars_steps[main_tag].append(global_step)
        self.received_scalars_names[main_tag] = set(tag_scalar_dict.keys())

    def add_image(self, tag, img_tensor, global_step=None):
        self.received_image_steps[tag].append(global_step)


class DelayingSplit:
    def __init__(self):
        self.loading_time = 0.0

    def set_loading_time(self, seconds):
        self.loading_time = seconds

    def __call__(self, *args, **kwargs):
        time.sleep(self.loading_time)
        return split_to_params_and_labels(*args, **kwargs)
