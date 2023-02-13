from typing import Callable, Any, Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.base import ClassifierMixin
from torch.nn import Module, BCEWithLogitsLoss
from torch.optim import Optimizer, Adam

from rattlinbog.estimators.base import EstimateDescription
from rattlinbog.estimators.nn_estimator import NNEstimator, ModelParams, LogConfig


def sigmoid(x: NDArray) -> NDArray:
    return 1 / (1 + np.exp(-x))


class WetlandClassifier(NNEstimator, ClassifierMixin):

    def __init__(self, net: Module, batch_size: int, log_cfg: Optional[LogConfig] = None):
        super().__init__(net, batch_size, lambda p: Adam(p), BCEWithLogitsLoss(), log_cfg)

    def predict(self, X: NDArray) -> NDArray:
        return sigmoid(super().predict(X))

    @property
    def out_description(self) -> EstimateDescription:
        return EstimateDescription({'class_probs': ['is_wetland']})
