from dataclasses import asdict
from typing import Optional, Dict

import numpy as np
from numpy.typing import NDArray
from sklearn.base import ClassifierMixin
from sklearn.metrics import confusion_matrix
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam

from rattlinbog.estimators.base import EstimateDescription, LogConfig, Score
from rattlinbog.estimators.nn_estimator import NNEstimator
from rattlinbog.evaluate.classification import score_first_order, score_second_order, score_zero_order
from rattlinbog.th_extensions.nn.unet import UNet


def sigmoid(x: NDArray) -> NDArray:
    return 1 / (1 + np.exp(-x))


def select_from_dict(d: Dict, i: int) -> Dict:
    return {k: v[i] for k, v in d.items()}


class WetlandClassifier(NNEstimator, ClassifierMixin):

    def __init__(self, net: UNet, batch_size: int, log_cfg: Optional[LogConfig] = None):
        super().__init__(net, batch_size, lambda p: Adam(p), BCEWithLogitsLoss(), log_cfg)

    def predict(self, X: NDArray) -> NDArray:
        return sigmoid(super().predict(X))

    def score_estimate(self, estimates: NDArray, ground_truth: NDArray) -> Score:
        estimates = (estimates > 0.5).ravel()
        cm = confusion_matrix(ground_truth.ravel(), estimates)
        zro = score_zero_order(cm)
        fst = score_first_order(zro)
        snd = score_second_order(fst)
        return select_from_dict({**asdict(fst), **asdict(snd)}, -1)

    @property
    def out_description(self) -> EstimateDescription:
        return EstimateDescription({'class_probs': ['is_wetland']}, self.net.num_hidden_layers)
