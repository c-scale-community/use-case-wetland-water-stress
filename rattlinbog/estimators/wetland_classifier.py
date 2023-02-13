import numpy as np
from numpy.typing import NDArray
from sklearn.base import ClassifierMixin

from rattlinbog.estimators.base import EstimateDescription
from rattlinbog.estimators.nn_estimator import NNEstimator


def sigmoid(x: NDArray) -> NDArray:
    return 1 / (1 + np.exp(-x))


class WetlandClassifier(NNEstimator, ClassifierMixin):
    def predict(self, X: NDArray) -> NDArray:
        return sigmoid(super().predict(X))

    @property
    def out_description(self) -> EstimateDescription:
        return EstimateDescription({'class_probs': ['is_wetland']})
