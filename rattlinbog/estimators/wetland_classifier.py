from typing import Sequence

from rattlinbog.estimators.base import ClassEstimatorMixin
from rattlinbog.estimators.nn_estimator import NNEstimator


class WetlandClassifier(NNEstimator, ClassEstimatorMixin):
    @property
    def classes(self) -> Sequence[str]:
        return ['is_wetland']