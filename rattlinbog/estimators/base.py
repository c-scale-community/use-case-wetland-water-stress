from abc import abstractmethod
from dataclasses import dataclass
from typing import Sequence, Dict

from sklearn.base import BaseEstimator

Coords = Sequence
DimsWithCoords = Dict[str, Coords]


@dataclass
class EstimateDescription:
    dims: DimsWithCoords


# turn of inspections that collide with scikit-learn API requirements & style guide, see:
# https://scikit-learn.org/stable/developers/develop.html
# noinspection PyPep8Naming,PyAttributeOutsideInit
class Estimator(BaseEstimator):
    @abstractmethod
    def predict(self, X):
        ...

    @property
    @abstractmethod
    def out_description(self) -> EstimateDescription:
        ...
