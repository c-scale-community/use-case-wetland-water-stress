from abc import abstractmethod
from dataclasses import dataclass
from typing import Sequence, Dict, Callable, Optional
from typing_extensions import Protocol

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

    @abstractmethod
    def score(self, X) -> float:
        ...

    @property
    @abstractmethod
    def out_description(self) -> EstimateDescription:
        ...


class LogSink(Protocol):
    @abstractmethod
    def add_scalar(self, tag, scalar_value, global_step=None):
        ...


@dataclass
class Validation:
    loss: float
    score: float


@dataclass
class ValidationConfig:
    frequency: int
    validator: Callable[[Estimator], Validation]
    log_sink: LogSink


@dataclass
class LogConfig:
    log_sink: LogSink
    validation: Optional[ValidationConfig] = None
