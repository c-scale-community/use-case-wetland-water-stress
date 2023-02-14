from abc import abstractmethod
from dataclasses import dataclass
from typing import Sequence, Dict, Callable, Optional

from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from typing_extensions import Protocol

Coords = Sequence
DimsWithCoords = Dict[str, Coords]
Score = Dict[str, float]


@dataclass
class EstimateDescription:
    dims: DimsWithCoords
    num_divisions: int


# turn of inspections that collide with scikit-learn API requirements & style guide, see:
# https://scikit-learn.org/stable/developers/develop.html
# noinspection PyPep8Naming,PyAttributeOutsideInit
class Estimator(BaseEstimator):
    @abstractmethod
    def predict(self, X: NDArray) -> NDArray:
        ...

    @abstractmethod
    def score(self, X: NDArray, y: NDArray) -> Score:
        ...

    @property
    @abstractmethod
    def out_description(self) -> EstimateDescription:
        ...


class LogSink(Protocol):
    @abstractmethod
    def add_scalar(self, tag, scalar_value, global_step=None):
        ...

    @abstractmethod
    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None):
        ...

    @abstractmethod
    def add_image(self, tag, img_tensor, global_step=None):
        ...


@dataclass
class Validation:
    loss: float
    score: Score


@dataclass
class IntervalLogging:
    frequency: int
    log_sink: LogSink


@dataclass
class ValidationLogging(IntervalLogging):
    validator: Callable[[Estimator], Validation]


@dataclass
class ImageLogging(IntervalLogging):
    image_producer: Callable[[Estimator], NDArray]


@dataclass
class LogConfig:
    log_sink: LogSink
    validation: Optional[ValidationLogging] = None
    image: Optional[ImageLogging] = None
