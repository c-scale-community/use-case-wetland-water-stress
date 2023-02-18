from abc import abstractmethod, ABC
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
    def predict(self, X: NDArray, **kwargs) -> NDArray:
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
    def add_image(self, tag, img_tensor, global_step=None):
        ...

    @abstractmethod
    def add_images(self, tag, img_tensor, global_step=None):
        ...


class ValidationSource(ABC):
    @property
    @abstractmethod
    def ground_truth(self) -> NDArray:
        ...

    @abstractmethod
    def make_estimation_using(self, model: Estimator, estimation_kwargs: Optional[Dict] = None) -> NDArray:
        ...


@dataclass
class ValidationLogging:
    log_sink: LogSink
    source: ValidationSource
    score_frequency: Optional[int] = None
    image_frequency: Optional[int] = None


@dataclass
class LogConfig:
    log_sink: LogSink
    validation: Optional[ValidationLogging] = None
