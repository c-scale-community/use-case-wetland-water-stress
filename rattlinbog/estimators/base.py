from abc import abstractmethod
from typing import Sequence

from sklearn.base import ClassifierMixin


# turn of inspections that collide with scikit-learn API requirements & style guide, see:
# https://scikit-learn.org/stable/developers/develop.html
# noinspection PyPep8Naming,PyAttributeOutsideInit
class ClassEstimatorMixin(ClassifierMixin):
    @abstractmethod
    def predict(self, X):
        ...

    @property
    @abstractmethod
    def classes(self) -> Sequence[str]:
        ...
