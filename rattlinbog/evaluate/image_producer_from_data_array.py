from numpy._typing import NDArray
from xarray import DataArray

from rattlinbog.estimators.apply import apply
from rattlinbog.estimators.base import Estimator


class ImageProducerFromDataArray:
    def __init__(self, validation_da: DataArray):
        self.validation_da = validation_da

    def __call__(self, estimator: Estimator) -> NDArray:
        return apply(estimator).to(self.validation_da).compute().values
