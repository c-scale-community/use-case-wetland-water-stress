from typing import Optional, Dict

from numpy._typing import NDArray
from xarray import Dataset

from rattlinbog.estimators.apply import apply
from rattlinbog.estimators.base import ValidationSource, Estimator
from rattlinbog.th_extensions.utils.dataset_splitters import GROUND_TRUTH_KEY, PARAMS_KEY


class ValidationSourceFromDataset(ValidationSource):
    def __init__(self, ds: Dataset):
        self.ds = ds

    @property
    def ground_truth(self) -> NDArray:
        return self.ds[GROUND_TRUTH_KEY].values

    def make_estimation_using(self, model: Estimator, estimation_kwargs: Optional[Dict] = None) -> NDArray:
        return apply(model, estimation_kwargs).to(self.ds[PARAMS_KEY]).compute().values
