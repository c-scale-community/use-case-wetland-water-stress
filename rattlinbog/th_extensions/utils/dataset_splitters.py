from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from xarray import Dataset

PARAMS_KEY = 'params'
GROUND_TRUTH_KEY = 'ground_truth'


def split_to_params_and_ground_truth(ds: Dataset) -> Tuple[NDArray, NDArray]:
    return ds[PARAMS_KEY].values, ds[GROUND_TRUTH_KEY].values[np.newaxis, ...]
