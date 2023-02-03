from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from xarray import Dataset


def split_to_params_and_labels(ds: Dataset) -> Tuple[NDArray, NDArray]:
    return ds['params'].values, ds['mask'].values[np.newaxis, ...]
