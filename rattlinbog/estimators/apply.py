from threading import Lock
from typing import Dict, Optional

import numpy as np
from xarray import DataArray

from rattlinbog.estimators.base import Estimator


class _DataArrayMapper:
    def __init__(self, estimator: Estimator, predictor_kwargs: Dict):
        self._estimator = estimator
        self._predictor_kwargs = predictor_kwargs
        self._estimator_lock = Lock()

    def to(self, array: DataArray) -> DataArray:
        out_dims = self._estimator.out_description.dims
        out_template = array[0, :, :].drop_vars(array.dims[0]).expand_dims(out_dims)
        estimated = array.data.map_blocks(self._estimate_block, drop_axis=0, new_axis=0, chunks=out_template.chunks,
                                          meta=out_template.data)
        return out_template.copy(data=estimated)

    def _estimate_block(self, block):
        n_divs = self._estimator.out_description.num_divisions
        y_size = block.shape[1]
        x_size = block.shape[2]
        y_padding = self._calc_to_even_padding(y_size, n_divs)
        x_padding = self._calc_to_even_padding(x_size, n_divs)
        return self._trunc_pad(self._safe_estimate(self._pad_to(block, y_padding, x_padding)),
                               y_padding, y_size, x_padding, x_size)

    def _safe_estimate(self, x):
        with self._estimator_lock:
            return self._estimator.predict(x, **self._predictor_kwargs)

    @staticmethod
    def _calc_to_even_padding(size, num_divisions):
        power_two_n = int(np.power(2, num_divisions))
        d = int(np.ceil(size / power_two_n))
        addition = (d * power_two_n) - size
        return (addition // 2), (addition // 2) + addition % 2

    @staticmethod
    def _pad_to(array, y_padding, x_padding):
        return np.pad(array, ((0, 0), y_padding, x_padding), mode='reflect')

    @staticmethod
    def _trunc_pad(array, y_padding, y_size, x_padding, x_size):
        return array[..., y_padding[0]:y_padding[0] + y_size, x_padding[0]:x_padding[0] + x_size]


def apply(estimator: Estimator, predict_kwargs: Optional[Dict] = None) -> _DataArrayMapper:
    return _DataArrayMapper(estimator, predict_kwargs or {})
