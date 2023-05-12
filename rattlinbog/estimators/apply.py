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

        n_divs = self._estimator.out_description.num_divisions
        y_border = min(int(array.data.chunksize[1] * 0.1), 64)
        x_border = min(int(array.data.chunksize[2] * 0.1), 64)
        overlap_size_y = array.data.chunksize[1] + y_border * 2
        overlap_size_x = array.data.chunksize[2] + x_border * 2
        y_padding, y_excess = self._calc_to_even_padding(overlap_size_y, n_divs)
        x_padding, x_excess = self._calc_to_even_padding(overlap_size_x, n_divs)

        depth = {0: 0, 1: y_padding + y_border, 2: x_padding + x_border}
        out_chunks = tuple(out_template.data.chunksize[d] + p * 2 for d, p in depth.items())
        estimated = array.data.map_overlap(self._estimate_block, depth=depth, boundary='reflect', allow_rechunk=False,
                                           chunks=out_chunks, meta=out_template.data,
                                           y_excess=y_excess, x_excess=x_excess, out_dtype=out_template.dtype)
        return out_template.copy(data=estimated)

    def _estimate_block(self, block, y_excess, x_excess, out_dtype):
        if y_excess > 0 or x_excess > 0:
            out = np.empty((1, *block.shape[1:]), dtype=out_dtype)
            out[..., y_excess:, x_excess:] = self._safe_estimate(block[..., y_excess:, x_excess:])
            return out
        else:
            return self._safe_estimate(block)

    def _safe_estimate(self, x):
        with self._estimator_lock:
            return self._estimator.predict(x, **self._predictor_kwargs)

    @staticmethod
    def _calc_to_even_padding(size, num_divisions):
        power_two_n = int(np.power(2, num_divisions))
        d = int(np.ceil(size / power_two_n))
        addition = (d * power_two_n) - size
        excess = addition % 2
        return addition // 2 + excess, excess


def apply(estimator: Estimator, predict_kwargs: Optional[Dict] = None) -> _DataArrayMapper:
    return _DataArrayMapper(estimator, predict_kwargs or {})
