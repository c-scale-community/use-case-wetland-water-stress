from typing import Sequence

import numpy as np
from xarray import DataArray


def make_quantile_masks(da: DataArray, quantiles: Sequence[float]) -> Sequence[DataArray]:
    qs = [-np.inf] + list(da.quantile(quantiles).values)
    return [DataArray(np.logical_and(da.values > qs[i], da.values <= qs[i + 1]),
                      dims=da.dims, coords=da.coords, name=f"quantile_{quantiles[i]}")
            for i in range(len(qs) - 1)]
