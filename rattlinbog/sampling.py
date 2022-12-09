from typing import Sequence, Optional

import numpy as np
import xarray as xr
from numpy.random import RandomState
from xarray import Dataset, DataArray


def sample_uniformly(ds: Dataset, sample_masks: Sequence[DataArray], n: int, seed: Optional[int] = None) -> Dataset:
    def sample(mask):
        valid_indices = np.asarray(np.nonzero(mask))
        selection = np.arange(valid_indices.shape[-1])
        rnd = RandomState(seed)
        rnd.shuffle(selection)
        valid_indices = valid_indices[:, selection[:n]]
        return ds.isel(y=DataArray(valid_indices[0], dims=['sampled']),
                       x=DataArray(valid_indices[1], dims=['sampled']))

    return xr.concat([sample(m) for m in sample_masks],
                     dim='samples',
                     combine_attrs='identical') \
        .assign_coords(samples=[m.name for m in sample_masks])


def make_histogram_masks(da: DataArray, quantiles: Sequence[float]) -> Sequence[np.ndarray]:
    quantiles = [-np.inf] + list(da.quantile(quantiles).values)
    return [np.logical_and(da.values > quantiles[i], da.values <= quantiles[i + 1]) for i in range(len(quantiles) - 1)]
