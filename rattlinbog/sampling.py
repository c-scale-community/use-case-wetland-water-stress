from typing import Sequence, Optional, Union

import numpy as np
import xarray as xr
from numpy.random import RandomState
from xarray import Dataset, DataArray

Scalars = Union[Sequence[float], Sequence[int]]


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


def make_quantile_masks(da: DataArray, quantiles: Sequence[float]) -> Sequence[DataArray]:
    qs = [-np.inf] + list(da.quantile(quantiles).values)
    return [DataArray(np.logical_and(da.values > qs[i], da.values <= qs[i + 1]),
                      dims=da.dims, coords=da.coords, name=f"quantile_{quantiles[i]}")
            for i in range(len(qs) - 1)]


def make_histogram_masks(da: DataArray, bins: Union[int, Scalars, str]) -> Sequence[DataArray]:
    def make_mask(lower, upper, is_last):
        return DataArray(np.logical_and(da.values >= lower,
                                        # all but the right-most are half open - see note:
                                        # https://numpy.org/doc/stable/reference/generated/numpy.histogram.html
                                        da.values <= upper if is_last else da.values < upper),
                         dims=da.dims, coords=da.coords,
                         name=f"bin_[{lower}, {upper}{']' if is_last else ')'}")

    edges = np.histogram_bin_edges(da.values.flatten(), bins=bins)
    return [make_mask(edges[i], edges[i + 1], i == len(edges) - 2) for i in range(len(edges) - 1)]
