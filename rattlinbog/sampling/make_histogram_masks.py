from typing import Union, Sequence

import numpy as np
from xarray import DataArray

Scalars = Union[Sequence[float], Sequence[int]]


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
