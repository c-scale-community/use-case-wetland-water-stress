from abc import abstractmethod

import numpy as np
import xarray as xr

from rattlinbog.data_group import DataGroup

try:
    from typing import Protocol, Sequence
except ImportError:
    from typing_extensions import Protocol


class TransformDataGroup(Protocol):
    @abstractmethod
    def __call__(self, x: DataGroup) -> DataGroup:
        ...


class Compose(TransformDataGroup):
    def __init__(self, transformations: Sequence[TransformDataGroup]):
        self._transformations = transformations

    def __call__(self, x: DataGroup) -> DataGroup:
        for t in self._transformations:
            x = t(x)
        return x


class CoarsenAvgSpatially(TransformDataGroup):
    def __init__(self, stride: int):
        self._stride = stride

    def __call__(self, x: DataGroup) -> DataGroup:
        for k in x:
            x[k] = [ds.coarsen(x=self._stride, boundary='trim').mean().coarsen(y=self._stride, boundary='trim').mean()
                    .astype(np.float32)
                    for ds in x[k]]
        return x


class ClipRoi(TransformDataGroup):
    def __call__(self, x: DataGroup) -> DataGroup:
        for k in x:
            x[k] = [ds.rio.clip_box(*ds.attrs['roi'].geometry.bounds) for ds in x[k]]
        return x


class ConcatTimeSeries(TransformDataGroup):
    def __call__(self, x: DataGroup) -> DataGroup:
        for k in x:
            sorted_ds = sorted(x[k], key=lambda d: d.attrs['time'])
            times = [d.attrs['time'] for d in sorted_ds]
            x[k] = [xr.concat(sorted_ds, dim='time').assign_coords({'time': times})]
        return x


class ClipValues(TransformDataGroup):
    def __init__(self, vmin, vmax):
        self._vmin = vmin
        self._vmax = vmax

    def __call__(self, x: DataGroup) -> DataGroup:
        for k in x:
            x[k] = [ds.clip(self._vmin, self._vmax) for ds in x[k]]
        return x


class RoundToInt16(TransformDataGroup):
    def __call__(self, x: DataGroup) -> DataGroup:
        for k in x:
            x[k] = [ds.round().astype(np.int16) for ds in x[k]]
        return x
