from typing import Iterable, Callable, Tuple, Iterator, Optional

import dask
from numpy._typing import NDArray
from torch.utils.data import IterableDataset
from xarray import Dataset


class StreamedXArrayDataset(IterableDataset):
    def __init__(self, xarray_source: Iterable[Dataset],
                 input_label_splitter: Callable[[Dataset], Tuple[NDArray, NDArray]],
                 estimated_len: Optional[int] = None):
        self._delayed_splits = [dask.delayed(input_label_splitter)(d) for d in xarray_source]
        self.estimated_len = estimated_len

    def __iter__(self) -> Iterator[Tuple[NDArray, NDArray]]:
        for ds in dask.compute(*self._delayed_splits, scheduler='threads'):
            yield ds
