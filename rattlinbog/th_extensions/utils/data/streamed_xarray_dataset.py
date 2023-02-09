from typing import Iterable, Callable, Tuple, Iterator, Optional

import dask
from numpy._typing import NDArray
from toolz import partition_all
from torch.utils.data import IterableDataset
from xarray import Dataset


class StreamedXArrayDataset(IterableDataset):
    def __init__(self, xarray_source: Iterable[Dataset],
                 input_label_splitter: Callable[[Dataset], Tuple[NDArray, NDArray]],
                 stream_buffer: Optional[int] = 32,
                 estimated_len: Optional[int] = None):
        self._delayed_splits = map(dask.delayed(input_label_splitter), xarray_source)
        self._stream_buffer = stream_buffer
        self.estimated_len = estimated_len

    def __iter__(self) -> Iterator[Tuple[NDArray, NDArray]]:
        for batched_ds in map(lambda db: dask.compute(*db, scheduler='threads'),
                              partition_all(self._stream_buffer, self._delayed_splits)):
            for ds in batched_ds:
                yield ds
