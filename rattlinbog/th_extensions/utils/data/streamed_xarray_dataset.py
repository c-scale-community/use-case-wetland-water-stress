from typing import Iterable, Callable, Tuple, Iterator

from numpy._typing import NDArray
from torch.utils.data import IterableDataset
from xarray import Dataset


class StreamedXArrayDataset(IterableDataset):
    def __init__(self, xarray_source: Iterable[Dataset],
                 input_label_splitter: Callable[[Dataset], Tuple[NDArray, NDArray]]):
        self._source = xarray_source
        self._splitter = input_label_splitter

    def __iter__(self) -> Iterator[Tuple[NDArray, NDArray]]:
        return iter(map(self._splitter, self._source))
