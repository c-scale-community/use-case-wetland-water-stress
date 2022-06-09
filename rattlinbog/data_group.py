from abc import abstractmethod, ABC
from collections import defaultdict
from typing import MutableMapping, Sequence, TypeVar, Generic

from shapely.geometry import box
from xarray import Dataset

from rattlinbog.loaders import ROI

T = TypeVar("T")


class DataGroup(MutableMapping[str, Sequence[Dataset]]):
    def __init__(self, mapping: MutableMapping[str, Sequence[Dataset]]):
        self._mapping = mapping

    def __getitem__(self, k: str) -> Sequence[Dataset]:
        return self._mapping[k]

    def __setitem__(self, k: str, v: Sequence[Dataset]) -> None:
        self._mapping[k] = v

    def __delitem__(self, k: str) -> None:
        del self._mapping[k]

    def __len__(self) -> int:
        return len(self._mapping)

    def __iter__(self):
        return iter(self._mapping)


class Group(Generic[T]):
    @abstractmethod
    def get_name(self) -> str:
        ...

    @abstractmethod
    def add_meta_data_to(self, dataset: Dataset) -> None:
        ...

class GroupingRule(ABC):
    @abstractmethod
    def __call__(self, x: Dataset) -> Sequence[Group]:
        ...


class GroupByRois(GroupingRule):
    class GroupRoi(Group):
        def __init__(self, roi: ROI):
            self._roi = roi

        def get_name(self) -> str:
            return self._roi.name

        def add_meta_data_to(self, dataset: Dataset) -> None:
            dataset.attrs['roi'] = self._roi

    def __init__(self, rois: Sequence[ROI]):
        self._rois = rois

    def __call__(self, x: Dataset) -> Sequence[Group]:
        bounds = box(*x.rio.bounds())
        return [self.GroupRoi(r) for r in self._rois if bounds.intersects(r.geometry)]


def group_datasets(datasets: Sequence[Dataset], by_rule: GroupingRule) -> DataGroup:
    mapping = defaultdict(list)
    for ds in datasets:
        for group in by_rule(ds):
            group.add_meta_data_to(ds)
            mapping[group.get_name()].append(ds.copy())

    return DataGroup(mapping)
