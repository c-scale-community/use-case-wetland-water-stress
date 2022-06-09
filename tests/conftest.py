from pathlib import Path
from typing import Optional

import pytest
import xarray as xr

from approvaltests import Options, verify_with_namer_and_writer, Writer
from approvaltests.core import Comparator
from approvaltests.utils import create_directory_if_needed
from xarray import Dataset

from rattlinbog.loaders import load_s1_datasets_from_file_list, load_rois
from rattlinbog.serialize import store_dataset


@pytest.fixture
def resource_path():
    return Path(__file__).parent / "resources"


@pytest.fixture
def path_to_file_list(resource_path):
    return resource_path / "s1-file-list.txt"


@pytest.fixture
def path_to_shape_file():
    return Path("/shared/ramsar/RAMSAR_AT_01.shp")


@pytest.fixture
def vh_datasets(path_to_file_list):
    return load_s1_datasets_from_file_list(path_to_file_list, {'VH'})


@pytest.fixture
def ramsar_rois(path_to_shape_file):
    return load_rois(path_to_shape_file)


class CompareNetCDFs(Comparator):
    def compare(self, received_path: str, approved_path: str) -> bool:
        if not Path(received_path).exists() or not Path(approved_path).exists():
            return False

        received_ds = xr.open_dataset(received_path)
        approved_ds = xr.open_dataset(approved_path)
        if received_ds.attrs != approved_ds.attrs:
            return False

        if set(received_ds.keys()) != set(approved_ds.keys()):
            return False

        for band in received_ds:
            received_a = received_ds[band]
            approved_a = approved_ds[band]
            if received_a.attrs != approved_a.attrs:
                return False
            if (received_a != approved_a).any():
                return False

        return True


class DatasetWriter(Writer):
    def __init__(self, ds: Dataset):
        self._ds = ds

    def write_received_file(self, received_file: str) -> str:
        create_directory_if_needed(received_file)
        store_dataset(Path(received_file), self._ds)
        return received_file


@pytest.fixture
def verify_dataset():
    def _verify_fn(dataset: Dataset,
                   *,  # enforce keyword arguments - https://www.python.org/dev/peps/pep-3102/
                   options: Optional[Options] = None) -> None:
        options = options or Options()
        # options = options.with_comparator(CompareNetCDFs())
        options = options.for_file.with_extension(".nc")

        verify_with_namer_and_writer(
            options.namer,
            DatasetWriter(dataset),
            options=options
        )

    return _verify_fn
