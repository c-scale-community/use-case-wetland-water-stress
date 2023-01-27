from pathlib import Path
from typing import Dict

from numcodecs import Blosc
from xarray import Dataset


def _maybe_with_grid_mapping(new_encoding: Dict, var_encoding: Dict) -> Dict:
    if 'grid_mapping' in var_encoding:
        new_encoding['grid_mapping'] = var_encoding['grid_mapping']
    return new_encoding


def store_as_compressed_zarr(ds, out_path):
    encoding = make_encoding_for_compression({'cname': 'zstd', 'clevel': 5}, ds)
    ds = convert_path_and_string_objects_to_string(ds)
    ds.to_zarr(out_path, encoding=encoding)


def make_encoding_for_compression(compression: Dict, dataset: Dataset) -> Dict:
    compressor = Blosc(**compression)
    encoding = {name: _maybe_with_grid_mapping({"compressor": compressor}, var.encoding)
                for name, var in dataset.data_vars.items()}
    return encoding


def convert_path_and_string_objects_to_string(dataset: Dataset) -> Dataset:
    for cn, cc in dataset.coords.items():
        if cc.dtype == object and len(cc) > 0 and (isinstance(cc.values[0], Path) or isinstance(cc.values[0], str)):
            dataset.coords[cn] = cc.astype(str)
    return dataset
