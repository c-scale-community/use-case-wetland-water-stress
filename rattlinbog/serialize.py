from datetime import datetime
from pathlib import Path
from typing import Any, Union, get_args, List

import numpy as np
from xarray import Dataset

from rattlinbog.loaders import DATE_FORMAT, ROI

Number = Union[int, float]
Serializable = Union[str, Number, np.ndarray, list, tuple]


def serialize_datetime(value: datetime) -> str:
    return value.strftime(DATE_FORMAT)


def serialize_roi(value: ROI) -> List[str]:
    return [value.name, value.geometry.wkt]


def coerce_serializable(value: Any) -> Serializable:
    if isinstance(value, (tuple, list)):
        return tuple(coerce_serializable(v) for v in value)
    if isinstance(value, get_args(Serializable)):
        return value
    if isinstance(value, datetime):
        return serialize_datetime(value)
    if isinstance(value, ROI):
        return serialize_roi(value)
    raise NotImplementedError(f"Serialization not implemented for {type(value)}")


def coerce_attrs_serializable(attributable) -> None:
    for k in attributable.attrs:
        attributable.attrs[k] = coerce_serializable(attributable.attrs[k])

    if isinstance(attributable, Dataset):
        for k in attributable:
            coerce_attrs_serializable(attributable[k])


def store_dataset(file: Path, ds: Dataset) -> None:
    coerce_attrs_serializable(ds)
    ds.to_netcdf(file, engine='netcdf4')
