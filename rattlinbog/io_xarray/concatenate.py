from pathlib import Path
from typing import Tuple

import xarray as xr
from xarray import Dataset, DataArray


def concatenate_training_datasets(*zarrs: Tuple[Path, ...]) -> Dataset:
    return xr.open_mfdataset(zarrs, engine='zarr', combine='by_coords', chunks={})


def concatenate_indices_dataset(*zarrs: Tuple[Path, ...]) -> DataArray:
    return xr.open_mfdataset(zarrs, engine='zarr', concat_dim='pos', combine='nested', chunks={}).to_array()
