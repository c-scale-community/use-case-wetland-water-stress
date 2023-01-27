from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import xarray as xr
from approvaltests import verify_with_namer_and_writer
from numpy.random import RandomState
from pytest_approvaltests_geo import GeoOptions, CompareGeoZarrs, ReportGeoZarrs, ExistingDirWriter
from xarray import Dataset

from rattlinbog.io_xarray.store_as_compressed_zarr import store_as_compressed_zarr
from rattlinbog.sampling.sample_patches_from_dataset import sample_patches_from_dataset


# from watercloud_regularized.config.output import Compression
# from watercloud_regularized.io_xarray.store_in_zarr_archive import store_in_zarr_archive
# from rattlinbog.sampling.sample_patches_from_dataset import sample_patches_from_dataset
# from rattlinbog.th_extensions.utils.data.streamed_xarray_dataset import StreamedXArrayDataset
# from rattlinbog.th_extensions.utils.dataset_splitters import sig0_autoencoder_splitter


@pytest.fixture
def verify_raster_as_geo_zarr(tmp_path):
    def verify(dataset: Dataset,
               *,  # enforce keyword arguments - https://www.python.org/dev/peps/pep-3102/
               options: Optional[GeoOptions] = None):
        options = options or GeoOptions()
        file_name = tmp_path / f"verifying-{dataset.name}-tmp.zarr"
        store_as_compressed_zarr(dataset, file_name)
        zarr_comparator = CompareGeoZarrs(options.scrub_tags, options.tolerance)
        zarr_reporter = ReportGeoZarrs(options.scrub_tags, options.tolerance)
        options = options.with_comparator(zarr_comparator)
        options = options.with_reporter(zarr_reporter)

        namer = options.namer
        namer.set_extension(file_name.suffix)
        verify_with_namer_and_writer(
            namer=namer,
            writer=ExistingDirWriter(file_name.as_posix()),
            options=options)

    return verify


@pytest.fixture
def tile_dataset():
    zarr = Path(
        "/eodc/private/tuwgeo/users/braml/data/wetland/hparam/V1M0R1/EQUI7_EU020M/E051N015T3/SIG0-HPAR-MASK____RAMSAR-AT-01_E051N015T3_EU020M__.zarr")
    return xr.open_zarr(zarr)


def test_stochastic_patch_samples_from_dataset(tile_dataset, verify_raster_as_geo_zarr):
    patches = list(sample_patches_from_dataset(tile_dataset, 32, 16, np.random.default_rng(42)))
    assert len(patches) == 16
    verify_raster_as_geo_zarr(patches[0])


def test_patch_samples_are_balanced(tile_dataset, verify_raster_as_geo_zarr):
    patches = list(sample_patches_from_dataset(tile_dataset, 32, 2, np.random.default_rng(42)))
    a_has_mask_pixles = patches[0]['mask'].sum().values.item() > 0
    b_has_mask_pixles = patches[1]['mask'].sum().values.item() > 0
    assert (a_has_mask_pixles and not b_has_mask_pixles) or (b_has_mask_pixles and not a_has_mask_pixles)
