import argparse
from pathlib import Path

import rioxarray
import xarray as xr
import yaml
from eotransform_pandas.filesystem.gather import gather_files
from eotransform_pandas.filesystem.naming.geopathfinder_conventions import yeoda_naming_convention
from equi7grid.equi7grid import Equi7Grid
from geopathfinder.naming_conventions.yeoda_naming import YeodaFilename
from xarray import Dataset, DataArray

from rattlinbog.config import Restructure
from rattlinbog.io_xarray.store_as_compressed_zarr import store_as_compressed_zarr
from rattlinbog.loaders import load_harmonic_orbits
from rattlinbog.preprocessing import preprocess_hparams
from rattlinbog.rasterize_shape import get_sampling_from_tile


def restructure(tile: str, parameter_file_ds_root: Path, mask_file_ds_root: Path, dst_root: Path, config: Restructure):
    sampling = get_sampling_from_tile(tile)
    e7tile = Equi7Grid(sampling).create_tile(tile)
    grid_name, tile_name = e7tile.name.split('_')
    parameter_files = gather_files(parameter_file_ds_root / f"EQUI7_{grid_name}" / tile_name, yeoda_naming_convention)
    parameter_files = parameter_files.sort_values('extra_field')
    parameters = list(sorted(set(parameter_files['var_name'])))

    parameters_arrays = preprocess_hparams(xr.concat([load_harmonic_orbits(parameter_files, p) for p in parameters],
                                                     dim=DataArray(parameters, dims=['parameter'])))

    mask_tile_root = mask_file_ds_root / f"EQUI7_{grid_name}" / tile_name
    mask_file = gather_files(mask_tile_root, yeoda_naming_convention)['filepath'].iloc[0]
    mask = rioxarray.open_rasterio(mask_file, chunks="auto")

    restructured_ds = Dataset(dict(params=parameters_arrays, ground_truth=mask[0]))

    for roi in config.rois:
        parent_extra = YeodaFilename.from_filename(mask_file.name)['extra_field']
        smart_name = Path(str(YeodaFilename(dict(var_name=f'{parameter_file_ds_root.parent.name}-MASK',
                                                 extra_field=f"{parent_extra}-ROI-{'-'.join(map(str, roi))}",
                                                 grid_name=grid_name,
                                                 tile_name=tile_name))))
        out = dst_root / f"EQUI7_{grid_name}" / tile_name / smart_name.with_suffix('.zarr')
        out.parent.mkdir(parents=True, exist_ok=True)

        top, left, height, width = roi
        roi = restructured_ds.isel(y=slice(top, top + height), x=slice(left, left + width))
        store_as_compressed_zarr(roi.chunk({'parameter': -1, 'y': config.chunk_size, 'x': config.chunk_size}), out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Restructure the parameters file and the mask to zarr archive "
                                                 "for given tile")
    parser.add_argument("tile", help="Target tile to rasterize the shape file to", type=str)
    parser.add_argument("param_root", help="Path to the parameter root file dataset", type=Path)
    parser.add_argument("mask_root", help="Path to rasterized mask file dataset root", type=Path)
    parser.add_argument("dst_root", help="Root path of destination file dataset", type=Path)
    parser.add_argument("config", help="Config file", type=Path)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    restructure(args.tile, args.param_root, args.mask_root, args.dst_root, cfg)
