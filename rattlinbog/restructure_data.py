import argparse
from pathlib import Path

import rioxarray
import xarray as xr

from eotransform_pandas.filesystem.gather import gather_files
from eotransform_pandas.filesystem.naming.geopathfinder_conventions import yeoda_naming_convention
from equi7grid.equi7grid import Equi7Grid
from geopathfinder.naming_conventions.yeoda_naming import YeodaFilename
from xarray import Dataset

from rattlinbog.io_xarray.store_as_compressed_zarr import store_as_compressed_zarr
from rattlinbog.loaders import load_harmonic_orbits
from rattlinbog.rasterize_shape import get_sampling_from_tile


def restructure(tile: str, parameter_file_ds_root: Path, mask_file_ds_root: Path, dst_root: Path):
    sampling = get_sampling_from_tile(tile)
    e7tile = Equi7Grid(sampling).create_tile(tile)
    grid_name, tile_name = e7tile.name.split('_')
    parameter_files = gather_files(parameter_file_ds_root / f"EQUI7_{grid_name}" / tile_name, yeoda_naming_convention)

    parameters = list(sorted(set(parameter_files['var_name'])))

    def collapse_orbits(ds, name):
        return ds['orbits'].mean(dim='orbit').expand_dims(parameter=[name])

    parameters_arrays = xr.concat([collapse_orbits(load_harmonic_orbits(parameter_files, p), p)
                                   for p in parameters], dim='parameter').chunk({'parameter': -1, 'y': 1000, 'x': 1000})

    mask_tile_root = mask_file_ds_root / f"EQUI7_{grid_name}" / tile_name
    mask_file = gather_files(mask_tile_root, yeoda_naming_convention)['filepath'].iloc[0]
    mask = rioxarray.open_rasterio(mask_file).chunk({'band': 1, 'y': 1000, 'x': 1000})

    restructured_ds = Dataset(dict(params=parameters_arrays, mask=mask[0]))


    smart_name = Path(str(YeodaFilename(dict(var_name=f'{parameter_file_ds_root.parent.name}-MASK',
                                             extra_field=YeodaFilename.from_filename(mask_file.name)['extra_field'],
                                             grid_name=grid_name,
                                             tile_name=tile_name))))
    out = dst_root / f"EQUI7_{grid_name}" / tile_name / smart_name.with_suffix('.zarr')
    out.parent.mkdir(parents=True, exist_ok=True)

    store_as_compressed_zarr(restructured_ds, out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Restructure the parameters file and the mask to zarr archive "
                                                 "for given tile")
    parser.add_argument("tile", help="Target tile to rasterize the shape file to", type=str)
    parser.add_argument("param_root", help="Path to the parameter root file dataset", type=Path)
    parser.add_argument("mask_root", help="Path to rasterized mask file dataset root", type=Path)
    parser.add_argument("dst_root", help="Root path of destination file dataset", type=Path)
    args = parser.parse_args()
    restructure(args.tile, args.param_root, args.mask_root, args.dst_root)
