import argparse
from datetime import datetime
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

    if config.parameter_type == 'hparam':
        parameters_arrays = load_hparam_ds(parameter_file_ds_root, grid_name, tile_name)
    elif config.parameter_type == 'mmeans':
        parameters_arrays = load_mmean_ds(parameter_file_ds_root, grid_name, tile_name, config)
    else:
        raise NotImplementedError(config.parameter_type)

    mask_tile_root = mask_file_ds_root / f"EQUI7_{grid_name}" / tile_name
    mask_df = gather_files(mask_tile_root, yeoda_naming_convention)['filepath']
    if config.mask_extra_field is not None:
        mask_df = mask_df[mask_df['extra_field'] == config.mask_extra_field]
    mask_file = mask_df.iloc[0]
    mask = rioxarray.open_rasterio(mask_file, chunks="auto")

    restructured_ds = Dataset(dict(params=parameters_arrays, ground_truth=mask[0]))

    for roi in config.rois:
        parent_extra = YeodaFilename.from_filename(mask_file.name)['extra_field']
        smart_name = Path(str(YeodaFilename(dict(var_name=f'{parameter_file_ds_root.parent.name}-MASK-{mask_file_ds_root.parent.name}',
                                                 extra_field=f"{parent_extra}-ROI-{'-'.join(map(str, roi))}",
                                                 datetime_1=f"{config.datetime_1_year}0101T000000" if config.datetime_1_year else "",
                                                 datetime_2=f"{config.datetime_2_year}0101T000000" if config.datetime_2_year else "",
                                                 grid_name=grid_name,
                                                 tile_name=tile_name))))
        out = dst_root / f"EQUI7_{grid_name}" / tile_name / smart_name.with_suffix('.zarr')
        if out.exists():
            print(f"{out.name} already exists, skipping.")
            continue

        out.parent.mkdir(parents=True, exist_ok=True)

        top, left, height, width = roi
        roi = restructured_ds.isel(y=slice(top, top + height), x=slice(left, left + width))
        store_as_compressed_zarr(roi.chunk({'parameter': -1, 'y': config.chunk_size, 'x': config.chunk_size}), out)


def load_hparam_ds(ds_root, grid_name, tile_name):
    parameter_files = gather_files(ds_root / f"EQUI7_{grid_name}" / tile_name, yeoda_naming_convention)
    parameter_files = parameter_files.sort_values('extra_field')
    parameters = list(sorted(set(parameter_files['var_name'])))
    parameters_arrays = preprocess_hparams(xr.concat([load_harmonic_orbits(parameter_files, p) for p in parameters],
                                                     dim=DataArray(parameters, dims=['parameter'])))
    return parameters_arrays


def load_mmean_ds(ds_root: Path, grid_name: str, tile_name: str, config: Restructure) -> DataArray:
    params_df = gather_files(ds_root / f"EQUI7_{grid_name}" / tile_name, yeoda_naming_convention)
    only_oavg = params_df['extra_field'].str.startswith('OAVG')
    params_df = params_df[only_oavg]
    year_selection = (params_df['datetime_1'].dt.year == config.datetime_1_year) & \
                     (params_df['datetime_2'].dt.year == config.datetime_2_year)
    params_df = params_df[year_selection]
    params_df['month'] = [int(e.split('-')[-1]) for e in params_df['extra_field']]
    params_df = params_df.sort_values(['datetime_1', 'month'])
    params_ds = xr.open_mfdataset(params_df['filepath'], concat_dim='band', combine='nested', mask_and_scale=False)
    params_ds = params_ds.rename({'band': 'parameter', 'band_data': 'mmeans'})
    months = [datetime(r['datetime_1'].year, r['month'], 1) for _, r in params_df[['datetime_1', 'month']].iterrows()]
    params_ds = params_ds.assign_coords({'parameter': ('parameter', months)})
    return params_ds['mmeans'].load()


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
