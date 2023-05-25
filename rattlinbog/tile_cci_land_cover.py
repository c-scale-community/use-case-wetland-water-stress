import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd
import xarray as xr
from affine import Affine
from equi7grid.equi7grid import Equi7Grid
from geopathfinder.naming_conventions.yeoda_naming import YeodaFilename
from pyresample import geometry, image, kd_tree
from xarray import DataArray


def main(tiles: Sequence[str], src: Path, dst: Path) -> None:
    cci_lc = xr.open_dataset(src, decode_coords='all', mask_and_scale=False)
    cci_year = pd.to_datetime(cci_lc.time.values[0].item()).year
    cci_lc.rio.write_crs('EPSG:4326', inplace=True)
    for tile in tiles:
        e7tile = Equi7Grid(20).create_tile(tile)
        mask_roi = cci_lc['lccs_class'].rio.clip_box(*e7tile.bbox_proj, crs=e7tile.core.projection.proj4)
        mask_roi.rio.write_nodata(0, inplace=True)

        cci_area = geometry.AreaDefinition('cci', 'cci grid', 'cci', mask_roi.rio.crs,
                                           mask_roi.shape[2], mask_roi.shape[1], mask_roi.rio.bounds(True))

        equi7_area = geometry.AreaDefinition('equi7', 'equi7 EI', 'equi7', e7tile.core.projection.proj4,
                                             *e7tile.shape_px(), extent_of_e7tile(e7tile))

        mask_values = kd_tree.resample_nearest(cci_area, mask_roi.values[0], equi7_area, 3000, nprocs=16)

        mask_da = DataArray(mask_values[None, ...], dims=['band', 'y', 'x'])
        mask_da.rio.write_crs(e7tile.core.projection.proj4, inplace=True)
        mask_da.rio.write_transform(Affine.from_gdal(*e7tile.geotransform()), inplace=True)
        mask_da = mask_wetlands(mask_da).astype('uint8')
        tile_dir = dst / e7tile.shortname
        tile_dir.mkdir(parents=True, exist_ok=True)
        out_name = YeodaFilename(dict(var_name='MASK-CCI',
                                      extra_field=cci_year,
                                      tile_name=e7tile.shortname,
                                      grid_name=tile.split('_')[0]))
        mask_da.rio.to_raster(tile_dir / str(out_name), compress='ZSTD')


def mask_wetlands(mask_roi):
    return (mask_roi == 160) | (mask_roi == 170) | (mask_roi == 180)


def extent_of_e7tile(tile):
    return tile.llx, tile.lly, tile.llx + tile.core.tile_xsize_m, tile.lly + tile.core.tile_ysize_m


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Rasterize the given shape file for the specified equi7 tile")
    parser.add_argument("tiles", help="Comma seperated list of tiles", type=str)
    parser.add_argument("src", help="CCI Land Cover source NetCDF", type=Path)
    parser.add_argument("dst_root", help="Root path of destination", type=Path)
    args = parser.parse_args()
    main(args.tiles.split(','), args.src, args.dst_root)
