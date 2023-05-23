import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd
import xarray as xr
from equi7grid.equi7grid import Equi7Grid
from geopathfinder.naming_conventions.yeoda_naming import YeodaFilename
from rasterio.enums import Resampling


def main(tiles: Sequence[str], src: Path, dst: Path) -> None:
    cci_lc = xr.open_dataset(src, decode_coords='all', mask_and_scale=False)
    cci_year = pd.to_datetime(cci_lc.time.values[0].item()).year
    cci_lc.rio.write_crs('EPSG:4326', inplace=True)
    for tile in tiles:
        e7tile = Equi7Grid(20).create_tile(tile)
        mask_roi = cci_lc['lccs_class'].rio.clip_box(*e7tile.bbox_proj, crs=e7tile.core.projection.proj4)
        mask_roi.rio.write_nodata(0, inplace=True)
        mask_roi = mask_roi.rio.reproject(e7tile.core.projection.proj4, resolution=20, resampling=Resampling.nearest)
        mask_roi = mask_roi.rio.clip_box(*e7tile.bbox_proj, crs=e7tile.core.projection.proj4)
        mask_roi = mask_wetlands(mask_roi).astype('uint8')
        tile_dir = dst / e7tile.shortname
        tile_dir.mkdir(parents=True, exist_ok=True)
        out_name = YeodaFilename(dict(var_name='MASK-CCI',
                                      extra_field=cci_year,
                                      tile_name=e7tile.shortname,
                                      grid_name=tile.split('_')[0]))
        mask_roi.rio.to_raster(tile_dir / str(out_name))


def mask_wetlands(mask_roi):
    return (mask_roi == 160) | (mask_roi == 170) | (mask_roi == 180)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Rasterize the given shape file for the specified equi7 tile")
    parser.add_argument("tiles", help="Comma seperated list of tiles", type=str)
    parser.add_argument("src", help="CCI Land Cover source NetCDF", type=Path)
    parser.add_argument("dst_root", help="Root path of destination", type=Path)
    args = parser.parse_args()
    main(args.tiles.split(','), args.src, args.dst_root)
