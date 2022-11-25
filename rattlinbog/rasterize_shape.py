import argparse
import re
from pathlib import Path

import numpy as np
import geopandas as pd
import rioxarray    # noqa # pylint: disable=unused-import

from affine import Affine
from equi7grid.equi7grid import Equi7Grid
from geopathfinder.naming_conventions.yeoda_naming import YeodaFilename
from shapely.geometry import mapping
from xarray import DataArray

REGION_REGEX = re.compile(r"(EU|AF)\d\d\dM")
SAMPLING_REGEX = re.compile(r"\d\d\d")


def get_sampling_from_tile(tile_name: str) -> int:
    region = REGION_REGEX.search(tile_name).group()
    return int(SAMPLING_REGEX.search(region).group())


def main(tile: str, src_shape: Path, dst_file_dataset_root: Path):
    sampling = get_sampling_from_tile(tile)
    e7tile = Equi7Grid(sampling).create_tile(tile)

    shape = pd.read_file(src_shape)

    raster = DataArray(np.ones((1, *e7tile.shape_px()), dtype=np.uint8), dims=['band', 'y', 'x'])
    raster.rio.write_crs(e7tile.projection.proj4, inplace=True)
    raster.rio.write_transform(Affine.from_gdal(*e7tile.geotransform()), inplace=True)
    raster.rio.write_nodata(0, encoded=False, inplace=True)
    raster = raster.rio.clip(shape.geometry.apply(mapping), shape.crs, drop=False, invert=False)

    grid_name, tile_name = e7tile.name.split('_')
    out_smart_name = YeodaFilename(dict(var_name='MASK',
                                        extra_field=src_shape.stem.replace('_', '-').upper(),
                                        tile_name=tile_name,
                                        grid_name=grid_name))
    dst = dst_file_dataset_root / f"EQUI7_{grid_name}" / tile_name / str(out_smart_name)
    dst.parent.mkdir(parents=True, exist_ok=True)
    raster.rio.to_raster(dst, compress='ZSTD')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Rasterize the given shape file for the specified equi7 tile")
    parser.add_argument("tile", help="Target tile to rasterize the shape file to", type=str)
    parser.add_argument("shape", help="Path to the shape file to rasterize", type=Path)
    parser.add_argument("dst_root", help="Root path of destination file dataset", type=Path)
    args = parser.parse_args()
    main(args.tile, args.shape, args.dst_root)
