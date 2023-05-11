import argparse
import re
from pathlib import Path

import xarray as xr
import yaml
from eotransform_pandas.filesystem.gather import gather_files
from eotransform_pandas.filesystem.naming.geopathfinder_conventions import yeoda_naming_convention
from geopathfinder.naming_conventions.yeoda_naming import YeodaFilename

from rattlinbog.config import SamplingConfig
from rattlinbog.io_xarray.store_as_compressed_zarr import store_as_compressed_zarr
from rattlinbog.sampling.sample_patches_from_dataset import make_balanced_sample_indices_for


def sample(src_root: Path, dst_root: Path, config: SamplingConfig) -> None:
    src_df = gather_files(src_root, yeoda_naming_convention, [
        re.compile('hparam'),
        re.compile('V1M0R1'),
        re.compile('EQUI7_EU020M'),
        re.compile('E\d\d\dN\d\d\dT3'),
    ])
    for zarr in src_df['filepath']:
        ds = xr.open_zarr(zarr).persist()
        sample_indices = make_balanced_sample_indices_for(ds, config)
        out_name = Path(str(YeodaFilename.from_filename(Path(zarr).stem))).with_suffix('.zarr')
        store_as_compressed_zarr(sample_indices.to_dataset(name='samples'), dst_root / out_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sample from zarr archives")
    parser.add_argument("src_root", help="Root path of data zarr data archives", type=Path)
    parser.add_argument("dst_root", help="Root path of samples zarr data archives", type=Path)
    parser.add_argument("config", help="Config file", type=Path)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    sample(args.src_root, args.dst_root, cfg)
