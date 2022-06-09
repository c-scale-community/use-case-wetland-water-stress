import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Tuple, Sequence, Set

import geopandas as pd
import rasterio
import rioxarray
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from shapely.geometry import Polygon
from xarray import Dataset

DateRange = Tuple[datetime, datetime]
DATE_FORMAT = "%Y%m%dT%H%M%S"
DATE_REGEX = re.compile(r"\d\d\d\d\d\d\d\dT\d\d\d\d\d\d_\d\d\d\d\d\d\d\dT\d\d\d\d\d\d")


def _daterange_from_safe_stem(stem: str) -> DateRange:
    dr_str = DATE_REGEX.search(stem).group(0)
    s_str, e_str = dr_str.split('_')
    return datetime.strptime(s_str, DATE_FORMAT), datetime.strptime(e_str, DATE_FORMAT)


def load_s1_dataset(files: Sequence[Path], bands: Set[str]) -> Dataset:
    data_vars = dict()
    min_time = datetime.max
    for band in bands:
        file = [f for f in files if band.lower() in f.stem.lower()][0]
        with rasterio.open(file) as src:
            _, crs = src.get_gcps()
            with WarpedVRT(src, resampling=Resampling.bilinear) as vrt:
                array = rioxarray.open_rasterio(vrt).squeeze(dim='band')
                array.attrs['crs'] = crs
                d_range = _daterange_from_safe_stem(file.parent.parent.stem)
                array.attrs['observed_time'] = d_range
                if d_range[0] < min_time:
                    min_time = d_range[0]
                data_vars[band] = array

    return Dataset(data_vars=data_vars, attrs=dict(time=min_time))


def load_s1_datasets_from_file_list(file_list: Path, bands: Set[str]) -> Sequence[Dataset]:
    safes = map(Path, file_list.read_text().splitlines())
    files = [[f for f in (s / "measurement").glob("*.tiff")] for s in safes]
    return [load_s1_dataset(fs, bands) for fs in files]


@dataclass
class ROI:
    name: str
    geometry: Polygon


def load_rois(shape_file: Path) -> Sequence[ROI]:
    df = pd.read_file(shape_file)
    return [ROI(row['officialna'], row['geometry']) for _, row in df.iterrows()]
