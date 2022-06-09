from datetime import datetime
from typing import Dict, Sequence

import numpy as np
import xarray as xr
from xarray import Dataset

from rattlinbog.data_group import DataGroup
from rattlinbog.transforms import CoarsenAvgSpatially, ClipRoi, ConcatTimeSeries, ClipValues, RoundToInt16


def test_coarsen_data_group_spatially_trimming_edges():
    data_group = make_data_group(
        dict(area_0=[make_dataset([[[2, 2], [2, 2]],
                                   [[2, 1], [1, 2]]])],
             area_1=[make_dataset([[[2, 1, 3], [2, 1, 3]]])])
    )
    coarsen = CoarsenAvgSpatially(stride=2)
    assert_group_eq(coarsen(data_group), dict(area_0=make_dataset([[[2]],
                                                                   [[3 / 2]]]),
                                              area_1=make_dataset([[[3 / 2]]])))


def make_data_group(in_datas: Dict[str, Sequence[Dataset]]) -> DataGroup:
    return DataGroup(in_datas)


def make_dataset(values, attrs=None, times=None, dtype=None) -> Dataset:
    a = np.array(values, dtype=dtype)
    if a.ndim == 3:
        if times is None:
            ds = Dataset(data_vars={"1": (["time", "x", "y"], a)}, attrs=attrs)
        else:
            ds = Dataset(data_vars={"1": (["time", "x", "y"], a)}, coords={'time': times}, attrs=attrs)
    else:
        ds = Dataset(data_vars={"1": (["x", "y"], a)}, attrs=attrs)
    return ds


def assert_group_eq(actual: DataGroup, expected_datas: Dict[str, Dataset]) -> None:
    for k, ds in expected_datas.items():
        xr.testing.assert_equal(actual[k][0]['1'], ds['1'])


def test_clip_roi_bounds(vh_datasets, ramsar_rois):
    data_group = make_data_group(
        dict(area_0=[Dataset(data_vars={'VH': vh_datasets[0]['VH']}, attrs=dict(roi=ramsar_rois[4]))])
    )
    clipped = ClipRoi()
    assert clipped(data_group)['area_0'][0]['VH'].shape == (5260, 5149)


def test_concat_sorted():
    data_group = make_data_group(
        dict(area_0=[make_dataset([[2, 2], [2, 2]], attrs=dict(time=datetime(2021, 1, 2))),
                     make_dataset([[2, 1], [1, 2]], attrs=dict(time=datetime(2021, 1, 1)))])
    )
    stacked = ConcatTimeSeries()
    assert_group_eq(stacked(data_group), dict(area_0=make_dataset([[[2, 1], [1, 2]],
                                                                   [[2, 2], [2, 2]]], times=[datetime(2021, 1, 1),
                                                                                             datetime(2021, 1, 2)])))


def test_clamp():
    data_group = make_data_group(
        dict(area_0=[make_dataset([[[10, 2], [3, 4]],
                                   [[2, 0], [0, 1]]])])
    )
    clamped = ClipValues(vmin=1, vmax=4)
    assert_group_eq(clamped(data_group), dict(area_0=make_dataset([[[4, 2], [3, 4]],
                                                                   [[2, 1], [1, 1]]])))


def test_round_to_int16():
    data_group = make_data_group(
        dict(area_0=[make_dataset([[[10.5, 10.6], [10.4, 10]],
                                   [[11.5, 11.6], [11.4, 11]]], dtype=np.float32)])
    )
    rounded = RoundToInt16()
    assert_group_eq(rounded(data_group), dict(area_0=make_dataset([[[10, 11], [10, 10]],
                                                                   [[12, 12], [11, 11]]], dtype=np.int16)))

