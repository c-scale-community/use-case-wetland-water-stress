from datetime import datetime
from typing import Dict, Sequence

import numpy as np
import xarray as xr
from xarray import Dataset

from rattlinbog.data_group import DataGroup
from rattlinbog.transforms import CoarsenAvgSpatially, ClipRoi, ConcatTimeSeries, ClipValues, RoundToInt16, \
    StoreAsNetCDF, NameDatasets, EatMyData


def test_coarsen_data_group_spatially_trimming_edges():
    data_group = make_data_group(
        dict(area_0=[make_dataset([[[2, 2], [2, 2]],
                                   [[2, 1], [1, 2]]])],
             area_1=[make_dataset([[[2, 1, 3], [2, 1, 3]]])])
    )
    coarsen = CoarsenAvgSpatially(stride=2)
    assert_group_arrays_eq(coarsen(data_group), dict(area_0=make_dataset([[[2]],
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


def assert_group_arrays_eq(actual: DataGroup, expected_datas: Dict[str, Dataset]) -> None:
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
    assert_group_arrays_eq(stacked(data_group), dict(area_0=make_dataset([[[2, 1], [1, 2]],
                                                                          [[2, 2], [2, 2]]],
                                                                         times=[datetime(2021, 1, 1),
                                                                                datetime(2021, 1, 2)])))


def test_clamp():
    data_group = make_data_group(
        dict(area_0=[make_dataset([[[10, 2], [3, 4]],
                                   [[2, 0], [0, 1]]])])
    )
    clamped = ClipValues(vmin=1, vmax=4)
    assert_group_arrays_eq(clamped(data_group), dict(area_0=make_dataset([[[4, 2], [3, 4]],
                                                                          [[2, 1], [1, 1]]])))


def test_round_to_int16():
    data_group = make_data_group(
        dict(area_0=[make_dataset([[[10.5, 10.6], [10.4, 10]],
                                   [[11.5, 11.6], [11.4, 11]]], dtype=np.float32)])
    )
    rounded = RoundToInt16()
    assert_group_arrays_eq(rounded(data_group), dict(area_0=make_dataset([[[10, 11], [10, 10]],
                                                                          [[12, 12], [11, 11]]], dtype=np.int16)))


def test_namer():
    data_group = make_data_group(
        dict(area_0=[make_dataset([[0]], attrs=dict(some="distinguishing_attribute"))])
    )

    def names_dataset(ds: Dataset) -> str:
        return f"foo_{ds.attrs['some']}"

    named = NameDatasets(names_dataset)
    assert_group_dataset_attrs_eq(named(data_group), dict(area_0=[dict(some="distinguishing_attribute",
                                                                       name="foo_distinguishing_attribute")]))


def assert_group_dataset_attrs_eq(actual: DataGroup, expected_attrs: Dict[str, Dict]) -> None:
    for k, ds_attrs in expected_attrs.items():
        for a_d, e_attrs in zip(actual[k], ds_attrs):
            assert a_d.attrs == e_attrs


def test_store_as_net_cdf(tmp_path):
    storage_path = tmp_path / "storage"
    data_group = make_data_group(
        dict(area_0=[make_dataset([[[1, 2], [5, 6]],
                                   [[3, 4], [7, 8]]], attrs=dict(name="dataset_0")),
                     make_dataset([[[11, 12], [15, 16]],
                                   [[13, 14], [17, 18]]], attrs=dict(name="dataset_1"))],
             area_1=[make_dataset([[[0.1, 0.2], [0.3, 0.4]],
                                   [[0.5, 0.6], [0.7, 0.8]]], attrs=dict(name="dataset_0"))]))
    stored = StoreAsNetCDF(storage_path)
    stored(data_group)

    xr.testing.assert_equal(xr.open_dataset(storage_path / "area_0/dataset_0.nc"),
                            make_dataset([[[1, 2], [5, 6]],
                                          [[3, 4], [7, 8]]], attrs=dict(name="dataset_0")))
    xr.testing.assert_equal(xr.open_dataset(storage_path / "area_0/dataset_1.nc"),
                            make_dataset([[[11, 12], [15, 16]],
                                          [[13, 14], [17, 18]]], attrs=dict(name="dataset_1")))
    xr.testing.assert_equal(xr.open_dataset(storage_path / "area_1/dataset_0.nc"),
                            make_dataset([[[0.1, 0.2], [0.3, 0.4]],
                                          [[0.5, 0.6], [0.7, 0.8]]], attrs=dict(name="dataset_0")))


def assert_dataset_eq(actual: Dataset, expected: Dataset):
    xr.testing.assert_equal(actual, expected)


def test_eat_my_data():
    data_group = make_data_group(dict(area_0=[make_dataset([[0]]),
                                              make_dataset([[1]])],
                                      area_1=[make_dataset([[1]])]))
    eaten = EatMyData()
    assert len(eaten(data_group)) == 0
