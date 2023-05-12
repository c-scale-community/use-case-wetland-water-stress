from datetime import datetime
from typing import Tuple, Sequence, Set

import pytest
from rasterio.crs import CRS
from xarray import Dataset

from rattlinbog.loaders import DateRange, load_s1_datasets_from_file_list, ROI, load_rois

FloatBBox = Tuple[float, float, float, float]

@pytest.mark.skip(reason='deprecated')
def test_loaded_data_arrays_from_list(path_to_file_list):
    arrays = load_s1_datasets_from_file_list(path_to_file_list, bands={'VH'})
    assert len(arrays) == 4
    assert_dataset_has(arrays[0], expected_bands={'VH'},
                       expected_bounds=(14.998117204417797, 47.573239187100604, 18.935188791209686, 49.479297581063086),
                       expected_crs=CRS.from_epsg(4326),
                       expected_observed_time=(datetime(2021, 12, 20, 16, 43, 8), datetime(2021, 12, 20, 16, 43, 33)))
    assert_dataset_has(arrays[1], expected_bands={'VH'},
                       expected_bounds=(11.14906467170389, 46.62118174717388, 15.055245103226198, 48.52427789290543),
                       expected_crs=CRS.from_epsg(4326),
                       expected_observed_time=(datetime(2021, 12, 12, 16, 58, 38), datetime(2021, 12, 12, 16, 59, 3)))
    assert_dataset_has(arrays[2], expected_bands={'VH'},
                       expected_bounds=(10.682511614023792, 48.217672991959255, 14.669717319061082, 50.123224509758344),
                       expected_crs=CRS.from_epsg(4326),
                       expected_observed_time=(datetime(2021, 12, 30, 16, 59, 46), datetime(2021, 12, 30, 17, 0, 11)))
    assert_dataset_has(arrays[3], expected_bands={'VH'},
                       expected_bounds=(8.87466159290116, 46.482527084688044, 12.74313020725043, 48.38184834366949),
                       expected_crs=CRS.from_epsg(4326),
                       expected_observed_time=(datetime(2021, 12, 15, 5, 26, 59), datetime(2021, 12, 15, 5, 27, 24)))


def assert_dataset_has(actual: Dataset, expected_bands: Set[str], expected_bounds: FloatBBox, expected_crs,
                       expected_observed_time: DateRange) -> None:
    assert set(actual.keys()) == expected_bands
    assert actual.rio.bounds() == expected_bounds
    assert actual.attrs['time'] == expected_observed_time[0]
    for band in expected_bands:
        assert actual[band].attrs['crs'] == expected_crs
        assert actual[band].attrs['observed_time'] == expected_observed_time


@pytest.mark.skip(reason='deprecated')
def test_load_rois(path_to_shape_file):
    rois = load_rois(path_to_shape_file)
    assert_rois_identity(rois, [
        ("Rheindelta", (9.561491318000037, 47.47985014200003, 9.676376617000074, 47.52508633300005)),
        ("Güssing Fishponds", (16.297327728637786, 47.045862249488685, 16.32202745563777, 47.05807951448869)),
        ("Wilder Kaiser", (12.20633199200006, 47.525188670000034, 12.423422865000077, 47.58324239400008)),
        ("Upper Drava River", (12.899867800229947, 46.72667071537859, 13.57698093027587, 46.833589755209786)),
        ("Donau-March-Thaya-Auen", (16.467305404000058, 48.10792485400003, 17.070127617000026, 48.72381368900005)),
        ("Bayerische Wildalm and Wildalmfilz",
         (11.789137484000037, 47.56767621400007, 11.817882354000062, 47.58815855600005)),
        ("Autertal - St. Lorenzener Hochmoor",
         (13.914479628421905, 46.85915070992317, 13.9281742993367, 46.87135409359562))])


def assert_rois_identity(actual: Sequence[ROI], expected_identity: Sequence[Tuple[str, FloatBBox]]) -> None:
    for actual_roi, (name, bounds) in zip(actual, expected_identity):
        assert actual_roi.name == name
        assert actual_roi.geometry.bounds == bounds
