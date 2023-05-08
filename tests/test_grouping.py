from datetime import datetime
from typing import Dict, Sequence

import pytest

from rattlinbog.data_group import group_datasets, GroupByRois, DataGroup
from rattlinbog.loaders import ROI


@pytest.mark.skip(reason='deprecated')
def test_group_datasets_by_roi(vh_datasets, ramsar_rois):
    group = group_datasets(vh_datasets, by_rule=GroupByRois(ramsar_rois))
    assert_grouped_identity(group, {'Autertal - St. Lorenzener Hochmoor': [datetime(2021, 12, 12, 16, 58, 38)],
                                    'Bayerische Wildalm and Wildalmfilz': [datetime(2021, 12, 12, 16, 58, 38),
                                                                           datetime(2021, 12, 15, 5, 26, 59)],
                                    'Donau-March-Thaya-Auen': [datetime(2021, 12, 20, 16, 43, 8)],
                                    'Rheindelta': [datetime(2021, 12, 15, 5, 26, 59)],
                                    'Upper Drava River': [datetime(2021, 12, 12, 16, 58, 38)],
                                    'Wilder Kaiser': [datetime(2021, 12, 12, 16, 58, 38),
                                                      datetime(2021, 12, 15, 5, 26, 59)]})


def assert_grouped_identity(actual: DataGroup, expected_identity: Dict[str, Sequence[datetime]]) -> None:
    assert set(actual.keys()) == set(expected_identity.keys())
    for k, actual_ds in actual.items():
        actual_times = [ad.attrs['time'] for ad in actual_ds]
        assert actual_times == expected_identity[k], f"with {k}"


@pytest.mark.skip(reason='deprecated')
def test_group_datasets_adds_roi_to_attribute(vh_datasets, ramsar_rois):
    group = group_datasets(vh_datasets, by_rule=GroupByRois(ramsar_rois))
    assert_rois_eq(group, {'Autertal - St. Lorenzener Hochmoor': [ramsar_rois[6]],
                           'Bayerische Wildalm and Wildalmfilz': [ramsar_rois[5], ramsar_rois[5]],
                           'Donau-March-Thaya-Auen': [ramsar_rois[4]],
                           'Rheindelta': [ramsar_rois[0]],
                           'Upper Drava River': [ramsar_rois[3]],
                           'Wilder Kaiser': [ramsar_rois[2], ramsar_rois[2]]})


def assert_rois_eq(actual: DataGroup, expected_rois: Dict[str, Sequence[ROI]]) -> None:
    for k, actual_ds in actual.items():
        actual_rois = [ad.attrs['roi'] for ad in actual_ds]
        assert actual_rois == expected_rois[k], k
