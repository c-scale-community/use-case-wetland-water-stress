from approval_utilities.utilities.exceptions.exception_collector import gather_all_exceptions_and_throw
from approvaltests.namer import NamerFactory

from rattlinbog.data_group import group_datasets, GroupByRois
from rattlinbog.transforms import ClipRoi, ConcatTimeSeries, CoarsenAvgSpatially, ClipValues, RoundToInt16
from rattlinbog.transforms import Compose


def test_approve_load_and_process_dataset_pipeline(vh_datasets, ramsar_rois, verify_dataset):
    transformation_pipeline = Compose([ClipRoi(),
                                       ConcatTimeSeries(),
                                       ClipValues(vmin=0, vmax=200),
                                       CoarsenAvgSpatially(stride=15),
                                       RoundToInt16()])
    group = group_datasets(vh_datasets, by_rule=GroupByRois(ramsar_rois))
    group = transformation_pipeline(group)

    non_empy_rois = [r.name for r in ramsar_rois if len(group[r.name]) > 0]
    gather_all_exceptions_and_throw(non_empy_rois, lambda r: verify_dataset(group[r][0],
                                                                            options=NamerFactory.with_parameters(r)))
