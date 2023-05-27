import re
from pathlib import Path

from eotransform_pandas.filesystem.gather import gather_files
from eotransform_pandas.filesystem.naming.geopathfinder_conventions import yeoda_naming_convention
from pandas import DataFrame

from rattlinbog.config import ParameterSelection, SamplesSelection

DATA_ROOT = Path("/data/wetland/")


def retrieve_params_df(selection: ParameterSelection) -> DataFrame:
    params_df = gather_files(Path(selection.root), yeoda_naming_convention, [
        re.compile(selection.parameter_type),
        re.compile('V1M0R1'),
        re.compile('EQUI7_EU020M'),
        re.compile('E\d\d\dN\d\d\dT3')
    ])
    sel_mask = (params_df['var_name'] == selection.var_name) & \
               (params_df['datetime_1'].dt.year == selection.datetime_1_year) & \
               (params_df['datetime_2'].dt.year == selection.datetime_2_year)
    if selection.extra_field is not None:
        sel_mask = sel_mask & (params_df['extra_field'].map(str) == selection.extra_field)
    return params_df[sel_mask]


def retrieve_sample_df(selection: SamplesSelection):
    samples_df = gather_files(Path(selection.root), yeoda_naming_convention,
                         [re.compile('samples'), re.compile('V1M0R1'), re.compile('EQUI7_EU020M'),
                          re.compile('E\d\d\dN\d\d\dT3')])
    sel_mask = (samples_df['var_name'] == selection.var_name) & \
               (samples_df['extra_field'].map(str) == selection.extra_field) & \
               (samples_df['sensor_field'].map(str) == selection.sensor_field)
    return samples_df[sel_mask]
