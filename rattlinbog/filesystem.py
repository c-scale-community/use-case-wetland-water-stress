import re
from pathlib import Path

from eotransform_pandas.filesystem.gather import gather_files
from eotransform_pandas.filesystem.naming.geopathfinder_conventions import yeoda_naming_convention
from pandas import DataFrame

from rattlinbog.config import ParameterSelection

DATA_ROOT = Path("/data/wetland/")


def retrieve_params_df(selection: ParameterSelection) -> DataFrame:
    params_df = gather_files(Path(selection.root), yeoda_naming_convention, [
        re.compile(selection.parameter_type),
        re.compile('V1M0R1'),
        re.compile('EQUI7_EU020M'),
        re.compile('E\d\d\dN\d\d\dT3')
    ])
    return params_df


def retrieve_sample_df():
    return gather_files(DATA_ROOT, yeoda_naming_convention, [
        re.compile('samples'),
        re.compile('V1M0R1'),
        re.compile('EQUI7_EU020M'),
        re.compile('E\d\d\dN\d\d\dT3')
    ])
