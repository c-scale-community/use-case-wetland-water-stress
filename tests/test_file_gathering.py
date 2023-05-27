from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
from geopathfinder.naming_conventions.yeoda_naming import YeodaFilename

from rattlinbog.config import ParameterSelection
from rattlinbog.filesystem import retrieve_params_df


@pytest.fixture
def data_root(tmp_path):
    return tmp_path


@pytest.fixture
def expected_hparam_archives(data_root):
    tile_root = data_root / "hparam/V1M0R1/EQUI7_EU020M"
    # generate_file(tile_root, "E051N015T3", "SIG0-HPAR-MASK-CCI", datetime(2019, 1, 1), datetime(2021, 1, 1), "2020")
    # generate_file(tile_root, "E060N012T3", "SIG0-HPAR-MASK-CCI", datetime(2019, 1, 1), datetime(2021, 1, 1), "2020")
    # generate_file(tile_root, "E051N015T3", "SIG0-HPAR-MASK-CCI", datetime(2017, 1, 1), datetime(2018, 1, 1), "2020")
    # generate_file(tile_root, "E060N012T3", "SIG0-HPAR-MASK-CCI", datetime(2017, 1, 1), datetime(2018, 1, 1), "2020")
    # generate_file(tile_root, "E051N015T3", "SIG0-HPAR-MASK-CORINE-BOGS-MARSHES", datetime(2017, 1, 1),
    #               datetime(2018, 1, 1), "2018")
    # generate_file(tile_root, "E060N012T3", "SIG0-HPAR-MASK-CORINE-BOGS-MARSHES", datetime(2017, 1, 1),
    #               datetime(2018, 1, 1), "2018")
    expected = []
    expected.append(generate_file(tile_root, "E051N015T3", "SIG0-HPAR-MASK-CCI", datetime(2017, 1, 1),
                                  datetime(2018, 1, 1), "2018"))
    expected.append(generate_file(tile_root, "E060N012T3", "SIG0-HPAR-MASK-CCI", datetime(2017, 1, 1),
                                  datetime(2018, 1, 1), "2018"))
    return expected


def generate_file(tile_root: Path, tile: str, var_name: str, datetime_1: datetime, datetime_2: datetime,
                  extra: str) -> Path:
    smart_name = YeodaFilename(dict(var_name=var_name, datetime_1=datetime_1, datetime_2=datetime_2, extra_field=extra,
                                    tile_name=tile, grid_name="EU020M"))
    file = tile_root / tile / str(smart_name)
    file.parent.mkdir(parents=True, exist_ok=True)
    file.touch(exist_ok=True)
    return file


@pytest.fixture
def hparam_selection(data_root):
    return ParameterSelection(data_root.as_posix(), "hparam", "SIG0-HPAR-MASK-CCI", 2017, 2018, "2018")


def test_retrieve_hparam_archives(data_root, hparam_selection, expected_hparam_archives):
    df = retrieve_params_df(hparam_selection)
    assert set(df['filepath']) == set(expected_hparam_archives)
