import os
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch as th

from rattlinbog.loaders import load_s1_datasets_from_file_list, load_rois

sys.path.append((Path(__file__).parent / "helpers").as_posix())

PERFORMANCE_TEST_ENV_VAR = 'USE_CASE_WETLAND_WATER_STRESS_RUN_PERF_TESTS'


def pytest_addoption(parser):
    parser.addoption(
        "--performance", action="store_true", default=False, help="run performance tests"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--performance") or os.getenv(PERFORMANCE_TEST_ENV_VAR):
        return
    skip_performance = pytest.mark.skip(reason="need --performance option to run")
    for i in items:
        if "performance" in i.keywords:
            i.add_marker(skip_performance)


@pytest.fixture
def resource_path():
    return Path(__file__).parent / "resources"


@pytest.fixture
def path_to_file_list(resource_path):
    return resource_path / "s1-file-list.txt"


@pytest.fixture
def path_to_shape_file():
    return Path("/shared/ramsar/RAMSAR_AT_01.shp")


@pytest.fixture
def vh_datasets(path_to_file_list):
    return load_s1_datasets_from_file_list(path_to_file_list, {'VH'})


@pytest.fixture(scope='module')
def fixed_seed():
    py_state = random.getstate()
    np_state = np.random.get_state()
    th_state = th.random.get_rng_state()
    random.seed(42)
    np.random.seed(42)
    th.random.manual_seed(42)
    yield 42
    random.setstate(py_state)
    np.random.set_state(np_state)
    th.random.set_rng_state(th_state)


@pytest.fixture
def ramsar_rois(path_to_shape_file):
    return load_rois(path_to_shape_file)


def pytest_addoption(parser):
    parser.addoption("--plot", action="store_true")


@pytest.fixture(scope="session")
def should_plot(pytestconfig):
    return pytestconfig.getoption("plot")
