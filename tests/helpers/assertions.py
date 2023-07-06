from typing import Sequence

import xarray as xr
from xarray import Dataset


def to_string(exception: Exception) -> str:
    return f"{type(exception).__name__}: {str(exception)}"


class MultipleExceptions(Exception):
    def __init__(self, exceptions: Sequence[Exception]):
        msg = "\n  " + "\n  ".join(map(to_string, exceptions))
        super().__init__(msg)


def gather_all_exceptions(params, code_to_execute):
    class _Collector:
        def __init__(self):
            self.exceptions = []

        def add(self, exception):
            self.exceptions.append(exception)

        def assert_any_is_true(self):
            if len(params) == len(self.exceptions):
                raise MultipleExceptions(self.exceptions)

    collector = _Collector()
    for p in params:
        try:
            code_to_execute(p)
        except Exception as e:
            collector.add(e)

    return collector


def assert_dataset_eq(actual: Dataset, expected: Dataset):
    xr.testing.assert_equal(actual, expected)


def assert_arrays_identical(actual, expected):
    xr.testing.assert_identical(actual, expected)
