import numpy as np
import pytest
from xarray import DataArray

from rattlinbog.sampling import make_quantile_masks, make_histogram_masks


@pytest.fixture
def balanced_da():
    return DataArray([
        [-2] * 5,
        [-1] * 5,
        [0] * 5,
        [1] * 5,
        [2] * 5,
    ], dims=('y', 'x'))


def test_make_quantile_masks(balanced_da):
    masks = make_quantile_masks(balanced_da, quantiles=[0, 0.25, 0.5, 0.75, 1.0])
    np.testing.assert_equal(masks[0].values, true_at_row(0))
    np.testing.assert_equal(masks[1].values, true_at_row(1))
    np.testing.assert_equal(masks[2].values, true_at_row(2))
    np.testing.assert_equal(masks[3].values, true_at_row(3))


def true_at_row(row):
    m = np.full((5, 5), False)
    m[row, :] = True
    return m


def test_quantile_masks_upper_bound_is_greater_or_equal(balanced_da):
    masks = make_quantile_masks(balanced_da, quantiles=[0.5, 1.0])
    np.testing.assert_equal(masks[0].values, true_at_row(slice(0, 3)))
    np.testing.assert_equal(masks[1].values, true_at_row(slice(3, 5)))


def test_quantile_masks_come_with_a_name(balanced_da):
    masks = make_quantile_masks(balanced_da, quantiles=[0.5, 1.0])
    assert masks[0].name == "quantile_0.5"
    assert masks[1].name == "quantile_1.0"


def test_histogram_mask(balanced_da):
    masks = make_histogram_masks(balanced_da, bins=5)
    np.testing.assert_equal(masks[0].values, true_at_row(0))
    np.testing.assert_equal(masks[1].values, true_at_row(1))
    np.testing.assert_equal(masks[2].values, true_at_row(2))
    np.testing.assert_equal(masks[3].values, true_at_row(3))


def test_histogram_mask_with_bins_as_a_sequence(balanced_da):
    masks = make_histogram_masks(balanced_da, bins=[-2, 0, 2])
    np.testing.assert_equal(masks[0].values, true_at_row(slice(0, 2)))
    np.testing.assert_equal(masks[1].values, true_at_row(slice(2, 5)))

def test_histogram_masks_come_with_a_name(balanced_da):
    masks = make_histogram_masks(balanced_da, bins=[-2, -1, 1, 2])
    assert masks[0].name == "bin_[-2, -1)"
    assert masks[1].name == "bin_[-1, 1)"
    assert masks[2].name == "bin_[1, 2]"