import numpy as np
from xarray import DataArray

from rattlinbog.sampling import make_histogram_masks


def test_make_quantile_masks():
    dataset = DataArray([
        [-2] * 5,
        [-1] * 5,
        [0] * 5,
        [1] * 5,
        [2] * 5,
    ], dims=('y', 'x'))

    masks = make_histogram_masks(dataset, quantiles=[0, 0.25, 0.5, 0.75, 1.0])

    np.testing.assert_equal(masks[0], true_at_row(0))
    np.testing.assert_equal(masks[1], true_at_row(1))
    np.testing.assert_equal(masks[2], true_at_row(2))
    np.testing.assert_equal(masks[3], true_at_row(3))


def true_at_row(row):
    m = np.full((5, 5), False)
    m[row, :] = True
    return m
