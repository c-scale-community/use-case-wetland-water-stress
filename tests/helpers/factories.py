import numpy as np
from xarray import DataArray


def make_raster(values, param_dim=None):
    values = np.asarray(values)
    coords = {
        'y': ('y', np.arange(values.shape[-2], 0, -1)),
        'x': ('x', np.arange(values.shape[-1])),
        'spatial_ref': DataArray(0, attrs={'GeoTransform': '-0.5 1.0 0.0 -0.5 0.0 -1.0'})
    }
    dims = ('y', 'x')
    if values.ndim == 3:
        dim, pc = param_dim or ('parameters', np.arange(values.shape[0]))
        coords[dim] = (dim, pc)
        dims = (dim,) + dims

    return DataArray(values, coords=coords, dims=dims)
