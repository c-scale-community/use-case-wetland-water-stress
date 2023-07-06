import numpy as np
from xarray import DataArray


def make_raster(values, param_dim=None, coords=None):
    coords = coords or {}
    values = np.asarray(values)
    coords['y'] = coords.get('y', np.arange(values.shape[-2], 0, -1))
    coords['x'] = coords.get('x', np.arange(values.shape[-1]))
    values = np.asarray(values)
    coords = {
        'y': ('y', coords['y']),
        'x': ('x', coords['x']),
        'spatial_ref': DataArray(0, attrs={'GeoTransform': '-0.5 1.0 0.0 -0.5 0.0 -1.0'})
    }
    dims = ('y', 'x')
    if values.ndim == 3:
        dim, pc = param_dim or ('parameters', np.arange(values.shape[0]))
        coords[dim] = (dim, pc)
        dims = (dim,) + dims

    return DataArray(values, coords=coords, dims=dims)
