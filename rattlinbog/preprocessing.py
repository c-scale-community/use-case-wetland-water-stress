import numpy as np
import xarray as xr
from xarray import DataArray


def preprocess_hparams(da: DataArray) -> DataArray:
    c1 = da.sel(parameter='SIG0-HPAR-C1')
    s1 = da.sel(parameter='SIG0-HPAR-S1')
    da_w_amp_a_phs = xr.concat([da,
                                np.sqrt(c1 ** 2 + s1 ** 2).expand_dims(parameter=['SIG0-HPAR-AMP']),
                                np.arctan2(s1, c1).expand_dims(parameter=['SIG0-HPAR-PHS'])], dim='parameter')
    nobs = da_w_amp_a_phs.sel(parameter='SIG0-HPAR-NOBS')
    mean = da_w_amp_a_phs.weighted(nobs).mean(dim='orbit', skipna=True, keep_attrs=True)
    mean.loc['SIG0-HPAR-NOBS', ...] = nobs.sum('orbit', skipna=True, keep_attrs=True)
    return mean.assign_coords(spatial_ref=da.coords['spatial_ref']).astype(np.float32)
