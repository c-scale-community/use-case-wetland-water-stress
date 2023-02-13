from xarray import DataArray

from rattlinbog.estimators.base import Estimator


class _DataArrayMapper:
    def __init__(self, estimator: Estimator):
        self._estimator = estimator

    def to(self, array: DataArray) -> DataArray:
        out_dims = self._estimator.out_description.dims
        out_template = array[0, :, :].drop_vars(array.dims[0]).expand_dims(out_dims)

        estimated = array.data.map_blocks(lambda block: self._estimator.predict(block), drop_axis=0, new_axis=0,
                                          chunks=out_template.chunks, meta=out_template.data)
        return out_template.copy(data=estimated)


def apply(estimator: Estimator) -> _DataArrayMapper:
    return _DataArrayMapper(estimator)
