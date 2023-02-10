from xarray import DataArray

from rattlinbog.estimators.base import ClassEstimatorMixin


class _DataArrayMapper:
    def __init__(self, estimator: ClassEstimatorMixin):
        self._estimator = estimator

    def to(self, array: DataArray) -> DataArray:
        classes = self._estimator.classes
        c_sizes = array.chunksizes
        out_template = array[0, :, :].drop_vars(array.dims[0]).expand_dims({'classes': classes})
        estimated = array.data.map_blocks(lambda block: self._estimator.predict(block), drop_axis=0, new_axis=0,
                                          chunks=(len(classes), c_sizes['y'], c_sizes['x']), meta=out_template.data)
        return out_template.copy(data=estimated)


def apply_classification(estimator: ClassEstimatorMixin) -> _DataArrayMapper:
    return _DataArrayMapper(estimator)
