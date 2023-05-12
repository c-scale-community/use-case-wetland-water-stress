from typing import Optional

from numpy.random import Generator
from xarray import Dataset, DataArray

from rattlinbog.estimators.nn_estimator import NNEstimator
from rattlinbog.sampling.sample_patches_from_dataset import sample_patches_from_dataset
from rattlinbog.th_extensions.utils.data.streamed_xarray_dataset import StreamedXArrayDataset
from rattlinbog.th_extensions.utils.dataset_splitters import split_to_params_and_ground_truth


def train(estimator: NNEstimator, train_ds: Dataset, sample_indices: DataArray, n_draws: int,
          rnd_generator: Optional[Generator] = None):
    array_stream = StreamedXArrayDataset(
        sample_patches_from_dataset(train_ds, sample_indices, n_draws, rnd_generator),
        split_to_params_and_ground_truth, estimator.batch_size * 2, n_draws)
    estimator.fit(array_stream)
    return estimator
