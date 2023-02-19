from typing import Optional

from numpy.random import Generator
from xarray import Dataset

from rattlinbog.estimators.nn_estimator import NNEstimator
from rattlinbog.sampling.sample_patches_from_dataset import SamplingConfig, sample_patches_from_dataset, \
    make_balanced_sample_indices_for
from rattlinbog.th_extensions.utils.data.streamed_xarray_dataset import StreamedXArrayDataset
from rattlinbog.th_extensions.utils.dataset_splitters import split_to_params_and_ground_truth


def train(estimator: NNEstimator, train_ds: Dataset, sampling_cfg: SamplingConfig,
          rnd_generator: Optional[Generator] = None):
    sample_indices = make_balanced_sample_indices_for(train_ds, sampling_cfg, rnd_generator)
    array_stream = StreamedXArrayDataset(
        sample_patches_from_dataset(train_ds, sample_indices),
        split_to_params_and_ground_truth, estimator.batch_size * 2, sampling_cfg.n_samples)
    estimator.fit(array_stream)
    return estimator
