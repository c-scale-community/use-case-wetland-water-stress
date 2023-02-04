from typing import Optional, Iterator

import numpy as np
from numpy.random import Generator
from skimage.morphology import binary_dilation
from xarray import Dataset

from rattlinbog.geometry.rect_int import RectInt


def sample_patches_from_dataset(dataset: Dataset, patch_size: int, n_samples: int, never_nans: Optional[bool] = False,
                                rnd_generator: Optional[Generator] = None) -> Iterator[Dataset]:
    mask_yes_wl = dataset['mask'].fillna(0).astype(bool).values
    mask_no_wl = np.logical_not(mask_yes_wl)
    ps_h2 = patch_size // 2

    if never_nans:
        nan_masks = dataset.drop_vars('mask').isnull().to_array(dim='is_nan')
        nan_mask = binary_dilation(nan_masks.sum(dim=('is_nan', nan_masks.dims[1])) > 0, np.ones((ps_h2, ps_h2)))
        mask_yes_wl = np.logical_and(mask_yes_wl, np.logical_not(nan_mask))
        mask_no_wl = np.logical_and(mask_no_wl, np.logical_not(nan_mask))

    _mask_border(mask_yes_wl, ps_h2)
    _mask_border(mask_no_wl, ps_h2)

    balanced_sample_indices = sample_indices_balanced_from_masks(n_samples, mask_no_wl, mask_yes_wl, rnd_generator)
    for i in range(balanced_sample_indices.shape[1]):
        xy = balanced_sample_indices[:, i]
        selected_roi = RectInt(xy[1] - ps_h2, xy[1] + ps_h2, xy[0] - ps_h2, xy[0] + ps_h2)
        sampled = dataset.isel(selected_roi.to_slice_dict()).copy()
        sampled = sampled.load()
        sampled.attrs['name'] = f"sample_{i}"
        yield sampled


def sample_indices_balanced_from_masks(n_samples, mask_no_wl, mask_yes_wl, rnd_generator: Optional[Generator] = None):
    rnd_generator = rnd_generator or np.random.default_rng()
    indices_yes_wl = np.stack(np.nonzero(mask_yes_wl))
    indices_no_wl = np.stack(np.nonzero(mask_no_wl))
    choices_yes_wl = np.arange(indices_yes_wl.shape[1])
    rnd_generator.shuffle(choices_yes_wl)
    choices_no_wl = np.arange(indices_no_wl.shape[1])
    rnd_generator.shuffle(choices_no_wl)
    choices_yes_wl = choices_yes_wl[:n_samples // 2]
    choices_no_wl = choices_no_wl[:n_samples - (n_samples // 2)]
    balanced_sample_indices = np.concatenate([indices_yes_wl[:, choices_yes_wl],
                                              indices_no_wl[:, choices_no_wl]], axis=1)
    shuffled_indices = np.arange(n_samples)
    rnd_generator.shuffle(shuffled_indices)
    return balanced_sample_indices[:, shuffled_indices]


def _mask_border(array, border_width):
    array[:border_width, :] = False
    array[-border_width:, :] = False
    array[:, :border_width] = False
    array[:, -border_width:] = False
