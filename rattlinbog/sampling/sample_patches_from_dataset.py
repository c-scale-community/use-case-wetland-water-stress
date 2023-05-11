from dataclasses import asdict
from typing import Optional, Iterator

import numpy as np
from numpy.random import Generator
from skimage.morphology import binary_dilation
from xarray import Dataset, DataArray

from rattlinbog.config import SamplingConfig
from rattlinbog.geometry.rect_geo import RectGeo
from rattlinbog.th_extensions.utils.dataset_splitters import GROUND_TRUTH_KEY, PARAMS_KEY

SAMPLED_INDICES_KEY = 'sampled_indices'


def sample_patches_from_dataset(dataset: Dataset, sample_indices: DataArray, n_draws: int,
                                rnd_generator: Optional[Generator] = None) -> Iterator[Dataset]:
    cfg = SamplingConfig(**sample_indices.attrs)

    x_res, y_res = dataset.rio.resolution()
    ps_h2_y = cfg.patch_size // 2 * y_res
    ps_h2_x = cfg.patch_size // 2 * x_res

    rnd_generator = rnd_generator or np.random.default_rng()
    draws = rnd_generator.choice(sample_indices.shape[1], n_draws, replace=False)
    indices = sample_indices[:, draws]
    for i in range(n_draws):
        yx = indices.pos[i]
        selected_roi = RectGeo(yx.y.item() - ps_h2_y,
                               yx.y.item() + ps_h2_y - y_res,
                               yx.x.item() - ps_h2_x,
                               yx.x.item() + ps_h2_x - x_res)
        sampled = dataset.sel(selected_roi.to_slice_dict()).copy()
        sampled.attrs['name'] = f"sample_{i}"
        yield sampled


def _needs_resampling(dataset, config: SamplingConfig):
    if SAMPLED_INDICES_KEY not in dataset.data_vars:
        return True
    indices_attrs = dataset[SAMPLED_INDICES_KEY].attrs
    return indices_attrs != asdict(config)


def make_balanced_sample_indices_for(dataset: Dataset, config: SamplingConfig,
                                     rnd_generator: Optional[Generator] = None) -> DataArray:
    ps_h2 = config.patch_size // 2
    mask_yes_wl, mask_no_wl = _calc_yes_and_no_masks(dataset, ps_h2, config.never_nans)

    rnd_generator = rnd_generator or np.random.default_rng()
    indices_yes_wl = np.stack(np.nonzero(mask_yes_wl))
    indices_no_wl = np.stack(np.nonzero(mask_no_wl))
    choices_yes_wl = np.arange(indices_yes_wl.shape[1])
    rnd_generator.shuffle(choices_yes_wl)
    choices_no_wl = np.arange(indices_no_wl.shape[1])
    rnd_generator.shuffle(choices_no_wl)

    n_samples = min(mask_yes_wl.sum() * config.oversampling_size, config.n_samples)

    choices_yes_wl = choices_yes_wl[:n_samples // 2]
    choices_no_wl = choices_no_wl[:n_samples - (n_samples // 2)]
    indices = np.concatenate([indices_yes_wl[:, choices_yes_wl],
                              indices_no_wl[:, choices_no_wl]], axis=1)
    idc_coords = dataset[GROUND_TRUTH_KEY].load().isel(y=indices[0], x=indices[1]).coords
    return DataArray(indices,
                     {'axes': ['y', 'x'],
                      'y': ('pos', idc_coords['y'].data),
                      'x': ('pos', idc_coords['x'].data)}, ('axes', 'pos'), attrs=asdict(config))


def _calc_yes_and_no_masks(dataset, ps_h2, never_nans):
    mask_yes_wl = dataset[GROUND_TRUTH_KEY].fillna(0).astype(bool).values
    mask_no_wl = np.logical_not(mask_yes_wl)
    if never_nans:
        nan_masks = dataset[PARAMS_KEY].isnull()
        nan_mask = binary_dilation(nan_masks.sum(dim=nan_masks.dims[0]) > 0, np.ones((ps_h2, ps_h2)))
        mask_yes_wl = np.logical_and(mask_yes_wl, np.logical_not(nan_mask))
        mask_no_wl = np.logical_and(mask_no_wl, np.logical_not(nan_mask))
    _mask_border(mask_yes_wl, ps_h2)
    _mask_border(mask_no_wl, ps_h2)
    return mask_yes_wl, mask_no_wl


def _mask_border(array, border_width):
    array[:border_width, :] = False
    array[-border_width:, :] = False
    array[:, :border_width] = False
    array[:, -border_width:] = False
