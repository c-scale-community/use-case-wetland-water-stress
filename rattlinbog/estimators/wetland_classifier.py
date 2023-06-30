from dataclasses import asdict
from typing import Optional, Dict

import numpy as np
import torch as th

from numpy.typing import NDArray
from sklearn.base import ClassifierMixin
from sklearn.metrics import confusion_matrix
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torchvision.utils import make_grid, draw_segmentation_masks

from rattlinbog.estimators.base import EstimateDescription, LogConfig, Score
from rattlinbog.estimators.nn_estimator import NNEstimator
from rattlinbog.evaluate.classification import score_first_order, score_second_order, score_zero_order
from rattlinbog.evaluate.confusion_matrix_binary import confusion_matrix_fast_binary
from rattlinbog.th_extensions.nn.unet import UNet


def sigmoid(x: NDArray) -> NDArray:
    return 1 / (1 + np.exp(-x))


def select_from_dict(d: Dict, i: int) -> Dict:
    return {k: v[i] for k, v in d.items()}


class WetlandClassifier(NNEstimator, ClassifierMixin):

    def __init__(self, net: UNet, batch_size: int, log_cfg: Optional[LogConfig] = None):
        super().__init__(net, batch_size, lambda p: Adam(p, lr=0.01), BCEWithLogitsLoss(), log_cfg)

    def _refine_raw_prediction(self, estimate: NDArray) -> NDArray:
        return sigmoid(estimate)

    def score_estimate(self, estimates: NDArray, ground_truth: NDArray) -> Score:
        estimates = (estimates > 0.5).ravel()
        cm = confusion_matrix_fast_binary(ground_truth.ravel(), estimates)
        zro = score_zero_order(cm)
        fst = score_first_order(zro)
        snd = score_second_order(fst)
        return select_from_dict({**asdict(fst), **asdict(snd)}, -1)

    def _visualize_batch(self, params, estimates, ground_truths):
        return make_grid([self._visualize(p, e, gt) for p, e, gt in zip(params, estimates, ground_truths)], 4)

    def _visualize(self, param, estimate, ground_truth):
        param = th.as_tensor(param)
        est_b = th.as_tensor(estimate > 0.5)
        gt_b = th.as_tensor(ground_truth == 1)
        if gt_b.ndim == 2:
            gt_b = gt_b[None, :, :]
        if param.shape[0] > 3:
            param = param[:3]

        mask = th.concat([gt_b, est_b, gt_b & est_b])
        bg = (th.clip(param, 0, 1) * 160).type(th.uint8)
        return draw_segmentation_masks(bg, mask, 0.8, ['#56B4E9', '#F0E442', '#00FF00'])


    @property
    def out_description(self) -> EstimateDescription:
        return EstimateDescription({'class_probs': ['is_wetland']}, self.net.num_hidden_layers)
