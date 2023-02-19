from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Union, Iterator, Dict, Any, Callable, Iterable, Optional

import torch as th
from numpy.typing import NDArray
from torch.nn import Parameter, Module
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

from rattlinbog.estimators.base import LogConfig, Score, Estimator

ModelParams = Union[Iterator[Parameter], Dict[Any, Parameter]]


@contextmanager
def evaluating(net):
    is_training = net.training
    try:
        net.eval()
        yield net
    finally:
        if is_training:
            net.train()


# turn of inspections that collide with scikit-learn API requirements & style guide, see:
# https://scikit-learn.org/stable/developers/develop.html
# noinspection PyPep8Naming,PyAttributeOutsideInit
class NNEstimator(Estimator, ABC):
    def __init__(self, net: Module, batch_size: int,
                 optim_factory: Callable[[ModelParams], Optimizer],
                 loss_fn: Callable[[Any, Any], Any], log_cfg: Optional[LogConfig] = None):
        self.net = net
        self.batch_size = batch_size
        self.optim_factory = optim_factory
        self.loss_fn = loss_fn
        self.log_cfg = log_cfg

    def fit(self, X: Dataset, y=None) -> "NNEstimator":
        dataloader = DataLoader(X, batch_size=self.batch_size)
        model_device = next(self.net.parameters()).device

        self.net.train()
        optimizer = self.optim_factory(self.net.parameters())

        estimated_len = getattr(X, 'estimated_len', None)
        total = estimated_len // self.batch_size if estimated_len else None
        for step, (x_batch, y_batch) in tqdm(enumerate(dataloader), "fitting", total):
            loss, estimate = self._optimization_step(optimizer, x_batch, y_batch, model_device)
            if self.log_cfg:
                self._log_progress(x_batch, y_batch, estimate.detach().cpu(), loss, step)

        self.is_fitted_ = True
        return self

    def _optimization_step(self, optimizer, x_batch, y_batch, model_device):
        estimate = self.net(x_batch.to(device=model_device))
        loss = self.loss_fn(estimate, y_batch.to(device=model_device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss, estimate

    def _log_progress(self, x_batch, y_batch, estimate, loss, step):
        self.log_cfg.log_sink.add_scalar("loss", loss.item(), step)

        valid_est = None
        valid_gt = None
        valid_cfg = self.log_cfg.validation
        if valid_cfg and self._should_log(valid_cfg.score_frequency, step):
            for n, s in self.score(x_batch.numpy(), y_batch.numpy()).items():
                self.log_cfg.log_sink.add_scalar(n, s, step)

            valid_src = valid_cfg.source
            valid_estimate_raw = valid_src.make_estimation_using(self, dict(raw=True))
            valid_gt = valid_src.ground_truth
            valid_est = self._refine_raw_prediction(valid_estimate_raw)

            valid_loss = self._loss_for_estimate(valid_estimate_raw, valid_gt)
            valid_score = self._score_estimate(valid_est, valid_gt)
            valid_cfg.log_sink.add_scalar("loss", valid_loss, step)
            for n, s in valid_score.items():
                valid_cfg.log_sink.add_scalar(n, s, step)

        if valid_cfg and self._should_log(valid_cfg.image_frequency, step):
            valid_src = valid_cfg.source
            if valid_est is None or valid_gt is None:
                valid_est = self._refine_raw_prediction(valid_src.make_estimation_using(self, dict(raw=True)))
                valid_gt = valid_src.ground_truth

            self.log_cfg.log_sink.add_image("images", self._visualize_batch(x_batch, estimate, y_batch), step)
            valid_cfg.log_sink.add_image("images", self._visualize(valid_src.parameters, valid_est, valid_gt), step)

    def _loss_for_estimate(self, estimate: NDArray, ground_truth: NDArray) -> float:
        x = th.from_numpy(estimate)
        y = th.from_numpy(ground_truth)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        while y.ndim < 4:
            y = y.unsqueeze(0)
        return self.loss_fn(x, y)

    def score(self, X: NDArray, y: NDArray) -> Score:
        return self._score_estimate(self.predict(X), y)

    @abstractmethod
    def _score_estimate(self, x: NDArray, y: NDArray) -> Score:
        ...

    @staticmethod
    def _should_log(frequency, step):
        return frequency is not None and step % frequency == 0

    def _visualize_batch(self, params, estimates, ground_truths):
        return make_grid(estimates)

    def _visualize(self, param, estimate, ground_truth):
        return estimate

    def predict(self, X: NDArray, raw=False) -> NDArray:
        if raw:
            return self._raw_prediction(X)
        return self._refine_raw_prediction(self._raw_prediction(X))

    def _raw_prediction(self, X):
        model_device = next(self.net.parameters()).device
        with th.no_grad(), evaluating(self.net) as net:
            x = th.from_numpy(X)
            if x.ndim == 3:
                x = x.unsqueeze(0)
            estimate = net(x.to(device=model_device))
            if estimate.shape[0] == 1:
                estimate = estimate.squeeze(0)
            return estimate.cpu().numpy()

    def _refine_raw_prediction(self, estimate: NDArray) -> NDArray:
        return estimate

    def _more_tags(self):
        return {'X_types': [Iterable[NDArray]], 'y_types': [Iterable[NDArray]]}
