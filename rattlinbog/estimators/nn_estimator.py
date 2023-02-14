from abc import ABC
from typing import Union, Iterator, Dict, Any, Callable, Iterable, Optional

import torch as th
from numpy.typing import NDArray
from torch.nn import Parameter, Module
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from rattlinbog.estimators.base import LogConfig, ScoreableEstimator, Score

ModelParams = Union[Iterator[Parameter], Dict[Any, Parameter]]


# turn of inspections that collide with scikit-learn API requirements & style guide, see:
# https://scikit-learn.org/stable/developers/develop.html
# noinspection PyPep8Naming,PyAttributeOutsideInit
class NNEstimator(ScoreableEstimator, ABC):
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
            estimate = self.net(x_batch.to(device=model_device))
            loss = self.loss_fn(estimate, y_batch.to(device=model_device))

            if self.log_cfg:
                self._log_progress(x_batch, y_batch, loss, step)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.is_fitted_ = True
        return self

    def _log_progress(self, x_batch, y_batch, loss, step):
        self.log_cfg.log_sink.add_scalar("loss", loss.item(), step)
        if self._should_log(self.log_cfg.validation, step):
            self.log_cfg.log_sink.add_scalars("score", self.score(x_batch.numpy(), y_batch.numpy()), step)
            validation = self.log_cfg.validation.validator(self)
            self.log_cfg.validation.log_sink.add_scalar("loss", validation.loss, step)
            self.log_cfg.validation.log_sink.add_scalars("score", validation.score, step)

        if self._should_log(self.log_cfg.image, step):
            self.log_cfg.image.log_sink.add_image("images", self.log_cfg.image.image_producer(self), step)

    @staticmethod
    def _should_log(cfg, step):
        return cfg and step % cfg.frequency == 0

    def predict(self, X: NDArray) -> NDArray:
        model_device = next(self.net.parameters()).device
        with th.no_grad():
            self.net.eval()
            x = th.from_numpy(X)
            if x.ndim == 3:
                x = x.unsqueeze(0)
            estimate = self.net(x.to(device=model_device))
            if estimate.shape[0] == 1:
                estimate = estimate.squeeze(0)
            return estimate.cpu().numpy()

    def score(self, X: NDArray, y: NDArray) -> Score:
        return self.score_estimate(self.predict(X), y)

    def loss_for_estimate(self, estimate: NDArray, ground_truth: NDArray) -> float:
        return self.loss_fn(estimate, ground_truth)

    def _more_tags(self):
        return {'X_types': [Iterable[NDArray]], 'y_types': [Iterable[NDArray]]}
