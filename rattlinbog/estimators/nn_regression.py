from abc import abstractmethod
from dataclasses import dataclass
from typing import Union, Iterator, Dict, Any, Callable, Iterable, Optional

import torch as th
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from torch.nn import Parameter, Module
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing_extensions import Protocol

ModelParams = Union[Iterator[Parameter], Dict[Any, Parameter]]


class LogSink(Protocol):
    @abstractmethod
    def add_scalar(self, tag, scalar_value, global_step=None):
        ...


@dataclass
class Validation:
    frequency: int
    validator: Callable[["NNEstimator"], float]


@dataclass
class LogConfig:
    log_sink: LogSink
    validation: Optional[Validation] = None


# turn of inspections that collide with scikit-learn API requirements & style guide, see:
# https://scikit-learn.org/stable/developers/develop.html
# noinspection PyPep8Naming,PyAttributeOutsideInit
class NNEstimator(BaseEstimator):
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
        for x_batch, y_batch in tqdm(dataloader, "fitting", total):
            estimate = self.net(x_batch.to(device=model_device))
            loss = self.loss_fn(estimate, y_batch.to(device=model_device))
            if self.log_cfg:
                self.log_cfg.log_sink.add_scalar("loss", loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.is_fitted_ = True
        return self

    def predict(self, X: NDArray) -> NDArray:
        model_device = next(self.net.parameters()).device
        with th.no_grad():
            self.net.eval()
            return self.net(th.from_numpy(X).unsqueeze(0).to(device=model_device)).squeeze(0).cpu().numpy()

    def _more_tags(self):
        return {'X_types': [Iterable[NDArray]], 'y_types': [Iterable[NDArray]]}
