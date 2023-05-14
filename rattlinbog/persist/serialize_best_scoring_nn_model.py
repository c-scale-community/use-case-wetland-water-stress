from pathlib import Path
from typing import Dict

import numpy as np
import torch as th

from rattlinbog.estimators.base import ModelSink
from rattlinbog.estimators.nn_estimator import NNEstimator


class SerializeBestScoringNNModel(ModelSink):
    def __init__(self, out_path: Path, score: str):
        self.out_path = out_path
        self.score_key = score
        self.last_maximum = -np.inf

    def snapshot(self, model: NNEstimator, score: Dict) -> None:
        current = score[self.score_key]
        if current > self.last_maximum:
            th.save(model.net, self.out_path / f"{model.__class__.__name__}-{self.score_key}-best.pt")
            self.last_maximum = current
