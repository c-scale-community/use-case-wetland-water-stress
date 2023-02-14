from xarray import Dataset

from rattlinbog.estimators.apply import apply
from rattlinbog.estimators.base import ScoreableEstimator, Validation
from rattlinbog.th_extensions.utils.dataset_splitters import GROUND_TRUTH_KEY, PARAMS_KEY


class ValidatorOfDataset:
    def __init__(self, validation_ds: Dataset):
        self._validation_ds = validation_ds

    def __call__(self, estimator: ScoreableEstimator) -> Validation:
        estimate = apply(estimator).to(self._validation_ds[PARAMS_KEY]).compute()
        ground_truth = self._validation_ds[GROUND_TRUTH_KEY].load()
        loss = estimator.loss_for_estimate(estimate.values, ground_truth.values)
        scores = estimator.score_estimate(estimate.values, ground_truth.values)
        return Validation(loss, scores)
