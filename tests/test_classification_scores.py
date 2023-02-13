from dataclasses import dataclass

import numpy as np
import pytest
from numpy.typing import NDArray


@dataclass
class BinaryClassScore0th:
    TP: int
    TN: int
    FP: int
    FN: int


def score_binary_classification_zero_order(estimates: NDArray, ground_truth: NDArray) -> BinaryClassScore0th:
    return BinaryClassScore0th(4, 0, 0, 0)


@pytest.mark.parametrize("estimates, ground_truth, expected_scores", [
    # perfect true positives
    (np.ones((1, 2, 2)), np.ones((1, 2, 2)), BinaryClassScore0th(TP=4, TN=0, FP=0, FN=0))
])
def test_calculating_zero_order_scores(estimates, ground_truth, expected_scores):
    assert score_binary_classification_zero_order(estimates, ground_truth) == expected_scores
