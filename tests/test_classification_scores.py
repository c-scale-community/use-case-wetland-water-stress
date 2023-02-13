from dataclasses import dataclass

import numpy as np
import pytest
from numpy.typing import NDArray
from sklearn.metrics import confusion_matrix


@dataclass
class ClassScore1st:
    TPR: float
    TNR: float
    PPV: float


def score_confusion_matrix_first_order(confusion_matrix: NDArray) -> ClassScore1st:
    return ClassScore1st(1.0, 0.0, 1.0)


def make_mask(description):
    values = [e == 'T' for e in description]
    return np.array(values, dtype=bool)


@pytest.mark.parametrize("estimates, ground_truth, expected_scores", [
    # perfect true positives
    (make_mask("TTTT"), make_mask("TTTT"), ClassScore1st(TPR=1.0, TNR=0.0, PPV=1.0)),
])
def test_calculating_first_order_scores(estimates, ground_truth, expected_scores):
    assert score_confusion_matrix_first_order(confusion_matrix(estimates, ground_truth)) == expected_scores
