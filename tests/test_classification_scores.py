from dataclasses import dataclass

import numpy as np
import pytest
from numpy.typing import NDArray
from sklearn.metrics import confusion_matrix


@dataclass
class ClassScore1st:
    TPR: NDArray
    TNR: NDArray
    PPV: NDArray

    def __eq__(self, other: "ClassScore1st") -> bool:
        return np.all(self.TPR == other.TPR) \
            and np.all(self.TNR == other.TNR) \
            and np.all(self.PPV == other.PPV)


def score_confusion_matrix_first_order(confusion_matrix: NDArray) -> ClassScore1st:
    tp = np.diag(confusion_matrix)
    fp = confusion_matrix.sum(axis=0) - tp
    fn = confusion_matrix.sum(axis=1) - tp
    tn = confusion_matrix.sum() - (fp + fn + tp)

    p = tp + fn
    n = tn + fp

    has_p = p > 0
    has_not_p = ~has_p
    tpr = np.empty_like(tp)
    tpr[has_p] = tp[has_p] / p[has_p]
    tpr[has_not_p] = 1.0 - tn[has_not_p] / n[has_not_p]

    has_n = n > 0
    has_not_n = ~has_n
    tnr = np.empty_like(tp)
    tnr[has_n] = tn[has_n] / n[has_n]
    tnr[has_not_n] = 1 - tp[has_not_n] / p[has_not_n]

    ppv = tp / (tp + fp)

    return ClassScore1st(tpr, tnr, ppv)


def make_mask(description):
    values = [int(e) for e in description]
    return np.array(values, dtype=int)


def as_a(*values) -> NDArray:
    return np.asarray(values)


@pytest.fixture(params=["perfect true positives", "perfect false positives", "perfectly correct"])
def result_setup(request):
    return request.param


@pytest.fixture
def cm(result_setup):
    if result_setup == "perfect true positives":
        return confusion_matrix(make_mask("1111"), make_mask("1111"))
    if result_setup == "perfect false positives":
        return confusion_matrix(make_mask("1111"), make_mask("1111"))
    if result_setup == "perfectly correct":
        return confusion_matrix(make_mask("0011"), make_mask("0011"))


@pytest.fixture
def expected_score(result_setup):
    if result_setup == "perfect true positives":
        return ClassScore1st(TPR=as_a(1.0), TNR=as_a(0.0), PPV=as_a(1.0))
    if result_setup == "perfect false positives":
        return ClassScore1st(TPR=as_a(1.0), TNR=as_a(0.0), PPV=as_a(1.0))
    if result_setup == "perfectly correct":
        return ClassScore1st(TPR=as_a(1.0, 1.0), TNR=as_a(1.0, 1.0), PPV=as_a(1.0, 1.0))


def test_calculating_first_order_scores(cm, expected_score):
    assert score_confusion_matrix_first_order(cm) == expected_score
