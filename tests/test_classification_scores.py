import numpy as np
import pytest
from numpy.typing import NDArray
from sklearn.metrics import confusion_matrix

from rattlinbog.scoring.classification import ClassScore0th, ClassScore1st, score_zero_order, \
    score_first_order


def make_mask(description):
    values = [int(e) for e in description]
    return np.array(values, dtype=int)


def as_a(*values) -> NDArray:
    return np.asarray(values)


TEST_SETUP = {
    "perfect true positives": (make_mask("1111"), make_mask("1111"),
                               ClassScore0th(TP=as_a(4), TN=as_a(0), FP=as_a(0), FN=as_a(0)),
                               ClassScore1st(TPR=as_a(1.0), TNR=as_a(0.0), PPV=as_a(1.0))),
    # ClassScore2nd(F1=as_a(1.0), BA=as_a(0.0), MCC=as_a(1.0))),
    "perfect false positives": (make_mask("0000"), make_mask("0000"),
                                ClassScore0th(TP=as_a(4), TN=as_a(0), FP=as_a(0), FN=as_a(0)),
                                ClassScore1st(TPR=as_a(1.0), TNR=as_a(0.0), PPV=as_a(1.0))),
    # ClassScore2nd(F1=as_a(1.0), BA=as_a(0.0), MCC=as_a(1.0))),
    "perfectly correct": (make_mask("0011"), make_mask("0011"),
                          ClassScore0th(TP=as_a(2, 2), TN=as_a(2, 2), FP=as_a(0, 0), FN=as_a(0, 0)),
                          ClassScore1st(TPR=as_a(1.0, 1.0), TNR=as_a(1.0, 1.0), PPV=as_a(1.0, 1.0))),
    # ClassScore2nd(F1=as_a(1.0), BA=as_a(0.0), MCC=as_a(1.0))),
    "perfectly incorrect": (make_mask("0011"), make_mask("1100"),
                            ClassScore0th(TP=as_a(0, 0), TN=as_a(0, 0), FP=as_a(2, 2), FN=as_a(2, 2)),
                            ClassScore1st(TPR=as_a(0.0, 0.0), TNR=as_a(0.0, 0.0), PPV=as_a(0.0, 0.0))),
    # ClassScore2nd(F1=as_a(1.0), BA=as_a(0.0), MCC=as_a(1.0))),
    "mostly correct": (make_mask("0011"), make_mask("0111"),
                       ClassScore0th(TP=as_a(1, 2), TN=as_a(2, 1), FP=as_a(0, 1), FN=as_a(1, 0)),
                       ClassScore1st(TPR=as_a(0.5, 1.0), TNR=as_a(1.0, 0.5), PPV=as_a(1.0, 2 / 3))),
    # ClassScore2nd(F1=as_a(1.0), BA=as_a(0.0), MCC=as_a(1.0))),
    "mostly incorrect": (make_mask("1111"), make_mask("0001"),
                         ClassScore0th(TP=as_a(0, 1), TN=as_a(1, 0), FP=as_a(3, 0), FN=as_a(0, 3)),
                         ClassScore1st(TPR=as_a(0.75, 0.25), TNR=as_a(0.25, 0.75), PPV=as_a(0.0, 1.0))),
    # ClassScore2nd(F1=as_a(1.0), BA=as_a(0.0), MCC=as_a(1.0))),
}


@pytest.fixture(params=TEST_SETUP.keys())
def active_setup(request):
    return request.param


@pytest.fixture
def cm(active_setup):
    setup = TEST_SETUP[active_setup]
    return confusion_matrix(setup[0], setup[1])


@pytest.fixture
def expected_score_0th(active_setup):
    return TEST_SETUP[active_setup][2]


@pytest.fixture
def expected_score_1st(active_setup):
    return TEST_SETUP[active_setup][3]


# @pytest.fixture
# def expected_score_2nd(active_setup):
#     return TEST_SETUP[active_setup][3]


def test_calculating_zero_order_scores(cm, expected_score_0th):
    assert score_zero_order(cm) == expected_score_0th


def test_calculating_first_order_scores(expected_score_0th, expected_score_1st):
    assert score_first_order(expected_score_0th) == expected_score_1st

# def test_calculating_second_order_scores(cm, expected_score_1st, expected_score_2nd):
#     assert score_confusion_matrix_second_order(cm, expected_score_1st) == expected_score_2nd
