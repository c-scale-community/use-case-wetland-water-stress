from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class ClassScore0th:
    TP: NDArray
    TN: NDArray
    FP: NDArray
    FN: NDArray

    def __eq__(self, other: "ClassScore0th") -> bool:
        return np.array_equal(self.TP, other.TP) \
            and np.array_equal(self.TN, other.TN) \
            and np.array_equal(self.FP, other.FP) \
            and np.array_equal(self.FN, other.FN)


def score_zero_order(confusion_matrix: NDArray) -> ClassScore0th:
    tp = np.diag(confusion_matrix)
    fp = confusion_matrix.sum(axis=0) - tp
    fn = confusion_matrix.sum(axis=1) - tp
    tn = confusion_matrix.sum() - (fp + fn + tp)
    return ClassScore0th(tp, tn, fp, fn)


@dataclass
class ClassScore1st:
    TPR: NDArray
    TNR: NDArray
    PPV: NDArray

    def __eq__(self, other: "ClassScore1st") -> bool:
        return np.array_equal(self.TPR, other.TPR) \
            and np.array_equal(self.TNR, other.TNR) \
            and np.array_equal(self.PPV, other.PPV)


def score_first_order(zero_order: ClassScore0th) -> ClassScore1st:
    p = zero_order.TP + zero_order.FN
    n = zero_order.TN + zero_order.FP

    has_p = p > 0
    has_not_p = ~has_p
    tpr = np.empty_like(zero_order.TP, dtype=np.float32)
    tpr[has_p] = zero_order.TP[has_p] / p[has_p]
    tpr[has_not_p] = 1.0 - zero_order.TN[has_not_p] / n[has_not_p]

    has_n = n > 0
    has_not_n = ~has_n
    tnr = np.empty_like(zero_order.TN, dtype=np.float32)
    tnr[has_n] = zero_order.TN[has_n] / n[has_n]
    tnr[has_not_n] = 1 - zero_order.TP[has_not_n] / p[has_not_n]

    t_and_f_ps = zero_order.TP + zero_order.FP
    has_ppv = t_and_f_ps > 0
    ppv = np.empty_like(zero_order.TP, dtype=np.float32)
    ppv[has_ppv] = zero_order.TP[has_ppv] / t_and_f_ps[has_ppv]
    ppv[~has_ppv] = 0.0

    return ClassScore1st(tpr, tnr, ppv)


@dataclass
class ClassScore2nd:
    F1: NDArray
    BA: NDArray

    def __eq__(self, other: "ClassScore2nd") -> bool:
        return np.array_equal(self.F1, other.F1) \
            and np.array_equal(self.BA, other.BA)


def score_second_order(first_order: ClassScore1st) -> ClassScore2nd:
    f1 = np.empty_like(first_order.TPR)
    precision_and_rate = first_order.PPV + first_order.TPR
    is_valid = precision_and_rate > 0
    f1[is_valid] = 2.0 * (first_order.PPV[is_valid] * first_order.TPR[is_valid]) / precision_and_rate[is_valid]
    f1[~is_valid] = 0.0
    ba = (first_order.TPR + first_order.TNR) / 2.0
    return ClassScore2nd(f1, ba)
