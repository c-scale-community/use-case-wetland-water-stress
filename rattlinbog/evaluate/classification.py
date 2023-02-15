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
    FPR: NDArray
    FNR: NDArray
    PPV: NDArray

    def __eq__(self, other: "ClassScore1st") -> bool:
        return np.array_equal(self.TPR, other.TPR) \
            and np.array_equal(self.TNR, other.TNR) \
            and np.array_equal(self.PPV, other.PPV)


def score_first_order(zero_order: ClassScore0th) -> ClassScore1st:
    p = zero_order.TP + zero_order.FN
    n = zero_order.TN + zero_order.FP

    tpr = zero_order.TP / p
    tnr = zero_order.TN / n
    fpr = zero_order.FP / n
    fnr = zero_order.FN / p

    no_p = p == 0
    no_n = n == 0

    tpr[no_p] = 0.0
    fnr[no_p] = 0.0
    tnr[no_n] = 0.0
    fpr[no_n] = 0.0


    t_and_f_ps = zero_order.TP + zero_order.FP
    valid_ppv = t_and_f_ps != 0

    ppv = np.zeros_like(t_and_f_ps)
    ppv[valid_ppv] = zero_order.TP[valid_ppv] / t_and_f_ps[valid_ppv]

    return ClassScore1st(tpr, tnr, fpr, fnr, ppv)


@dataclass
class ClassScore2nd:
    F1: NDArray
    BA: NDArray

    def __eq__(self, other: "ClassScore2nd") -> bool:
        return np.array_equal(self.F1, other.F1) \
            and np.array_equal(self.BA, other.BA)


def score_second_order(first_order: ClassScore1st) -> ClassScore2nd:
    precision_and_rate = first_order.PPV + first_order.TPR
    valid_f1 = precision_and_rate != 0

    f1 = np.zeros_like(precision_and_rate)
    f1[valid_f1] = 2.0 * (first_order.PPV[valid_f1] * first_order.TPR[valid_f1]) / precision_and_rate[valid_f1]

    ba = (first_order.TPR + first_order.TNR) / 2.0
    return ClassScore2nd(f1, ba)
