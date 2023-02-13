from dataclasses import dataclass

import numpy as np
from numpy._typing import NDArray


@dataclass
class ClassScore1st:
    TPR: NDArray
    TNR: NDArray
    PPV: NDArray

    def __eq__(self, other: "ClassScore1st") -> bool:
        return np.array_equal(self.TPR, other.TPR) \
            and np.array_equal(self.TNR, other.TNR) \
            and np.array_equal(self.PPV, other.PPV)


def score_confusion_matrix_first_order(confusion_matrix: NDArray) -> ClassScore1st:
    tp = np.diag(confusion_matrix)
    fp = confusion_matrix.sum(axis=0) - tp
    fn = confusion_matrix.sum(axis=1) - tp
    tn = confusion_matrix.sum() - (fp + fn + tp)

    p = tp + fn
    n = tn + fp

    has_p = p > 0
    has_not_p = ~has_p
    tpr = np.empty_like(tp, dtype=np.float32)
    tpr[has_p] = tp[has_p] / p[has_p]
    tpr[has_not_p] = 1.0 - tn[has_not_p] / n[has_not_p]

    has_n = n > 0
    has_not_n = ~has_n
    tnr = np.empty_like(tp, dtype=np.float32)
    tnr[has_n] = tn[has_n] / n[has_n]
    tnr[has_not_n] = 1 - tp[has_not_n] / p[has_not_n]

    ppv = tp / (tp + fp)

    return ClassScore1st(tpr, tnr, ppv)
