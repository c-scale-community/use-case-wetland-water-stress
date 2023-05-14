import numpy as np
from numba import njit, prange, vectorize, uint32, boolean
from numpy._typing import NDArray


def confusion_matrix_fast_binary(ground_truth: NDArray, predicted: NDArray) -> NDArray:
    cm = np.zeros((2, 2), dtype=np.uint32)
    binary_confusion_matrix_numba(ground_truth, predicted, cm)
    return cm



@njit(parallel=True)
def binary_confusion_matrix_numba(ground_truth: NDArray, predicted: NDArray, out: NDArray) -> None:
    tn = 0
    fp = 0
    fn = 0
    tp = 0
    for i in prange(ground_truth.shape[0]):
        a = ground_truth[i]
        p = predicted[i]
        tn += a == p == 0
        fp += a == 0 and p == 1
        fn += a == 1 and p == 0
        tp += a == p == 1

    out[0, 0] = tn
    out[0, 1] = fp
    out[1, 0] = fn
    out[1, 1] = tp
