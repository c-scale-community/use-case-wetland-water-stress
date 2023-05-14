import numpy as np
from numba import njit
from numpy._typing import NDArray


def confusion_matrix_fast_binary(ground_truth: NDArray, predicted: NDArray) -> NDArray:
    cm = np.zeros((2, 2), dtype=np.uint32)
    binary_confusion_matrix_numba(ground_truth, predicted, cm)
    return cm


@njit(parallel=True)
def binary_confusion_matrix_numba(ground_truth: NDArray, predicted: NDArray, out: NDArray) -> None:
    for i in range(ground_truth.shape[0]):
        a = ground_truth[i]
        p = predicted[i]
        out[0, 0] += a == p == 0
        out[0, 1] += a == 0 and p == 1
        out[1, 0] += a == 1 and p == 0
        out[1, 1] += a == p == 1
