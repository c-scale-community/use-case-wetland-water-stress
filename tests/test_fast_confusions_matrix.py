from timeit import timeit

import numpy as np
from sklearn.metrics import confusion_matrix

from rattlinbog.evaluate.confusion_matrix_binary import confusion_matrix_fast_binary


def test_quick_fast_confusion_matrix_produces_the_same_results_as_sklearn():
    for _ in range(100):
        ground_truth = np.random.randint(0, 2, 1000) == 1
        predicted = np.random.randint(0, 2, 1000) == 1
        assert_confusion_matrices_eq(confusion_matrix_fast_binary(ground_truth, predicted),
                                     confusion_matrix(ground_truth, predicted))


def assert_confusion_matrices_eq(actual, expected):
    np.testing.assert_array_equal(actual, expected)


def test_fast_binary_confusion_matrix_performance():
    ground_truth = np.random.randint(0, 2, 100000) == 1
    predicted = np.random.randint(0, 2, 100000) == 1
    sklearn_runtime = timeit(lambda: confusion_matrix(ground_truth, predicted), number=100)
    binary_runtime = timeit(lambda: confusion_matrix_fast_binary(ground_truth, predicted), number=100)
    assert binary_runtime < sklearn_runtime * 0.5, sklearn_runtime
