import numpy as np


def mean_square_error(target_result, actual_result):
    return np.mean(np.power(target_result - actual_result, 2))


def mean_square_error_derivative(target_result, actual_result):
    return actual_result - target_result
