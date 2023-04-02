import numpy as np


class LossFunction:

    @staticmethod
    def function(target_result, actual_result):
        raise NotImplementedError

    @staticmethod
    def function_derivative(target_result, actual_result):
        raise NotImplementedError


class MeanSquareError(LossFunction):

    @staticmethod
    def function(target_result, actual_result):
        return np.mean(np.power(target_result - actual_result, 2))

    @staticmethod
    def function_derivative(target_result, actual_result):
        return 2 * (actual_result - target_result) / target_result.shape[0]
