import numpy as np


class ActivationFunction:

    @staticmethod
    def function(x):
        raise NotImplementedError

    @staticmethod
    def function_derivative(x):
        raise NotImplementedError


class Sigmoid(ActivationFunction):

    @staticmethod
    def function(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def function_derivative(x):
        return Sigmoid.function(x) * (1 - Sigmoid.function(x))


class Tanh(ActivationFunction):

    @staticmethod
    def function(x):
        return np.tanh(x)

    @staticmethod
    def function_derivative(x):
        return 1 - np.tanh(x) ** 2
