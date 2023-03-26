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


class ReLU(ActivationFunction):

    @staticmethod
    def function(x):
        return np.maximum(0, x)

    @staticmethod
    def function_derivative(x):
        data = x.flatten()
        data[data <= 0] = 0
        data[data > 0] = 1
        return data.reshape(x.shape)
