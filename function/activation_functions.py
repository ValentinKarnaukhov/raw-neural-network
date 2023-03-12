import numpy as np


class ActivationFunction:

    @classmethod
    def function(cls, x):
        raise NotImplementedError

    @classmethod
    def function_derivative(cls, x):
        raise NotImplementedError


class Sigmoid(ActivationFunction):

    @classmethod
    def function(cls, x):
        return 1 / (1 + np.exp(-x))

    @classmethod
    def function_derivative(cls, x):
        return cls.function(x) * (1 - cls.function(x))


class Tanh(ActivationFunction):

    @classmethod
    def function(cls, x):
        return np.tanh(x)

    @classmethod
    def function_derivative(cls, x):
        return 1 - np.tanh(x) ** 2
