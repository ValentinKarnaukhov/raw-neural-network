import numpy as np

from layer.layer import Layer


class FullyConnectedLayer(Layer):

    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    @classmethod
    def with_initial_weights(cls, weights, bias):
        weights = weights
        bias = bias
        return cls(weights, bias)

    @classmethod
    def with_random_weights(cls, input_size, output_size):
        weights = np.random.rand(input_size, output_size) * 2 - 1
        bias = np.random.rand(1, output_size) * 2 - 1
        return cls(weights, bias)

    # apply scalar product
    def forward_propagation(self, input_data):
        self.input = input_data.flatten().reshape(1, -1)
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output[0]

    def backward_propagation(self, output_gradient, learning_rate):
        input_error = np.dot(output_gradient, self.weights.T)
        weights_error = np.dot(self.input.T, output_gradient)
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_gradient
        return input_error
