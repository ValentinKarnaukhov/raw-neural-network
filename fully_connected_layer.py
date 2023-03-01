from layer import Layer
import numpy as np


class FullyConnectedLayer(Layer):

    # input_size - size of input vector
    # output_size - size of output vector
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    # def __init__(self, weights, bias):
    #     self.weights = weights
    #     self.bias = bias

    # apply scalar product
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward_propagation(self, output_gradient, learning_rate):
        input_error = np.dot(output_gradient, self.weights.T)
        weights_error = np.dot(self.input.T, output_gradient)
        self.weights = self.weights - learning_rate * weights_error
        return input_error
