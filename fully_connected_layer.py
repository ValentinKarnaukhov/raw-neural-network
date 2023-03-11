from layer import Layer
import numpy as np


class FullyConnectedLayer(Layer):

    # input_size - size of input vector
    # output_size - size of output vector
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size)
        self.bias = np.random.rand(1, output_size)

    # def __init__(self, weights, bias):
    #     self.weights = weights
    #     self.bias = bias

    # apply scalar product
    def forward_propagation(self, input_data):
        self.input = input_data.flatten().reshape(1, -1)
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output[0]

    def backward_propagation(self, output_gradient, learning_rate):
        input_error = np.dot(output_gradient, self.weights.T)
        weights_error = np.dot(self.input.T, output_gradient)
        self.weights = self.weights - learning_rate * weights_error
        self.bias = self.bias - learning_rate * output_gradient
        return input_error
