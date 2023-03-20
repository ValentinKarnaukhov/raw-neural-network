import numpy as np

from layer.layer import Layer


class ConvolutionalLayer(Layer):

    # kernel_shape - (n, m, k), n - number, m x k - size
    # input_shape - (n, m, k), n - number, m x k - size
    def __init__(self, kernel_shape, input_shape):
        self.kernel_shape = kernel_shape
        self.input_shape = input_shape
        self.output_shape = (
            kernel_shape[0], input_shape[1] - kernel_shape[1] + 1, input_shape[2] - kernel_shape[2] + 1)
        self.weights = np.random.rand(kernel_shape[0], kernel_shape[1], kernel_shape[2]) - 0.5

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.zeros(self.output_shape)
        for kernel_index in range(self.kernel_shape[0]):
            for i in range(self.output_shape[1]):
                for j in range(self.output_shape[2]):
                    receptive_field = input_data[i:i + self.kernel_shape[1], j:j + self.kernel_shape[2]]
                    convolved_values = np.multiply(receptive_field, self.weights[kernel_index])
                    self.output[kernel_index, i, j] += np.sum(convolved_values)
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        input_error = np.zeros(self.input_shape)
        weights_error = np.zeros(self.weights.shape)
        for kernel_index in range(self.kernel_shape[0]):
            for i in range(self.output_shape[0]):
                for j in range(self.output_shape[1]):
                    receptive_field = self.input[i:i + self.kernel_shape[1], j:j + self.kernel_shape[2]]
                    weights_error[kernel_index] += np.multiply(receptive_field, output_error[kernel_index, i, j])
                    input_error[i:i + self.kernel_shape[1], j:j + self.kernel_shape[2]] += np.multiply(
                        output_error[kernel_index, i, j], self.weights[kernel_index])
        self.weights -= learning_rate * weights_error
        return input_error
