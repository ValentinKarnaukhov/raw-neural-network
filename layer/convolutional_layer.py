import numpy as np

from layer.layer import Layer


class ConvolutionalLayer(Layer):

    # kernel_shape - dimensions (n, m, k), n - amount of filters, m x k - size of filter
    # input_shape - dimensions (p, q, l), l - amount of input channels, p x q - size
    # weights - dimensions (n, m, k, l), n - amount of filters, m x k - size of filter, l - kernels per filter
    def __init__(self, kernel_shape, input_shape):
        self.kernel_shape = kernel_shape
        self.input_shape = input_shape
        self.left_right_padding = self.kernel_shape[1] // 2
        self.up_bottom_padding = self.kernel_shape[2] // 2
        self.output_shape = (self.input_shape[0] + 2 * self.left_right_padding - self.kernel_shape[1] + 1,
                             self.input_shape[1] + 2 * self.up_bottom_padding - self.kernel_shape[2] + 1,
                             self.kernel_shape[0])
        self.weights = np.random.rand(kernel_shape[0], kernel_shape[1], kernel_shape[2], self.input_shape[2]) - 0.5

    def forward_propagation(self, input_data):
        self.output = np.zeros(self.output_shape)
        self.input = input_data
        padded_input = np.pad(input_data, [(self.up_bottom_padding, self.up_bottom_padding),
                                         (self.left_right_padding, self.left_right_padding), (0, 0)])

        for kernel_index in range(self.kernel_shape[0]):
            for i in range(padded_input.shape[0] - self.kernel_shape[1] + 1):
                for j in range(padded_input.shape[1] - self.kernel_shape[2] + 1):
                    receptive_field = padded_input[i:i + self.kernel_shape[1], j:j + self.kernel_shape[2], :]
                    convolved_values = np.multiply(receptive_field, self.weights[kernel_index])
                    self.output[i, j, kernel_index] += np.sum(convolved_values)
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        input_error = np.zeros(self.input_shape)
        padded_input = np.pad(self.input, [(self.up_bottom_padding, self.up_bottom_padding),
                                           (self.left_right_padding, self.left_right_padding), (0, 0)])

        for kernel_index in range(self.kernel_shape[0]):
            for i in range(padded_input.shape[0] - self.kernel_shape[1] + 1):
                for j in range(padded_input.shape[1] - self.kernel_shape[2] + 1):
                    input_error[i:i + self.kernel_shape[1], j:j + self.kernel_shape[2], :] += \
                        self.weights[kernel_index] * output_error[i, j]


        return input_error
