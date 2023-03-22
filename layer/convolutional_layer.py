import numpy as np

from layer.layer import Layer


class ConvolutionalLayer(Layer):

    # kernel_shape - (n, m, k), n - number, m x k - size
    # input_shape - (p, q, l), l - channels, p x q - size
    # weights - (n, m, k, l)
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
        self.input = np.pad(input_data, [(self.up_bottom_padding, self.up_bottom_padding),
                                         (self.left_right_padding, self.left_right_padding), (0, 0)])

        for kernel_index in range(self.kernel_shape[0]):
            for i in range(self.input.shape[0] - self.kernel_shape[1] + 1):
                for j in range(self.input.shape[1] - self.kernel_shape[2] + 1):
                    receptive_field = self.input[i:i + self.kernel_shape[1], j:j + self.kernel_shape[2], :]
                    convolved_values = np.multiply(receptive_field, self.weights[kernel_index])
                    self.output[i, j, kernel_index] += np.sum(convolved_values)
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        input_error = np.zeros(self.input.shape)
        padded_output_error = np.pad(output_error, [(self.up_bottom_padding, self.up_bottom_padding),
                                                    (self.left_right_padding, self.left_right_padding), (0, 0)])
        rotated_weights = np.flip(self.weights, axis=(1, 2))

        for kernel_index in range(self.kernel_shape[0]):
            for i in range(self.output.shape[0] - self.kernel_shape[1] + 1):
                for j in range(self.output.shape[1] - self.kernel_shape[2] + 1):
                    output_error_val = padded_output_error[i:i + self.kernel_shape[1],
                                       j:j + self.kernel_shape[2], kernel_index]
                    input_error[i:i + self.kernel_shape[1], j:j + self.kernel_shape[2], :] += (
                            output_error_val[:, :, np.newaxis] * rotated_weights[kernel_index])
                    self.weights[kernel_index] -= (
                            learning_rate * np.multiply(output_error_val[:, :, np.newaxis],
                                                        self.input[i:i + self.kernel_shape[1],
                                                        j:j + self.kernel_shape[2], :])
                    )

        return input_error
