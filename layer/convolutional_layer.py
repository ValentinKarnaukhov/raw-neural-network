import numpy as np
from scipy import signal

from layer.layer import Layer
from util.execution_time import execution_time


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
        self.weights = np.random.rand(kernel_shape[0], kernel_shape[1], kernel_shape[2], self.input_shape[2]) - 1
        self.bias = np.random.rand(kernel_shape[0]) - 1

    @execution_time
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.zeros(self.output_shape)
        padded_input = np.pad(input_data, [(self.up_bottom_padding, self.up_bottom_padding),
                                           (self.left_right_padding, self.left_right_padding), (0, 0)])

        for kernel_index in range(self.kernel_shape[0]):
            for channel in range(self.input_shape[-1]):
                self.output[:, :, kernel_index] += signal.correlate2d(padded_input[:, :, channel],
                                                                      self.weights[kernel_index, :, :, channel],
                                                                      'valid') + self.bias[kernel_index]
        return self.output

    @execution_time
    def backward_propagation(self, output_error, learning_rate):
        input_error = np.zeros(self.input_shape)
        weights_gradients = np.zeros(
            (self.kernel_shape[0], self.kernel_shape[1], self.kernel_shape[1], self.input_shape[-1]))
        bias_gradient = np.zeros(self.kernel_shape[0])

        for k in range(self.kernel_shape[0]):
            for d in range(self.input_shape[-1]):
                input_error[:, :, d] += signal.convolve2d(output_error[:, :, k], self.weights[k, :, :, d], 'full')[
                                     self.left_right_padding:-self.left_right_padding,
                                     self.up_bottom_padding:-self.up_bottom_padding]
                weights_gradients[k, :, :, d] = signal.correlate2d(self.input[:, :, d], output_error[:, :, k], 'valid')
            bias_gradient[k] = self.kernel_shape[0] * np.sum(output_error[:, :, k])

        self.weights -= learning_rate * weights_gradients
        self.bias -= learning_rate * bias_gradient
        return input_error
