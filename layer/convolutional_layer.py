import numpy as np
from scipy import signal

from layer.layer import Layer
from util.execution_time import execution_time


class ConvolutionalLayer(Layer):

    # kernel_shape - dimensions (m, k), m x k - size of filter
    # input_shape - dimensions (p, q, l), l - amount of input channels, p x q - size
    # weights - dimensions (n, m, k, l), n - amount of filters, m x k - size of filter, l - kernels per filter
    def __init__(self, kernel_shape, amount_of_filters, input_shape):
        super().__init__()
        self.kernel_shape = kernel_shape
        self.input_shape = input_shape
        self.amount_of_filters = amount_of_filters
        self.left_right_padding = self.kernel_shape[0] // 2
        self.up_bottom_padding = self.kernel_shape[1] // 2
        self.output_shape = (self.input_shape[0] + 2 * self.left_right_padding - self.kernel_shape[0] + 1,
                             self.input_shape[1] + 2 * self.up_bottom_padding - self.kernel_shape[1] + 1,
                             amount_of_filters)
        self.weights = np.random.rand(amount_of_filters, kernel_shape[0], kernel_shape[1], self.input_shape[-1]) - 0.5
        self.bias = np.random.rand(amount_of_filters) - 0.5

    @execution_time
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.zeros(self.output_shape)
        padded_input = np.pad(input_data, [(self.up_bottom_padding, self.up_bottom_padding),
                                           (self.left_right_padding, self.left_right_padding), (0, 0)])

        for filter_index in range(self.amount_of_filters):
            for channel_index in range(self.input_shape[-1]):
                self.output[:, :, filter_index] += signal.correlate2d(padded_input[:, :, channel_index],
                                                                      self.weights[filter_index, :, :, channel_index],
                                                                      'valid') + self.bias[filter_index]
        return self.output

    @execution_time
    def backward_propagation(self, output_gradient, learning_rate):
        input_gradient = np.zeros(self.input_shape)
        weights_gradients = np.zeros(
            (self.amount_of_filters, self.kernel_shape[0], self.kernel_shape[1], self.input_shape[-1]))
        bias_gradient = np.zeros(self.amount_of_filters)

        for filter_index in range(self.amount_of_filters):
            for channel_index in range(self.input_shape[-1]):
                padded_input_gradient = signal.convolve2d(output_gradient[:, :, filter_index],
                                                          self.weights[filter_index, :, :, channel_index],
                                                          'full')
                input_gradient[:, :, channel_index] += padded_input_gradient[
                                                       self.left_right_padding:-self.left_right_padding,
                                                       self.up_bottom_padding:-self.up_bottom_padding]
                weights_gradients[filter_index, :, :, channel_index] = signal.correlate2d(
                    self.input[:, :, channel_index], output_gradient[:, :, filter_index],
                    'valid')
            bias_gradient[filter_index] = self.amount_of_filters * np.sum(output_gradient[:, :, filter_index])

        self.weights -= learning_rate * weights_gradients
        self.bias -= learning_rate * bias_gradient
        return input_gradient
