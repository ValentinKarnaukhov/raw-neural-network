import numpy as np

from layer.layer import Layer


class MaxPooling(Layer):

    def __init__(self, pooling_size):
        super().__init__()
        self.pooling_size = pooling_size

    # input_data - (m, k, n), n - channels, m x k - size
    def forward_propagation(self, input_data):
        self.input = input_data
        output_height = self.input.shape[0] // self.pooling_size[0]
        output_width = self.input.shape[1] // self.pooling_size[1]
        self.output = np.zeros((output_height, output_width, self.input.shape[-1]))
        for channel_index in range(self.input.shape[-1]):
            for i in range(output_height):
                for j in range(output_width):
                    h_start = i * self.pooling_size[0]
                    h_end = h_start + self.pooling_size[0]
                    w_start = j * self.pooling_size[1]
                    w_end = w_start + self.pooling_size[1]

                    input_slice = input_data[h_start:h_end, w_start:w_end, channel_index]
                    self.output[i, j, channel_index] = np.max(input_slice)
        return self.output

    def backward_propagation(self, output_gradient, learning_rate):
        input_gradient = np.zeros_like(self.input)
        for channel_index in range(self.output.shape[-1]):
            for i in range(self.output.shape[0]):
                for j in range(self.output.shape[1]):
                    h_start = i * self.pooling_size[0]
                    h_end = h_start + self.pooling_size[0]
                    w_start = j * self.pooling_size[1]
                    w_end = w_start + self.pooling_size[1]

                    input_slice = self.input[h_start:h_end, w_start:w_end, channel_index]
                    mask = (input_slice == self.output[i, j, channel_index])
                    input_gradient[h_start:h_end, w_start:w_end, channel_index] += \
                        mask * output_gradient[i, j, channel_index]
        return input_gradient
