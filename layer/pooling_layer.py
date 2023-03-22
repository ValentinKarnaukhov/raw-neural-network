import numpy as np

from layer.layer import Layer


class MaxPooling(Layer):

    def __init__(self, pooling_size):
        self.pooling_size = pooling_size

    # input_data - (m, k, n), n - channels, m x k - size
    def forward_propagation(self, input_data):
        self.input = input_data
        output_height = self.input.shape[0] // self.pooling_size[0]
        output_width = self.input.shape[1] // self.pooling_size[1]
        self.output = np.zeros((output_height, output_width, self.input.shape[2]))
        for input_data_index in range(self.input.shape[2]):
            for h in range(output_height):
                for w in range(output_width):
                    vertical_start = h * self.pooling_size[0]
                    vertical_end = vertical_start + self.pooling_size[0]
                    horizontal_start = w * self.pooling_size[1]
                    horizontal_end = horizontal_start + self.pooling_size[1]

                    input_slice = input_data[vertical_start:vertical_end,
                                  horizontal_start:horizontal_end, input_data_index]
                    self.output[h, w, input_data_index] = np.max(input_slice)
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        input_error = np.zeros_like(self.input)
        for output_data_index in range(self.output.shape[2]):
            for h in range(self.output.shape[0]):
                for w in range(self.output.shape[1]):
                    vertical_start = h * self.pooling_size[0]
                    vertical_end = vertical_start + self.pooling_size[0]
                    horizontal_start = w * self.pooling_size[1]
                    horizontal_end = horizontal_start + self.pooling_size[1]

                    input_slice = self.input[vertical_start:vertical_end, horizontal_start:horizontal_end,output_data_index]
                    mask = (input_slice == self.output[h, w, output_data_index])
                    input_error[vertical_start:vertical_end, horizontal_start:horizontal_end,
                    output_data_index] += mask * output_error[h, w, output_data_index]
        return input_error
