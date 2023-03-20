import numpy as np

from layer.layer import Layer


class MaxPooling(Layer):

    def __init__(self, pooling_size):
        self.pooling_size = pooling_size

    def forward_propagation(self, input_data):
        self.input = input_data
        output_height = input_data.shape[0] // self.pooling_size[0]
        output_width = input_data.shape[1] // self.pooling_size[1]
        self.output = np.zeros((output_height, output_width))

        for h in range(output_height):
            for w in range(output_width):
                vertical_start = h * self.pooling_size[0]
                vertical_end = vertical_start + self.pooling_size[0]
                horizontal_start = w * self.pooling_size[1]
                horizontal_end = horizontal_start + self.pooling_size[1]

                input_slice = input_data[vertical_start:vertical_end, horizontal_start:horizontal_end]
                self.output[h, w] = np.max(input_slice)
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        input_error = np.zeros_like(self.input)
        for h in range(self.output.shape[0]):
            for w in range(self.output.shape[1]):
                vertical_start = h * self.pooling_size[0]
                vertical_end = vertical_start + self.pooling_size[0]
                horizontal_start = w * self.pooling_size[1]
                horizontal_end = horizontal_start + self.pooling_size[1]

                input_slice = self.input[vertical_start:vertical_end, horizontal_start:horizontal_end]
                mask = (input_slice == self.output[h, w])

                input_error[vertical_start:vertical_end, horizontal_start:horizontal_end] += mask * output_error[h, w]
        return input_error