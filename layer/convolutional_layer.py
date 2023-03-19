import numpy as np

from layer.layer import Layer


class ConvolutionalLayer(Layer):

    def __init__(self, filters, kernel_shape, input_shape):
        self.kernel_shape = kernel_shape
        self.input_shape = input_shape
        self.output_shape = (input_shape[0]-kernel_shape[0]+1, input_shape[1]-kernel_shape[1]+1, filters)
        self.weights = np.random.rand(kernel_shape[0], kernel_shape[1], filters) - 0.5

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.zeros(self.output_shape)
        pass

    def backward_propagation(self, output_error, learning_rate):
        pass