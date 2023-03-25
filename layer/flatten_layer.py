from layer.layer import Layer


class FlattenLayer(Layer):

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = input_data.flatten().reshape((1, -1))
        return self.output

    def backward_propagation(self, output_gradient, learning_rate):
        return output_gradient.reshape(self.input.shape)
