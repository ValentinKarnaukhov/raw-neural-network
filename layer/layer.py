class Layer:

    # input - input data of layer
    # output - output data of layer
    def __init__(self):
        self.input = None
        self.output = None

    def forward_propagation(self, input_data):
        raise NotImplementedError

    def backward_propagation(self, output_gradient, learning_rate):
        raise NotImplementedError

