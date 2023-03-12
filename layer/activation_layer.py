from layer.layer import Layer


class ActivationLayer(Layer):

    # activation - activate function
    def __init__(self, activation_function):
        self.activation_function = activation_function

    # apply activation function to input_data
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation_function.function(self.input)
        return self.output

    def backward_propagation(self, output_gradient, learning_rate):
        return self.activation_function.function_derivative(self.input) * output_gradient
