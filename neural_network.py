import numpy as np

from loss_functions import mean_square_error, mean_square_error_derivative


class NeuralNetwork:

    def __init__(self, scaler):
        self.layers = []
        self.scaler = scaler

    def add_layer(self, layer):
        self.layers.append(layer)

    # main run function
    def predict(self, input_data):
        result = self.scaler.transform(input_data.reshape(len(input_data), -1))
        # result = input_data
        for layer in self.layers:
            result = layer.forward_propagation(result)

        # return result
        return self.scaler.inverse_transform(result)

    def train(self, input_dataset, target_dataset, epochs, learning_rate):
        for epoch in range(epochs):
            dataset_length = len(input_dataset)
            for data_index in range(dataset_length):
                input_data = input_dataset[data_index]
                input_data = self.scaler.transform(input_data.reshape(len(input_data), -1)).reshape(input_data.shape)

                actual_result = input_data
                for layer in self.layers:
                    actual_result = layer.forward_propagation(actual_result)

                target_data = target_dataset[data_index]
                target_data = self.scaler.transform(target_data.reshape(len(target_data), -1)).reshape(target_data.shape)

                error = mean_square_error(target_data, actual_result)
                output_gradient = mean_square_error_derivative(target_data, actual_result)
                for layer in reversed(self.layers):
                    output_gradient = layer.backward_propagation(output_gradient, learning_rate)
