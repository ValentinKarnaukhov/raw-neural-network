import numpy as np

import serialization
from loss_functions import mean_square_error, mean_square_error_derivative


class NeuralNetwork:

    def __init__(self, scaler):
        self.layers = []
        self.scaler = scaler
        self.logging = False
        self.errors = np.empty((0, 1))

    @classmethod
    def from_file(cls, file_name):
        return serialization.deserialize(file_name)

    def add_layer(self, layer):
        self.layers.append(layer)

    # main run function
    def predict(self, input_data):
        result = self.scaler.transform(input_data.reshape(len(input_data), -1)).reshape(input_data.shape)
        for layer in self.layers:
            result = layer.forward_propagation(result)
        return self.scaler.inverse_transform(result.reshape(len(result), -1)).reshape(result.shape)

    def train(self, input_dataset, validation_dataset, epochs, learning_rate):
        for epoch in range(epochs):
            dataset_length = len(input_dataset)
            error = None
            for data_index in range(dataset_length):
                input_data = input_dataset[data_index]
                reshaped_input_data = input_data.reshape(len(input_data), -1)
                input_data = self.scaler.transform(reshaped_input_data).reshape(input_data.shape)

                actual_result = input_data
                for layer in self.layers:
                    actual_result = layer.forward_propagation(actual_result)

                validation_data = validation_dataset[data_index]
                reshaped_validation_data = validation_data.reshape(len(validation_data), -1)
                validation_data = self.scaler.transform(reshaped_validation_data).reshape(validation_data.shape)

                error = mean_square_error(validation_data, actual_result)
                output_gradient = mean_square_error_derivative(validation_data, actual_result).reshape(1, -1)
                for layer in reversed(self.layers):
                    output_gradient = layer.backward_propagation(output_gradient, learning_rate)

            self.errors = np.append(self.errors, error)
            if self.logging:
                print("Epoch: ", epoch, "Error: ", error)

    def save(self, file_name):
        serialization.serialize(file_name, self)
