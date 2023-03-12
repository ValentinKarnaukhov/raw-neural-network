import numpy as np

import serialization
from function.loss_functions import mean_square_error, mean_square_error_derivative


class NeuralNetwork:

    def __init__(self):
        self.layers = []
        self.logging = False
        self.errors = np.empty((0, 1))

    @classmethod
    def from_file(cls, file_name):
        return serialization.deserialize(file_name)

    def add_layer(self, layer):
        self.layers.append(layer)

    def predict(self, input_data):
        result = input_data
        for layer in self.layers:
            result = layer.forward_propagation(result)
        return result

    def predict_bulk(self, input_dataset):
        result = []
        for input_data in input_dataset:
            prediction_result = self.predict(input_data)
            result.append(prediction_result)
        return np.array(result)

    def train(self, input_dataset, validation_dataset, epochs, learning_rate):
        if self.logging:
            print("Training has been started")

        for epoch in range(epochs):
            dataset_length = len(input_dataset)
            error = 0
            for data_index in range(dataset_length):
                input_data = input_dataset[data_index]

                actual_result = self.predict(input_data)
                validation_data = validation_dataset[data_index]

                error = mean_square_error(validation_data, actual_result)
                output_gradient = mean_square_error_derivative(validation_data, actual_result).reshape(1, -1)
                for layer in reversed(self.layers):
                    output_gradient = layer.backward_propagation(output_gradient, learning_rate)

            self.errors = np.append(self.errors, error)
            avg_error = self.errors.mean()
            if self.logging:
                print('epoch: %d/%d   error: %f' % (epoch + 1, epochs, avg_error))

    def save(self, file_name):
        serialization.serialize(file_name, self)
