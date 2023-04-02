import time

import numpy as np

import plotter
import serialization


class NeuralNetwork:

    def __init__(self, loss_function):
        self.layers = []
        self.logging = False
        self.errors = np.empty((0, 1))
        self.loss_function = loss_function

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

    def train(self, input_dataset, validation_dataset, epochs, learning_rate, delta=0.0):
        if self.logging:
            print("Training has been started")

        for epoch in range(epochs):
            dataset_length = len(input_dataset)
            total_error = 0
            forward_time = 0
            backward_time = 0
            for i in range(dataset_length):

                start_time = time.perf_counter()
                actual_result = self.predict(input_dataset[i])
                end_time = time.perf_counter()
                forward_time += (end_time - start_time)

                target_result = validation_dataset[i]

                error = self.loss_function.function(target_result, actual_result)
                total_error += error
                output_gradient = self.loss_function.function_derivative(target_result, actual_result).reshape(1, -1)
                if error < delta:
                    continue

                start_time = time.perf_counter()
                for layer in reversed(self.layers):
                    output_gradient = layer.backward_propagation(output_gradient, learning_rate)
                end_time = time.perf_counter()
                backward_time += (end_time - start_time)

            total_error /= dataset_length
            forward_time /= dataset_length
            backward_time /= dataset_length
            self.errors = np.append(self.errors, total_error)
            if self.logging:
                print('epoch: %d/%d   error: %f   forward: %fs   backward: %fs'
                      % (epoch + 1, epochs, total_error, forward_time, backward_time))

    def save(self, file_name):
        serialization.serialize(file_name, self)

    def plot_error(self):
        plotter.plot_error(self.errors)
