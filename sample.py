import numpy as np

from activation_functions import sigmoid, sigmoid_derivative
from scaler_factory import min_max_scaler
from plotter import plot_error
from activation_layer import ActivationLayer
from fully_connected_layer import FullyConnectedLayer
from neural_network import NeuralNetwork

input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
validation_data = np.array([[0], [1], [1], [0]])

# input_data = np.empty((0, 1))
# validation_data = np.empty((0, 1))
#
# for i in np.arange(0, 10, 0.1):
#     input_data = np.append(input_data, np.array([[i]]), axis=0)
#     validation_data = np.append(validation_data, np.array([[i ** 2]]), axis=0)

min_max_scaler = min_max_scaler(-1, 1, (0, 1))

network = NeuralNetwork(min_max_scaler)
network.logging = True

network.add_layer(FullyConnectedLayer(2, 3))
network.add_layer(ActivationLayer(sigmoid, sigmoid_derivative))
network.add_layer(FullyConnectedLayer(3, 1))
network.add_layer(ActivationLayer(sigmoid, sigmoid_derivative))

network.train(input_data, validation_data, 100000, 0.5)

for input in input_data:
    print("Input:", input, "Output:", network.predict(input))

# plot_error(network.errors)
