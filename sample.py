import numpy as np

from activation_functions import sigmoid, sigmoid_derivative
from activation_layer import ActivationLayer
from fully_connected_layer import FullyConnectedLayer
from neural_network import NeuralNetwork

first_layer = np.array([[0.15, 0.25],
                        [0.20, 0.30]])

second_layer = np.array([[0.40, 0.50],
                         [0.45, 0.55]])

first_bias = 0.35
second_bias = 0.60

# network = NeuralNetwork()
# network.add_layer(FullyConnectedLayer(first_layer, first_bias))
# network.add_layer(ActivationLayer(sigmoid, sigmoid_derivative))
# network.add_layer(FullyConnectedLayer(second_layer, second_bias))
# network.add_layer(ActivationLayer(sigmoid, sigmoid_derivative))

network = NeuralNetwork()
network.add_layer(FullyConnectedLayer(2, 4))
network.add_layer(ActivationLayer(sigmoid, sigmoid_derivative))
network.add_layer(FullyConnectedLayer(4, 4))
network.add_layer(ActivationLayer(sigmoid, sigmoid_derivative))
network.add_layer(FullyConnectedLayer(4, 2))
network.add_layer(ActivationLayer(sigmoid, sigmoid_derivative))

input_data = np.array([[[1, 2]], [[0.2, 0.3]], [[0.3, 0.4]], [[0.4, 0.5]]])
target_data = np.array([[[2, 1]], [[0.3, 0.2]], [[0.4, 0.3]], [[0.5, 0.4]]])

network.train(input_data, target_data, 1000, 0.5)

for dataset in input_data:
    print(network.predict(dataset))

print(network.predict([[0.9, 0.4]]))
