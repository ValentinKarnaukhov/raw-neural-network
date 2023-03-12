import numpy as np

from function.activation_functions import Sigmoid
from layer.activation_layer import ActivationLayer
from layer.fully_connected_layer import FullyConnectedLayer
from neural_network import NeuralNetwork

input_dataset = np.array([[0.05, 0.1]])
validation_dataset = np.array([[0.01, 0.99]])

weights_1 = np.array([[0.15, 0.25],
                      [0.20, 0.30]])
bias_1 = np.array([[0.35, 0.35]])

weights_2 = np.array([[0.40, 0.50],
                      [0.45, 0.55]])
bias_2 = np.array([[0.60, 0.60]])

network = NeuralNetwork()
network.logging = True

network.add_layer(FullyConnectedLayer.with_initial_weights(weights_1, bias_1))
network.add_layer(ActivationLayer(Sigmoid))
network.add_layer(FullyConnectedLayer.with_initial_weights(weights_2, bias_2))
network.add_layer(ActivationLayer(Sigmoid))

network.train(input_dataset, validation_dataset, 100000, 0.5)

for input_data_index in range(len(input_dataset)):
    print("prediction:", network.predict(input_dataset[input_data_index]))
    print("expected:  ", validation_dataset[input_data_index])
    print("\n")
