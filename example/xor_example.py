import numpy as np

from function.activation_functions import Sigmoid
from function.loss_functions import MeanSquareError
from layer.activation_layer import ActivationLayer
from layer.fully_connected_layer import FullyConnectedLayer
from neural_network import NeuralNetwork

input_dataset = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
validation_dataset = np.array([[0], [1], [1], [0]])

network = NeuralNetwork(MeanSquareError)
network.logging = True

network.add_layer(FullyConnectedLayer.with_random_weights(2, 3))
network.add_layer(ActivationLayer(Sigmoid))
network.add_layer(FullyConnectedLayer.with_random_weights(3, 1))
network.add_layer(ActivationLayer(Sigmoid))

network.train(input_dataset, validation_dataset, 10000, 0.5)
network.plot_error()

for input_data_index in range(len(input_dataset)):
    print("prediction:", network.predict(input_dataset[input_data_index]).round())
    print("expected:  ", validation_dataset[input_data_index])
    print("\n")
