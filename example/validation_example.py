import numpy as np

from function.activation_functions import Sigmoid
from function.loss_functions import MeanSquareError
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

network = NeuralNetwork(MeanSquareError)
network.logging = True

network.add_layer(FullyConnectedLayer.with_initial_weights(weights_1, bias_1))
network.add_layer(ActivationLayer(Sigmoid))
network.add_layer(FullyConnectedLayer.with_initial_weights(weights_2, bias_2))
network.add_layer(ActivationLayer(Sigmoid))

print("validation prediction:", network.predict(input_dataset[0]))
print("expected result:", "[0.75136507, 0.77292847]")

network.train(input_dataset, validation_dataset, 1, 0.5)
new_weights = np.concatenate((network.layers[0].weights, network.layers[2].weights))

print("calculated new weights:", new_weights)
print("expected new weights:", "[[0.14978072 0.24975114]\n[0.19956143 0.29950229]\n"
                               "[0.35891648 0.51130127]\n[0.40866619 0.56137012]]")
