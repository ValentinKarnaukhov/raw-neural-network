import numpy as np

from activation_functions import sigmoid, sigmoid_derivative
from activation_layer import ActivationLayer
from fully_connected_layer import FullyConnectedLayer
from neural_network import NeuralNetwork
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

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

# input_data = np.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]])
# target_data = np.array([[0.2, 0.1], [0.3, 0.2], [0.4, 0.3], [0.5, 0.4]])

input_data = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
target_data = np.array([[2, 1], [3, 2], [4, 3], [5, 4]])

# input_data = np.array([[0.05, 0.10]])
# target_data = np.array([[0.01, 0.99]])

min_max_scaler = MinMaxScaler()

flatten_input_data = input_data.flatten()
flatten_target_data = target_data.flatten()

reshaped_input_data = flatten_input_data.reshape((len(flatten_input_data), 1))
reshaped_target_data = flatten_target_data.reshape((len(flatten_target_data), 1))

min_max_scaler.partial_fit(reshaped_input_data)
min_max_scaler.partial_fit(reshaped_target_data)

network = NeuralNetwork(min_max_scaler)
network.add_layer(FullyConnectedLayer(2, 10))
network.add_layer(ActivationLayer(sigmoid, sigmoid_derivative))
network.add_layer(FullyConnectedLayer(10, 10))
network.add_layer(ActivationLayer(sigmoid, sigmoid_derivative))
network.add_layer(FullyConnectedLayer(10, 2))
network.add_layer(ActivationLayer(sigmoid, sigmoid_derivative))

network.train(input_data, target_data, 10000, 0.5)

for dataset in input_data:
    print(network.predict(dataset))

print(network.predict(np.array([1, 4])))
print(network.predict(np.array([4, 1])))
