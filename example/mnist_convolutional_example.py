import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils

import plotter
from function.activation_functions import Tanh, Sigmoid
from layer.flatten_layer import FlattenLayer
from layer.activation_layer import ActivationLayer
from layer.convolutional_layer import ConvolutionalLayer
from layer.fully_connected_layer import FullyConnectedLayer
from layer.pooling_layer import MaxPooling
from neural_network import NeuralNetwork

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_train = x_train.astype('float32')
x_train /= 255

y_train = np_utils.to_categorical(y_train)

x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_test = x_test.astype('float32')
x_test /= 255
y_test = np_utils.to_categorical(y_test)

# network = NeuralNetwork.from_file("convolutional_mnist.d")


network = NeuralNetwork()
network.logging = True

network.add_layer(ConvolutionalLayer((2, 3, 3), (28, 28, 1)))
network.add_layer(ActivationLayer(Sigmoid()))
network.add_layer(MaxPooling((2, 2)))
network.add_layer(FlattenLayer())
network.add_layer(FullyConnectedLayer.with_random_weights(14 * 14 * 2, 100))
network.add_layer(ActivationLayer(Tanh()))
network.add_layer(FullyConnectedLayer.with_random_weights(100, 10))
network.add_layer(ActivationLayer(Tanh()))

network.train(x_train[0:2], y_train[0:2], 1000, 0.1)

features = network.layers[0].output
plotter.plot_digit(features, np.zeros((features.shape[-1], 10)))

# network.save("convolutional_mnist.d")

# for i in range(10):
#     print("prediction:", network.predict(x_test[i]).round())
#     print("expected:  ", y_test[i])
#     print("\n")
#
# plotter.plot_digit(x_test[15:30].reshape(-1, 28, 28), network.predict_bulk(x_test[15:30]))
