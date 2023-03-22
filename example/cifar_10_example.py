from keras.datasets import cifar10
from keras.utils import np_utils

from function.activation_functions import Tanh
from layer.flatten_layer import FlattenLayer
from layer.activation_layer import ActivationLayer
from layer.convolutional_layer import ConvolutionalLayer
from layer.fully_connected_layer import FullyConnectedLayer
from layer.pooling_layer import MaxPooling
from neural_network import NeuralNetwork

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype("float32") / 255
y_train = np_utils.to_categorical(y_train)

x_test = x_test.astype("float32") / 255
y_test = np_utils.to_categorical(y_test)

network = NeuralNetwork()
network.logging = True

network.add_layer(ConvolutionalLayer((32, 3, 3), (32, 32, 3)))
network.add_layer(ActivationLayer(Tanh()))
network.add_layer(ConvolutionalLayer((32, 3, 3), (32, 32, 32)))
network.add_layer(ActivationLayer(Tanh()))
network.add_layer(MaxPooling((2, 2)))
network.add_layer(ConvolutionalLayer((64, 3, 3), (16, 16, 32)))
network.add_layer(ActivationLayer(Tanh()))
network.add_layer(ConvolutionalLayer((64, 3, 3), (16, 16, 64)))
network.add_layer(ActivationLayer(Tanh()))
network.add_layer(MaxPooling((2, 2)))
network.add_layer(FlattenLayer())
network.add_layer(FullyConnectedLayer.with_random_weights(4096, 512))
network.add_layer(ActivationLayer(Tanh()))
network.add_layer(FullyConnectedLayer.with_random_weights(512, 10))
network.add_layer(ActivationLayer(Tanh()))

network.train(x_train[0:1], y_train[0:1], 1, 0.5)

output = network.predict(x_test[0])
print(output)
