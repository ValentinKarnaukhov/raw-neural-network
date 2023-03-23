from keras.datasets import mnist
from keras.utils import np_utils

from function.activation_functions import Tanh
from layer.flatten_layer import FlattenLayer
from layer.activation_layer import ActivationLayer
from layer.convolutional_layer import ConvolutionalLayer
from layer.fully_connected_layer import FullyConnectedLayer
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

network = NeuralNetwork()
network.logging = True

network.add_layer(ConvolutionalLayer((1, 3, 3), (28, 28, 1)))
network.add_layer(ActivationLayer(Tanh()))
network.add_layer(FlattenLayer())
network.add_layer(FullyConnectedLayer.with_random_weights(28 * 28 * 1, 100))
network.add_layer(ActivationLayer(Tanh()))
network.add_layer(FullyConnectedLayer.with_random_weights(100, 10))
network.add_layer(ActivationLayer(Tanh()))

network.train(x_train[0:1000], y_train[0:1000], 100, 0.1)

output = network.predict(x_test[0])
print(output)
