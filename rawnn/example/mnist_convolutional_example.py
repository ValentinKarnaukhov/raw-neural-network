from keras.datasets import mnist
from keras.utils import np_utils

from rawnn import plotter
from rawnn.function.activation_functions import Tanh, Sigmoid
from rawnn.function.loss_functions import MeanSquareError
from rawnn.layer.flatten_layer import FlattenLayer
from rawnn.layer.activation_layer import ActivationLayer
from rawnn.layer.convolutional_layer import ConvolutionalLayer
from rawnn.layer.fully_connected_layer import FullyConnectedLayer
from rawnn.layer.pooling_layer import MaxPooling
from rawnn.neural_network import NeuralNetwork

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')/255
y_train = np_utils.to_categorical(y_train)

x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')/255
y_test = np_utils.to_categorical(y_test)

network = NeuralNetwork(MeanSquareError)
network.logging = True

network.add_layer(ConvolutionalLayer((3, 3), 8, (28, 28, 1)))
network.add_layer(ActivationLayer(Sigmoid))
network.add_layer(MaxPooling((2, 2)))
network.add_layer(FlattenLayer())
network.add_layer(FullyConnectedLayer.with_random_weights(14 * 14 * 8, 50))
network.add_layer(ActivationLayer(Tanh))
network.add_layer(FullyConnectedLayer.with_random_weights(50, 10))
network.add_layer(ActivationLayer(Tanh))

network.train(x_train[0:1000], y_train[0:1000], 50, 0.1)

for i in range(10):
    print("prediction:", network.predict(x_test[i]).round())
    print("expected:  ", y_test[i])
    print("\n")

plotter.plot_predictions(x_test[0:10].reshape(-1, 28, 28), network.predict_bulk(x_test[0:10]))
