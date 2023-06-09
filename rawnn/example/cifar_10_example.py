from keras.datasets import cifar10
from keras.utils import np_utils

from rawnn import plotter
from rawnn.function.activation_functions import Sigmoid
from rawnn.function.loss_functions import MeanSquareError
from rawnn.layer.flatten_layer import FlattenLayer
from rawnn.layer.activation_layer import ActivationLayer
from rawnn.layer.convolutional_layer import ConvolutionalLayer
from rawnn.layer.fully_connected_layer import FullyConnectedLayer
from rawnn.layer.pooling_layer import MaxPooling
from rawnn.neural_network import NeuralNetwork

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype("float32") / 255
y_train = np_utils.to_categorical(y_train)

x_test = x_test.astype("float32") / 255
y_test = np_utils.to_categorical(y_test)

network = NeuralNetwork(MeanSquareError)
network.logging = True

network.add_layer(ConvolutionalLayer((3, 3), 16, (32, 32, 3)))
network.add_layer(ActivationLayer(Sigmoid))
network.add_layer(ConvolutionalLayer((3, 3), 16, (32, 32, 16)))
network.add_layer(ActivationLayer(Sigmoid))
network.add_layer(MaxPooling((2, 2)))
network.add_layer(FlattenLayer())
network.add_layer(FullyConnectedLayer.with_random_weights(16 * 16 * 16, 512))
network.add_layer(ActivationLayer(Sigmoid))
network.add_layer(FullyConnectedLayer.with_random_weights(512, 128))
network.add_layer(ActivationLayer(Sigmoid))
network.add_layer(FullyConnectedLayer.with_random_weights(128, 10))
network.add_layer(ActivationLayer(Sigmoid))

network.train(x_train[0:2], y_train[0:2], 200, 0.5)

for i in range(10):
    print("prediction:", network.predict(x_test[i]).round())
    print("expected:  ", y_test[i])
    print("\n")

values_dictionary = {0: "airplane",
                     1: "automobile",
                     2: "bird",
                     3: "cat",
                     4: "deer",
                     5: "dog",
                     6: "frog",
                     7: "horse",
                     8: "ship",
                     9: "truck"}
plotter.plot_predictions(x_train[0:10], network.predict_bulk(x_train[0:10]), values_dictionary)
