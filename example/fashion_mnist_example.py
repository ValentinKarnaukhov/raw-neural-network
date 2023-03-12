from keras.datasets import fashion_mnist
from keras.utils import np_utils

import plotter
from function import activation_functions
from layer.activation_layer import ActivationLayer
from layer.fully_connected_layer import FullyConnectedLayer
from neural_network import NeuralNetwork

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28 * 28).astype("float32") / 255
y_train = np_utils.to_categorical(y_train)

x_test = x_test.reshape(x_test.shape[0], 28 * 28).astype("float32") / 255
y_test = np_utils.to_categorical(y_test)

network = NeuralNetwork.from_file("fashion_mnist.d")
# network.logging = True
#
# network.add_layer(FullyConnectedLayer.with_random_weights(28 * 28, 100))
# network.add_layer(ActivationLayer(activation_functions.Sigmoid))
# network.add_layer(FullyConnectedLayer.with_random_weights(100, 50))
# network.add_layer(ActivationLayer(activation_functions.Sigmoid))
# network.add_layer(FullyConnectedLayer.with_random_weights(50, 10))
# network.add_layer(ActivationLayer(activation_functions.Sigmoid))
#
# network.train(x_train[0:1000], y_train[0:1000], 50, 0.1)

# network.save("fashion_mnist.d")

for i in range(10):
    print("prediction:", network.predict(x_test[i]).round())
    print("expected:  ", y_test[i])
    print("\n")

plotter.plot_digit(x_test[15:30].reshape(-1, 28, 28), network.predict_bulk(x_test[15:30]))
