from keras.datasets import fashion_mnist
from keras.utils import np_utils

from rawnn import plotter
from rawnn.function import activation_functions
from rawnn.function.loss_functions import MeanSquareError
from rawnn.layer.activation_layer import ActivationLayer
from rawnn.layer.fully_connected_layer import FullyConnectedLayer
from rawnn.neural_network import NeuralNetwork

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28 * 28).astype("float32") / 255
y_train = np_utils.to_categorical(y_train)

x_test = x_test.reshape(x_test.shape[0], 28 * 28).astype("float32") / 255
y_test = np_utils.to_categorical(y_test)

network = NeuralNetwork(MeanSquareError)
network.logging = True

network.add_layer(FullyConnectedLayer.with_random_weights(28 * 28, 100))
network.add_layer(ActivationLayer(activation_functions.Sigmoid))
network.add_layer(FullyConnectedLayer.with_random_weights(100, 50))
network.add_layer(ActivationLayer(activation_functions.Sigmoid))
network.add_layer(FullyConnectedLayer.with_random_weights(50, 10))
network.add_layer(ActivationLayer(activation_functions.Sigmoid))

network.train(x_train[0:1000], y_train[0:1000], 50, 0.1)

for i in range(10):
    print("prediction:", network.predict(x_test[i]).round())
    print("expected:  ", y_test[i])
    print("\n")

values_dictionary = {0: "T - shirt / top",
                     1: "Trouser",
                     2: "Pullover",
                     3: "Dress",
                     4: "Coat",
                     5: "Sandal",
                     6: "Shirt",
                     7: "Sneaker",
                     8: "Bag",
                     9: "Ankle boot"}
plotter.plot_predictions(x_test[0:10].reshape(10, 28, 28, 1), network.predict_bulk(x_test[0:10]), values_dictionary)
