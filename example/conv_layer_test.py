import numpy as np
from layer.convolutional_layer import ConvolutionalLayer

layer = ConvolutionalLayer((1, 3, 3), (1, 5, 5))

print(layer.weights)
print(np.sum(layer.weights.reshape(1, -1)))

input_data = np.array([[1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1]])

output_data = layer.forward_propagation(input_data)
print(output_data)